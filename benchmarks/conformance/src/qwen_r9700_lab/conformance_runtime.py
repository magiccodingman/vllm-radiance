"""Launch a pinned, separate vLLM API process for production-path qualification.

Run in the prepared Radiance Python environment/container. This module never
builds, installs, patches or restarts the production service. All mutable server
paths and IPC metadata are redirected to this campaign's new private directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import shutil
import socket
import sys
from pathlib import Path

from qwen_r9700_lab.conformance_shm import OwnedOffloadRegion
from qwen_r9700_lab.conformance_transport import OwnedClient, OwnedProcess
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, private_json, write_private

MODULES = (
    "vllm.v1.worker.gpu.model_runner",
    "qwen_radiance_fair_scheduler",
    "qwen_radiance_chat_tier",
    "qwen_radiance_cache",
    "radiance_verifyhead",
    "vllm.parser.qwen3",
    "vllm.parser.engine.parser_engine",
    "vllm.parser.engine.adapters",
    "vllm.parser.parser_manager",
)

SHUTDOWN_TIMEOUT_SECONDS = 60
SHUTDOWN_CLEANUP_SECONDS = 15


def isolated_config(config, root, nonce, port, *, graphs, speculation, asynchronous=False):
    config = copy.deepcopy(config)
    shutdown = config.setdefault("shutdown_timeout", SHUTDOWN_TIMEOUT_SECONDS)
    if isinstance(shutdown, bool) or not isinstance(shutdown, int) or shutdown <= 0:
        raise DiagnosticError("persistent-cache qualification requires a positive shutdown timeout")
    if not isinstance(config.get("model"), str) or not Path(config["model"]).is_absolute():
        raise DiagnosticError("qualification requires an explicit local checkpoint")
    if config.get("tensor_parallel_size", 1) != 1 or config.get("pipeline_parallel_size", 1) != 1:
        raise DiagnosticError("the qualification server admits TP1/PP1 only")
    if config.get("data_parallel_size", 1) != 1:
        raise DiagnosticError("distributed qualification is not implemented")
    config.update(
        host="127.0.0.1",
        port=port,
        api_key=nonce,
        max_num_seqs=2,
        async_scheduling=asynchronous,
        enforce_eager=not graphs,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        middleware=[
            "qwen_r9700_lab.conformance_runtime.identity_middleware",
            "qwen_radiance_request_guard.require_snapshot_abi",
        ],
    )
    if not speculation:
        config.pop("speculative_config", None)
    elif not isinstance(config.get("speculative_config"), dict):
        raise DiagnosticError("natural speculation requires the pinned drafter configuration")
    if graphs:
        config["compilation_config"] = {
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        }
    else:
        config.pop("compilation_config", None)
    # Async vLLM is a different, explicitly experimental scheduler domain. The
    # production FairScheduler rejects it; never imply that it was exercised.
    if asynchronous:
        config.pop("scheduler_cls", None)
        config.pop("additional_config", None)
        config.pop("kv_transfer_config", None)
    else:
        config["scheduler_cls"] = "qwen_radiance_fair_scheduler.FairScheduler"
        additional = config.setdefault("additional_config", {})
        fair = additional.setdefault("qwen_fair", {})
        fair.update(status_path=str(root / "fair"), max_cached_chats=2, policy="response_boundary")
        transfer = config.get("kv_transfer_config")
        if not isinstance(transfer, dict) or transfer.get("kv_connector") != "OffloadingConnector":
            raise DiagnosticError("production qualification requires the real snapshot connector")
        transfer.update(
            engine_id="conformance-" + nonce, kv_role="kv_both", kv_load_failure_policy="fail"
        )
        extra = transfer["kv_connector_extra_config"]
        tiers = extra.get("secondary_tiers", [])
        if len(tiers) != 1 or tiers[0].get("type") != "qwen_chat_fs":
            raise DiagnosticError("unreviewed qualification snapshot tier")
        tiers[0].update(
            root_dir=str(root / "data"),
            control_directory=str(root / "control"),
            tail_status_path=str(root / "tail.json"),
        )
    return config


def server_argv(config):
    argv = ["vllm.entrypoints.openai.api_server"]
    for key, value in config.items():
        if not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise DiagnosticError("invalid server argument name")
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            # vLLM's BooleanOptionalAction is used for async scheduling. Other
            # disabled switches must simply be omitted (e.g. enforce_eager).
            if value:
                argv.append(flag)
            elif key == "async_scheduling":
                argv.append("--no-async-scheduling")
        elif value is None:
            continue
        elif isinstance(value, dict):
            argv.extend((flag, json.dumps(value, separators=(",", ":"))))
        elif isinstance(value, list):
            if key == "middleware":
                for item in value:
                    argv.extend((flag, str(item)))
            else:
                argv.extend((flag, *(str(item) for item in value)))
        else:
            argv.extend((flag, str(value)))
    return argv


def worker_environment(spec, root):
    env = dict(os.environ)
    env.update(spec["environment"])
    for key, directory in {
        "VLLM_CACHE_ROOT": "vllm",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "XDG_CACHE_HOME": "xdg-cache",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        path = root / "runtime" / directory
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        env[key] = str(path)
    # AITER_ROOT_DIR includes its source/JIT tree, not just an empty cache.
    # Clone a configured tree before allowing the qualifier to compile in it.
    if env.get("AITER_ROOT_DIR"):
        aiter_source = Path(env["AITER_ROOT_DIR"]).resolve()
        aiter_copy = root / "runtime/aiter"
        if not aiter_copy.exists():
            shutil.copytree(aiter_source, aiter_copy)
        env["AITER_ROOT_DIR"] = str(aiter_copy)
    env.update(
        QWEN_CONFORMANCE_GPU="1",
        QWEN_RADIANCE_CACHE_ABI=spec["binding"]["live_data_abi"],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
    )
    # Source is explicitly bound in the campaign. Do not rely on whichever
    # editable installation an unrelated shell happens to have activated.
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    return env


class NativeServer:
    def __init__(
        self,
        spec,
        root,
        *,
        allow_gpu=False,
        graphs=True,
        speculation=True,
        head=True,
        head_audit=False,
        asynchronous=False,
        observe=True,
        dynamic_width=False,
        head_fault=False,
        primary_cache_bytes=None,
    ):
        if not allow_gpu:
            raise DiagnosticError("GPU use was not authorized")
        interpreter = Path(spec["python"])
        if hashlib.sha256(interpreter.read_bytes()).hexdigest() != spec["python_sha256"]:
            raise DiagnosticError("qualification Python artifact changed")
        self.spec, self.root = spec, Path(root).resolve()
        self.root.mkdir(mode=0o700)
        self.variant = {
            "graphs": graphs,
            "speculation": speculation,
            "head": head,
            "head_audit": head_audit,
            "asynchronous": asynchronous,
            "observe": observe,
            "dynamic_width": dynamic_width,
            "head_fault": head_fault,
            "primary_cache_bytes": primary_cache_bytes,
        }
        self.process, self.client, self.incarnation = None, None, 0
        self.offload_region = None
        self.offload_receipt = None
        self.executions = []
        self.shutdown_grace_seconds = SHUTDOWN_TIMEOUT_SECONDS + SHUTDOWN_CLEANUP_SECONDS

    def start(self):
        if self.process is not None:
            raise DiagnosticError("qualification server is already running")
        nonce = secrets.token_hex(32)
        reservation = socket.socket()
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        config = isolated_config(
            self.spec["server_config"],
            self.root,
            nonce,
            port,
            **{k: self.variant[k] for k in ("graphs", "speculation", "asynchronous")},
        )
        self.shutdown_grace_seconds = config["shutdown_timeout"] + SHUTDOWN_CLEANUP_SECONDS
        capacity = self.variant["primary_cache_bytes"]
        if capacity is not None:
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                reservation.close()
                raise DiagnosticError("invalid qualification primary-cache capacity")
            if self.variant["asynchronous"]:
                reservation.close()
                raise DiagnosticError("async qualification has no primary snapshot cache")
            extra = config["kv_transfer_config"]["kv_connector_extra_config"]
            original = extra["cpu_bytes_to_use"]
            if capacity >= original:
                reservation.close()
                raise DiagnosticError("eviction qualification must reduce primary-cache capacity")
            extra["cpu_bytes_to_use"] = capacity
        sources = self.spec["observer_sources"] if self.variant["observe"] else {}
        if self.variant["observe"] and set(sources) != set(MODULES):
            reservation.close()
            raise DiagnosticError("qualification observer source inventory is incomplete")
        settings = {
            "root": str(self.root),
            "nonce": nonce,
            "config": config,
            "observer_sources": sources,
            "head_audit": self.variant["head_audit"],
            "head_fault": self.variant["head_fault"],
            "binding": self.spec["binding"],
            "variant": self.variant,
            "incarnation": self.incarnation,
        }
        settings["execution"] = digest(settings)
        settings_path = self.root / f"server-{self.incarnation}.json"
        write_private(settings_path, settings)
        env = worker_environment(self.spec, self.root)
        env.update(
            QWEN_CONFORMANCE_SERVER_SETTINGS=str(settings_path),
            RADIANCE_VERIFY_HEAD="1" if self.variant["head"] else "0",
            RADIANCE_DYNAMIC_WIDTH="1" if self.variant["dynamic_width"] else "0",
        )
        overlay = self.root / f"overlay-{self.incarnation}"
        overlay.mkdir(mode=0o700)
        if self.variant["observe"]:
            # Python normally swallows sitecustomize exceptions. Exit explicitly
            # on a hook/bootstrap failure so an unobserved run cannot proceed.
            code = (
                "import os, traceback\ntry:\n"
                " from qwen_r9700_lab.conformance_observer import install_from_environment\n"
                " install_from_environment()\nexcept BaseException:\n"
                " traceback.print_exc()\n os._exit(87)\n"
            )
            (overlay / "sitecustomize.py").write_text(code)
            env["PYTHONPATH"] = str(overlay) + os.pathsep + env["PYTHONPATH"]
        argv = [self.spec["python"], "-m", "qwen_r9700_lab.conformance_runtime", str(settings_path)]
        reservation.close()
        try:
            transfer = config.get("kv_transfer_config")
            if transfer is not None:
                self.offload_region = OwnedOffloadRegion(transfer["engine_id"])
                self.offload_receipt = self.root / f"shared-memory-{self.incarnation}.json"
            self.process = OwnedProcess(
                argv,
                self.root / f"process-{self.incarnation}",
                env=env,
                timeout=self.spec["case_timeout_seconds"],
            )
            self.client = OwnedClient(self.process, port, nonce, settings["execution"])
            self.client.connect(timeout=self.spec["startup_timeout_seconds"])
        except BaseException:
            self.stop()
            raise
        self.settings = settings
        self.executions.append(settings["execution"])
        self.incarnation += 1
        return self

    def stop(self, *, crash=False):
        if self.process is not None:
            self.process.close(crash=crash, grace_seconds=self.shutdown_grace_seconds)
            self.process = None
        self.client = None
        if self.offload_region is not None:
            region, self.offload_region = self.offload_region, None
            write_private(self.offload_receipt, region.release())

    def restart(self, *, crash=False):
        self.stop(crash=crash)
        return self.start()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


async def identity_middleware(request, call_next):
    if request.url.path != "/qwen-conformance/identity":
        return await call_next(request)
    from starlette.responses import JSONResponse

    settings = private_json(Path(os.environ["QWEN_CONFORMANCE_SERVER_SETTINGS"]))
    if request.headers.get("authorization") != "Bearer " + settings["nonce"]:
        return JSONResponse({"error": "unauthorized qualification client"}, status_code=403)
    return JSONResponse(
        {"nonce": settings["nonce"], "execution": settings["execution"], "pid": os.getpid()}
    )


def main():
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("qualification server is not armed")
    settings = private_json(Path(sys.argv[1]))
    import importlib.util
    import runpy

    from qwen_r9700_lab.conformance_radiance import verify_sources

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise DiagnosticError("pinned Radiance package is not installed")
    verify_sources(Path(spec.origin).parent.parent, settings["binding"])
    sys.argv = server_argv(settings["config"])
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
