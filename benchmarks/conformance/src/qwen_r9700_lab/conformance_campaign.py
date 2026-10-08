"""Executable GPU campaign inventory, orchestration and an incomplete-coverage gate.

The declared matrix is finite. TESTED means the named oracle passed for these
inputs/artifacts; it never means PROVED. Every unselected, missing, failed,
timed-out or unsupported case remains visible in the complete report.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from qwen_r9700_lab.conformance_faults import FAULTS
from qwen_r9700_lab.conformance_runtime import worker_environment
from qwen_r9700_lab.conformance_transport import OwnedProcess
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)

SCHEMA = "urn:qwen:conformance-campaign:v1"
RESULT_SCHEMA = "urn:qwen:conformance-case-result:v1"
REPO = Path(__file__).resolve().parents[2]
FAMILIES = {
    "native_fault": "Actual device corruption detected against a clean native capture",
    "forced_reference": "Independent prefill and serial reference state at every selected boundary",
    "forced_d7": "Native M1 and forced D7 agree on aligned logical state and logits",
    "rejected_suffix": "Rejected proposals do not affect retained state or output",
    "operator": "Independent operator numerical oracle with fixed diagnostic tolerances",
    "natural": "Unforced greedy continuation across actual runtime configurations",
    "lifecycle": "Continuation and durable-head invariants through actual cache lifecycle",
    "priority": "Actual scheduler ownership and unchanged continuation through answer priorities",
    "protocol": "Actual parser/provider tool boundaries and fragmentation invariants",
}


def case_priority(case):
    family, variant = case["family"], case["variant"]
    n, seed = case["context"], case["seed"]
    if family == "native_fault":
        rank = {
            "gdn": 0,
            "conv": 1,
            "kv": 2,
            "pending": 3,
            "position": 4,
            "version": 5,
            "missing_observation": 6,
        }[variant]
        return (0, rank, n, seed, variant)
    if family == "operator" and variant == "native_dispatch":
        return (0, 7, n, seed, variant)
    focused = (
        family == "operator"
        or (family == "forced_reference" and n in {16, 129} and seed == 0)
        or (
            family == "forced_d7"
            and n in {129, 60000}
            and case["axes"]["accepted"] in {0, 1, 3, 7}
            and seed == 0
        )
        or (family == "rejected_suffix" and case["axes"]["accepted"] in {0, 3, 6} and seed == 0)
        or (
            family == "natural"
            and n in {129, 60000}
            and seed == 0
            and variant in {"speculation", "verify_head", "graphs", "head_omission_control"}
        )
        or (family == "lifecycle" and n == 8192)
        or (family == "priority" and n == 8192)
        or family == "protocol"
    )
    order = {
        "operator": 0,
        "forced_reference": 1,
        "forced_d7": 2,
        "rejected_suffix": 3,
        "natural": 4,
        "lifecycle": 5,
        "priority": 6,
        "protocol": 7,
    }
    return (1 if focused else 2, order[family], n, seed, variant)


def prepare_spec(
    *,
    python,
    site_packages,
    binding,
    operator_profile,
    native_config,
    server_config,
    reference_metadata,
    reference_profile="weight-only-bf16",
    pi_provider=None,
):
    """Pin the installed qualification environment without importing GPU code.

    Observers are pinned as candidate code, not thereby proved correct. The
    pre-reviewed native binding must already match before a spec can be made.
    This accepts model metadata only, never a conversation/replay transcript.
    """
    from qwen_r9700_lab.conformance_radiance import verify_sources
    from qwen_r9700_lab.conformance_runtime import MODULES

    if set(reference_metadata) != {"checkpoint_files", "kv_scales"}:
        raise DiagnosticError("suite preparation accepts model identities/scales only")
    package = Path(site_packages).resolve()
    verify_sources(package, binding)
    binding = reviewed_dispatch_binding(binding, operator_profile, reference_profile)
    sources = {}
    for module in MODULES:
        path = package / (module.replace(".", "/") + ".py")
        sources[module] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = {
        "python": str(Path(python).absolute()),
        "python_sha256": hashlib.sha256(Path(python).read_bytes()).hexdigest(),
        "binding": binding,
        "operator_profile": operator_profile,
        "observer_sources": sources,
        "environment": operator_profile["kernel_environment"],
        "native_config": native_config,
        "server_config": server_config,
        **reference_metadata,
        "reference_profile": reference_profile,
        "probe_manifest": {},
    }
    if pi_provider is not None:
        path = Path(pi_provider).resolve()
        spec.update(
            pi_runtime=str(path), pi_runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return validate_spec(spec)


def reviewed_dispatch_binding(binding, operator_profile, reference_profile):
    """Reuse the existing reviewed R4D export catalog, only for its exact binary."""
    import ast

    from qwen_r9700_lab.conformance_reference import reference_precision

    path = REPO / "experiments/radiance-public/r4d_dispatch_audit.py"
    values = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in {"NATIVE_SHA256", "GPU_EXPORTS", "METADATA_EXPORTS"}:
                values[name] = ast.literal_eval(node.value)
    if operator_profile.get("kernel_hashes", {}).get("r4d.so") != values["NATIVE_SHA256"]:
        return binding  # required native_dispatch case remains explicitly unsupported
    if binding.get("native_entrypoints"):
        return binding
    dtype = "fp8" if reference_precision(reference_profile)["kv_fp8"] else "bf16"
    entry = seal(
        {
            "schema": "urn:qwen:native-entrypoint-binding:v1",
            "module": "r4d",
            "library_sha256": values["NATIVE_SHA256"],
            "exports": {
                **dict.fromkeys(sorted(values["GPU_EXPORTS"]), "kernel"),
                **dict.fromkeys(sorted(values["METADATA_EXPORTS"]), "metadata"),
            },
            "required": [f"attn_decode_h256_gqa6_{dtype}kv"],
        }
    )
    return seal(
        {
            **{k: v for k, v in binding.items() if k != "sha256"},
            "native_entrypoints": [
                {
                    "binding": entry,
                    "aliases": ["radiance_gdn", "radiance_r4d_attn"],
                }
            ],
        }
    )


def source_identity():
    paths = sorted((REPO / "src/qwen_r9700_lab").glob("conformance_*.py"))
    paths += sorted((REPO / "tests").glob("conformance_*driver.*"))
    paths += [
        REPO / "src/qwen_r9700_lab" / name
        for name in (
            "diagnostic_contract.py",
            "radiance_cache.py",
            "ordered_reference_linear.py",
            "ordered_reference_linear.c",
            "exact_fp8_metrics.py",
        )
    ]
    paths += [
        REPO / "experiments/radiance-public" / name
        for name in (
            "probe_mxfp4_numerics.py",
            "probe_gdn_numerics.py",
            "probe_norm_rope_numerics.py",
            "probe_r4d_attention_numerics.py",
            "probe_dflash_sampling_rng.py",
            "r4d_dispatch_audit.py",
        )
    ]
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def validate_spec(value):
    required = {
        "python",
        "python_sha256",
        "binding",
        "observer_sources",
        "environment",
        "native_config",
        "server_config",
        "checkpoint_files",
        "kv_scales",
        "reference_profile",
        "operator_profile",
        "probe_manifest",
    }
    optional = {
        "contexts",
        "seeds",
        "max_context",
        "case_timeout_seconds",
        "startup_timeout_seconds",
        "output_tokens",
        "pi_runtime",
        "pi_runtime_sha256",
        "native_call_mode",
        "reference_linear",
        "reuse_serial_reference",
    }
    if not isinstance(value, dict) or required - set(value) or set(value) - required - optional:
        raise DiagnosticError("campaign spec has missing or unknown fields")
    value = json.loads(json.dumps(value, allow_nan=False))
    if "reuse_serial_reference" in value and type(value["reuse_serial_reference"]) is not bool:
        raise DiagnosticError("serial reference reuse must be explicitly boolean")
    if "reference_linear" in value:
        from qwen_r9700_lab.ordered_reference_linear import validate_binding

        validate_binding(value["reference_linear"])
    value.setdefault("native_call_mode", "metadata")
    if value["native_call_mode"] not in {"metadata", "tensor"}:
        raise DiagnosticError("invalid native semantic-call capture mode")
    if not Path(value["python"]).is_absolute():
        raise DiagnosticError("campaign Python must be an absolute pinned path")
    authenticate(value["binding"])
    if value["native_config"].get("model") != value["server_config"].get("model"):
        raise DiagnosticError("diagnostic and production configurations name different checkpoints")
    for key, default in (
        ("max_context", 253792),
        ("case_timeout_seconds", 14400),
        ("startup_timeout_seconds", 900),
        ("output_tokens", 256),
    ):
        value.setdefault(key, default)
        if type(value[key]) is not int or value[key] <= 0:
            raise DiagnosticError("invalid positive campaign parameter: " + key)
    if value["output_tokens"] < 16 or value["max_context"] <= value["output_tokens"] + 128:
        raise DiagnosticError("campaign must reserve enough context for continuation checks")
    if value["startup_timeout_seconds"] >= value["case_timeout_seconds"]:
        raise DiagnosticError("startup deadline must be shorter than the case deadline")
    value.setdefault("seeds", [0, 17])
    if (
        not value["seeds"]
        or len(set(value["seeds"])) != len(value["seeds"])
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in value["seeds"])
    ):
        raise DiagnosticError("campaign seeds must be explicit unique nonnegative integers")
    default_contexts = [
        8192,
        32768,
        60000,
        128000,
        200000,
        value["max_context"] - value["output_tokens"] - 1,
    ]
    value.setdefault(
        "contexts",
        sorted(
            {
                n
                for n in default_contexts
                if n > 0 and n + value["output_tokens"] < value["max_context"]
            }
        ),
    )
    if (
        not value["contexts"]
        or len(set(value["contexts"])) != len(value["contexts"])
        or any(
            type(n) is not int or n < 16 or n + value["output_tokens"] >= value["max_context"]
            for n in value["contexts"]
        )
    ):
        raise DiagnosticError(
            "context matrix is empty, duplicated or exceeds the configured window"
        )
    if not isinstance(value["environment"], dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or "\0" in k + v
        for k, v in value["environment"].items()
    ):
        raise DiagnosticError("campaign environment must contain string settings")
    # No production destinations or externally supplied hooks from a caller's
    # shell may redirect qualification into another session.
    if any(k.startswith("QWEN_CONFORMANCE_") or k == "PYTHONPATH" for k in value["environment"]):
        raise DiagnosticError("campaign environment cannot replace qualification wiring")
    return value


def build_campaign(spec):
    spec = validate_spec(spec)
    cases = []

    def add(family, variant, context=64, seed=0, **axes):
        identity = f"{family}.{variant}.ctx{context}.seed{seed}"
        cases.append(
            {
                "id": identity,
                "family": family,
                "variant": variant,
                "context": context,
                "seed": seed,
                "axes": axes,
                "oracle": FAMILIES[family],
            }
        )

    for fault in FAULTS:
        add("native_fault", fault)
    # Independent serial CPU reference is deliberately limited to short native
    # runs. Long contexts have separate self-consistency and operator oracles.
    for context in (16, 63, 64, 65, 127, 128, 129):
        for seed in spec["seeds"]:
            add("forced_reference", "independent", context, seed)
    boundaries = (63, 64, 65, 127, 128, 129, 1647, 1648, 1649, 2047, 2048, 2049, 8191, 8192, 8193)
    for context in sorted(
        set(
            spec["contexts"]
            + [n for n in boundaries if n + spec["output_tokens"] < spec["max_context"]]
        )
    ):
        for seed in spec["seeds"]:
            for width in range(8):
                add("forced_d7", f"accept{width}", context, seed, accepted=width)
            for variant in (
                "speculation",
                "verify_head",
                "dynamic_width",
                "graphs",
                "async_experimental",
            ):
                add("natural", variant, context, seed)
    add("natural", "head_omission_control", 129)
    for width in range(7):
        for seed in spec["seeds"]:
            add("rejected_suffix", f"accept{width}", 129, seed, accepted=width)
    for operator in ("mxfp4", "gdn", "norm_rope", "attention", "sampling"):
        add("operator", operator)
    add("operator", "native_dispatch", 256)
    for context in spec["contexts"]:
        for variant in (
            "warm",
            "ram",
            "eviction",
            "clean_restart",
            "shutdown_pending_tail",
            "crash_restart",
            "interrupted_write",
            "corrupt_disk",
            "missing_disk_block",
            "compaction",
            "cancellation",
        ):
            add("lifecycle", variant, context)
        for variant in (
            "equal0",
            "equal1",
            "equal2",
            "priority1_owner",
            "priority1_waiter",
            "priority2_waiter",
        ):
            add("priority", variant, context)
    for variant in (
        "parser_fragments",
        "pi_provider_fragments",
        "tool_roundtrip",
        "long_thinking",
        "minimal_release",
        "minimal_release_full_head",
        "repeat_release_full_head",
        "minimal_release_target_only",
        "repeat_release_target_only",
        "minimal_release_target_only_eager",
        "repeat_release_target_only_eager",
    ):
        add("protocol", variant)
    if len({case["id"] for case in cases}) != len(cases):
        raise DiagnosticError("duplicate campaign case IDs")
    cases.sort(key=case_priority)
    for case in cases:
        case["stage"] = ("pilot", "focused", "extended")[case_priority(case)[0]]
    return seal(
        {
            "schema": SCHEMA,
            "spec": spec,
            "sources": source_identity(),
            "cases": cases,
            "universal_correctness": "UNPROVED",
            "default_gpu_use": False,
            "scope_limits": [
                "Finite inputs and interleavings; no claim about all arbitrary inputs.",
                "Long-context M1/D7 agreement cannot exclude a bug shared by both native paths.",
                "Operator tolerance tests are distinct from bit-exact reference equality.",
                "Experimental async uses vLLM's scheduler, not the synchronous chat scheduler.",
                "Graph entry observations do not prove GPU ISA or compiler correctness.",
                "Tool/reasoning fixtures can reflect model errors; raw evidence is retained.",
            ],
        }
    )


def validate_campaign(campaign):
    authenticate(campaign)
    if campaign.get("schema") != SCHEMA:
        raise DiagnosticError("unsupported campaign format")
    canonical = build_campaign(campaign["spec"])
    if campaign != canonical:
        raise DiagnosticError(
            "campaign wiring, source identity or case inventory changed; re-plan explicitly"
        )
    return campaign


def coverage(campaign, results):
    validate_campaign(campaign)
    expected = {case["id"]: case for case in campaign["cases"]}
    if set(results) - set(expected):
        raise DiagnosticError("result names a case outside the campaign")
    rows = []
    for case in campaign["cases"]:
        attempts = results.get(case["id"], [])
        for result in attempts:
            authenticate(result)
            if (
                result.get("schema") != RESULT_SCHEMA
                or result.get("campaign") != campaign["sha256"]
                or result.get("case") != case
            ):
                raise DiagnosticError("case result belongs to different inputs, source or campaign")
            if result.get("status") not in {"TESTED", "FAILED", "ERROR", "TIMEOUT", "UNSUPPORTED"}:
                raise DiagnosticError("unsupported case status")
            if result["status"] == "TESTED" and (
                result.get("executed") is not True or not result.get("checks")
            ):
                # CPU-only parser/provider cases still execute in the explicitly
                # armed pinned qualification environment; their claim is scoped.
                raise DiagnosticError("passing result is missing actual execution/oracle evidence")
        status = (
            "NOT_RUN"
            if not attempts
            else (
                "TESTED"
                if all(r["status"] == "TESTED" for r in attempts)
                else next(r["status"] for r in attempts if r["status"] != "TESTED")
            )
        )
        rows.append(
            {
                "id": case["id"],
                "family": case["family"],
                "status": status,
                "attempts": [r["sha256"] for r in attempts],
                "flaky": len(attempts) > 1 and len({r["status"] for r in attempts}) > 1,
            }
        )
    return seal(
        {
            "schema": "urn:qwen:conformance-coverage:v1",
            "campaign": campaign["sha256"],
            "cases": rows,
            "counts": dict(Counter(row["status"] for row in rows)),
            "complete": bool(rows) and all(row["status"] == "TESTED" for row in rows),
            "universal_correctness": "UNPROVED",
            "scope_limits": campaign["scope_limits"],
        }
    )


def load_results(campaign, root):
    results = {}
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    roots = [Path(path).resolve() for path in roots]
    if not roots or len(set(roots)) != len(roots):
        raise DiagnosticError("result directories must be nonempty and unique")
    for directory in roots:
        if not directory.is_dir():
            raise DiagnosticError("campaign result directory is missing")
        saved_campaign = private_json(directory / "campaign.json")
        authenticate(saved_campaign)
        if saved_campaign != campaign:
            raise DiagnosticError("result directory belongs to another campaign")
        for path in sorted(
            directory.glob("case-*/result.json"),
            key=lambda p: (
                p.parent.name.split("-attempt-")[0],
                int(p.parent.name.split("-attempt-")[1]) if "-attempt-" in p.parent.name else 0,
            ),
        ):
            result = private_json(path)
            case = result.get("case", {}).get("id")
            results.setdefault(case, []).append(result)
    for attempts in results.values():
        attempts.sort(key=lambda result: result.get("finished_ns", 0))
    return results


def run_campaign(
    campaign,
    root,
    *,
    allow_gpu=False,
    selected=None,
    keep_going=False,
    resume=False,
    through="extended",
    budget_seconds=None,
    retry_failed=False,
    min_free_bytes=20 * 1024**3,
):
    if not allow_gpu:
        raise DiagnosticError("GPU use was not authorized; campaign remains NOT_RUN")
    validate_campaign(campaign)
    selected = set([case["id"] for case in campaign["cases"]] if selected is None else selected)
    if not selected or selected - {case["id"] for case in campaign["cases"]}:
        raise DiagnosticError("empty selection or unknown campaign case")
    spec = campaign["spec"]
    if hashlib.sha256(Path(spec["python"]).read_bytes()).hexdigest() != spec["python_sha256"]:
        raise DiagnosticError("qualification interpreter changed")
    from qwen_r9700_lab.conformance_queue import run

    return run(
        campaign,
        root,
        selected=selected,
        keep_going=keep_going,
        resume=resume,
        through=through,
        budget_seconds=budget_seconds,
        retry_failed=retry_failed,
        min_free_bytes=min_free_bytes,
    )


def execute_case(campaign, case, case_root):
    spec = campaign["spec"]
    started = time.monotonic()
    try:
        env = worker_environment(spec, case_root)
        env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
        env.pop("QWEN_CONFORMANCE_SERVER_SETTINGS", None)
        process = OwnedProcess(
            [spec["python"], "-m", "qwen_r9700_lab.conformance_campaign", str(case_root)],
            case_root / "process",
            env=env,
            timeout=spec["case_timeout_seconds"],
        )
        code = process.wait()
        result = private_json(case_root / "worker-result.json")
        if code and result.get("status") == "TESTED":
            raise DiagnosticError("case process failed after claiming success")
        return result
    except Exception as error:
        return case_result(
            campaign,
            case,
            "TIMEOUT" if isinstance(error, TimeoutError) else "ERROR",
            started=started,
            error_type=type(error).__name__,
            detail=str(error)[:1000],
        )


def case_result(campaign, case, status, *, started, checks=None, error_type=None, detail=None):
    return seal(
        {
            "schema": RESULT_SCHEMA,
            "campaign": campaign["sha256"],
            "case": case,
            "status": status,
            "executed": status == "TESTED",
            "gpu_executed": (
                False
                if case["family"] == "protocol"
                and case["variant"] in {"parser_fragments", "pi_provider_fragments"}
                else (True if status == "TESTED" else None)
            ),
            "seconds": time.monotonic() - started,
            "finished_ns": time.time_ns(),
            "checks": checks or [],
            "error_type": error_type,
            "detail": detail,
            "proof": "UNPROVED",
        }
    )


def main():
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("case worker is not armed")
    from qwen_r9700_lab.conformance_scenarios import SCENARIOS, UnavailableError

    root = Path(sys.argv[1])
    payload = private_json(root / "input.json")
    campaign, case = payload["campaign"], payload["case"]
    validate_campaign(campaign)
    started = time.monotonic()
    try:
        verify_runtime_artifacts(campaign["spec"], root)
        from contextlib import nullcontext

        from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

        # The independent reference owns the GPU only for its native arm.
        lease = (
            nullcontext() if case["family"] == "forced_reference" else gpu_lease(root / "gpu-lease")
        )
        with lease:
            checks = SCENARIOS[case["family"]](campaign["spec"], case, root)
        if not checks:
            raise DiagnosticError("case produced no observable checks")
        result = case_result(campaign, case, "TESTED", started=started, checks=checks)
    except Exception as error:
        result = case_result(
            campaign,
            case,
            "UNSUPPORTED" if isinstance(error, UnavailableError) else "FAILED",
            started=started,
            error_type=type(error).__name__,
            detail=str(error)[:1000],
        )
    write_private(root / "worker-result.json", result)
    return 0 if result["status"] == "TESTED" else 1


def verify_runtime_artifacts(spec, root):
    """CPU-only source/package/binary checks before any inference library import."""
    import importlib.metadata
    import importlib.util

    from qwen_r9700_lab.conformance_radiance import verify_sources

    module = importlib.util.find_spec("vllm")
    if module is None or module.origin is None:
        raise DiagnosticError("pinned Radiance runtime is unavailable")
    package = Path(module.origin).resolve().parent.parent
    verify_sources(package, spec["binding"])
    binary_hashes = spec["operator_profile"].get("kernel_hashes")
    versions = spec["operator_profile"].get("package_versions")
    if not binary_hashes or not versions:
        raise DiagnosticError("campaign omitted its native binary or package identity")
    for name, expected in binary_hashes.items():
        path = (package / name).resolve()
        if (
            not path.is_relative_to(package)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise DiagnosticError("native binary identity differs: " + name)
    for name, expected in versions.items():
        if importlib.metadata.version(name) != expected:
            raise DiagnosticError("native package identity differs: " + name)
    write_private(
        root / "runtime-identity.json",
        {
            "source_binding": spec["binding"]["sha256"],
            "binary_hashes": binary_hashes,
            "package_versions": versions,
            "python_sha256": spec["python_sha256"],
            "compiler_and_device_execution": "UNPROVED",
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
