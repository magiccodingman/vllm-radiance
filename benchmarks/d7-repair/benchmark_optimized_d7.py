"""Compiled/graph-mode old versus repaired D7, with separate correctness replay.

Natural performance runs retain ordinary sampling and stopping. Private tokens,
logs and raw traces remain in owner-only tmpfs. Public reports contain metrics.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import (
    authenticate,
    digest,
    private_json,
    seal,
    write_private,
)


def make_config(spec, lane, *, speculation=True, isolated_capture=False, execution_mode="compiled"):
    require(execution_mode in ("compiled", "compiled-no-graphs", "eager"), "invalid execution mode")
    config = dict(spec["native_config"])
    for key in ("kv_transfer_config", "scheduler_cls", "additional_config", "async_scheduling"):
        config.pop(key, None)
    config.update(
        enforce_eager=False,
        compilation_config={"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8]},
        worker_cls="optimized_d7_worker.OptimizedWorker",
        disable_log_stats=True,
    )
    if not speculation:
        config.pop("speculative_config", None)
        # Keep the same ordered scheduler used by this DFlash release.
        config["async_scheduling"] = False
    if isolated_capture:
        config["compilation_config"] = {"cudagraph_mode": "NONE"}
        config["worker_cls"] = "isolated_d7_capture.IsolatedCaptureWorker"
    if execution_mode != "compiled":
        config["enforce_eager"] = execution_mode == "eager"
        config["compilation_config"] = {"cudagraph_mode": "NONE"}
        if execution_mode == "eager":
            config["compilation_config"]["mode"] = 0
        config["worker_cls"] = (
            "execution_mode_d7_worker.ExecutionModeCaptureWorker"
            if isolated_capture
            else "execution_mode_d7_worker.ExecutionModeWorker"
        )
    return config


def validate_execution_metadata(metadata, *, execution_mode, isolated_capture):
    eager = execution_mode == "eager"
    require(metadata["enforce_eager"] == eager, "actual eager setting differs from requested mode")
    require((metadata["compilation_mode"] == 0) == eager, "actual compilation mode differs")
    expected_graph = "NONE" if isolated_capture or execution_mode != "compiled" else "PIECEWISE"
    require(
        str(metadata["graph_mode"]).split(".")[-1] == expected_graph,
        "actual graph mode differs from requested mode",
    )


def worker(args):
    from vllm import LLM, SamplingParams

    from qwen_r9700_lab.conformance_cli import reject_dead_native_rpcs
    from qwen_r9700_lab.conformance_radiance import verify_sources

    spec = private_json(args.spec)
    fixture = private_json(args.fixture)
    authenticate(fixture)
    package = Path(importlib.util.find_spec("vllm").origin).parent.parent
    verify_sources(package, spec["binding"])
    root = args.output / args.lane
    config = make_config(
        spec,
        args.lane,
        speculation=not args.m1,
        isolated_capture=args.isolated_capture,
        execution_mode=args.execution_mode,
    )
    write_private(root / "requested-config.json", seal(config))
    llm = LLM(**config)
    engine = llm.llm_engine
    with reject_dead_native_rpcs(engine.engine_core):
        metadata = llm.collective_rpc("qwen_optimized_metadata")[0]
    validate_execution_metadata(
        metadata, execution_mode=args.execution_mode, isolated_capture=args.isolated_capture
    )
    write_private(root / "actual-runtime.json", seal(metadata))
    passes = []
    modes = ["correctness"] if args.correctness else ["warmup"] + ["clean"] * args.repeats
    if args.profile and not args.correctness:
        modes.append("profile")
    if args.with_correctness and not args.correctness:
        modes.append("correctness")
    for index, mode in enumerate(modes):
        private = args.private / args.lane / f"pass-{index:02d}"
        task_path = None
        if mode == "correctness":
            task = seal(
                {
                    "continuation": str(args.fixture),
                    "continuation_sha256": fixture["sha256"],
                    "private_output": str(private / "correctness"),
                    "binding": spec["binding"],
                    "speculation": not args.m1,
                    "arm": "m1" if args.m1 else "m8",
                    "index": 0,
                    "report_root": str(root),
                    "trace_rows": False,
                    "repair_manifest": None,
                }
            )
            task_path = root / "correctness-task.json"
            write_private(task_path, task)
        limit = (
            len(fixture["output"])
            if mode == "correctness"
            else (256 if mode == "warmup" else args.tokens)
        )
        if mode == "profile":
            limit = 256
        params = SamplingParams(
            temperature=0 if mode == "correctness" else 1,
            top_p=1 if mode == "correctness" else 0.95,
            top_k=-1 if mode == "correctness" else 20,
            seed=117 + index,
            max_tokens=limit,
            ignore_eos=mode in ("warmup", "profile", "correctness"),
            detokenize=False,
        )
        request = f"optimized-{args.lane}-{index}"
        received = 0
        first_at = None
        first_count = 0
        timings = []
        profile_started = profile_stopped = False
        started = time.perf_counter()
        last_progress = started
        with reject_dead_native_rpcs(engine.engine_core):
            llm.collective_rpc(
                "qwen_optimized_begin",
                args=(str(private), mode == "profile", str(task_path) if task_path else None),
            )
            try:
                engine.add_request(
                    request,
                    {
                        "prompt_token_ids": fixture["prefix"],
                        "cache_salt": digest(
                            {"fixture": fixture["sha256"], "lane": args.lane, "pass": index}
                        ),
                    },
                    params,
                )
                while engine.has_unfinished_requests():
                    before = received
                    step_started = time.perf_counter()
                    outputs = engine.step()
                    elapsed = time.perf_counter() - step_started
                    for output in outputs:
                        require(output.request_id == request, "unexpected request")
                        if output.outputs:
                            last = output.outputs[0]
                            received = len(last.token_ids)
                    if first_at is None and received:
                        first_at, first_count = time.perf_counter(), received
                    elif first_at is not None:
                        timings.append({"seconds": elapsed, "tokens": received - before})
                    if mode == "profile" and len(timings) == 8:
                        llm.collective_rpc("qwen_optimized_profile", args=(True,))
                        profile_started = True
                    if mode == "profile" and len(timings) == 16:
                        llm.collective_rpc("qwen_optimized_profile", args=(False,))
                        profile_stopped = True
                    if time.perf_counter() - last_progress > 10:
                        print(
                            json.dumps(
                                {
                                    "lane": args.lane,
                                    "pass": index,
                                    "phase": "decode" if received else "prefill",
                                    "output_tokens": received,
                                }
                            ),
                            flush=True,
                        )
                        last_progress = time.perf_counter()
                finished = time.perf_counter()
                require(received > 0 and first_at is not None, "request produced no output")
                if mode == "profile":
                    require(profile_started and profile_stopped, "incomplete profile window")
                observation = llm.collective_rpc("qwen_optimized_finish")[0]
                if mode == "correctness":
                    require(list(last.token_ids) == fixture["output"], "forced output changed")
                write_private(
                    private / "output.json",
                    seal({"token_ids": list(last.token_ids), "finish_reason": last.finish_reason}),
                )
                trimmed = timings[8:]
                record = seal(
                    {
                        "lane": args.lane,
                        "mode": mode,
                        "execution_mode": args.execution_mode,
                        "index": index,
                        "fixture": fixture["sha256"],
                        "output_tokens": received,
                        "finish_reason": last.finish_reason,
                        "prefill_to_first_seconds": first_at - started,
                        "after_first_seconds": finished - first_at,
                        "after_first_tokens": received - first_count,
                        "tokens_per_second": (received - first_count) / (finished - first_at),
                        "steps_after_first": timings,
                        "steady_median_step_ms": 1000
                        * statistics.median(t["seconds"] for t in trimmed)
                        if trimmed
                        else None,
                        "steady_tokens_per_step": statistics.mean(t["tokens"] for t in trimmed)
                        if trimmed
                        else None,
                        "output_sha256": digest(list(last.token_ids)),
                        "observation": observation,
                    }
                )
                write_private(root / f"pass-{index:02d}.json", record)
                passes.append(record)
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "lane",
                                "mode",
                                "index",
                                "output_tokens",
                                "tokens_per_second",
                                "steady_median_step_ms",
                            )
                        }
                    ),
                    flush=True,
                )
            finally:
                engine.abort_request([request])
    clean = [p for p in passes if p["mode"] == "clean"]
    summary = {
        "status": "MEASURED" if clean else "REPLAYED",
        "lane": args.lane,
        "execution_mode": args.execution_mode,
        "fixture": fixture["sha256"],
        "metadata": metadata,
        "passes": [p["sha256"] for p in passes],
        "replay_result": next(
            (p["observation"]["forced"] for p in passes if p["mode"] == "correctness"), None
        ),
    }
    if clean:
        require(
            all(p["steady_median_step_ms"] is not None for p in clean),
            "natural completion too short for timing",
        )
        summary["median_step_ms"] = statistics.median(p["steady_median_step_ms"] for p in clean)
        summary["repeat_step_medians_ms"] = [p["steady_median_step_ms"] for p in clean]
        summary["tokens_per_second"] = sum(p["after_first_tokens"] for p in clean) / sum(
            p["after_first_seconds"] for p in clean
        )
        summary["tokens_per_step"] = statistics.mean(p["steady_tokens_per_step"] for p in clean)
    write_private(root / "summary.json", seal(summary))


def run(args):
    from benchmark_d7_equivalence import private_root

    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    args.output.mkdir(mode=0o700)
    args.private.mkdir(mode=0o700)
    private_root(args.private)
    private_root(args.fixture.parent)
    spec = private_json(args.spec)
    fixture = private_json(args.fixture)
    authenticate(fixture)
    write_private(
        args.output / "measurement.json",
        seal(
            {
                "fixture": fixture["sha256"],
                "binding": spec["binding"]["sha256"],
                "prefix_tokens": len(fixture["prefix"]),
                "lanes": args.lanes,
                "repeats": args.repeats,
                "max_output_tokens": args.tokens,
                "correctness": args.correctness,
                "m1": args.m1,
                "isolated_capture": args.isolated_capture,
                "execution_mode": args.execution_mode,
                "performance_manifest": str(args.performance_manifest)
                if args.performance_manifest
                else None,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        ),
    )
    with gpu_lease(args.output / "gpu-lease"):
        for lane in args.lanes:
            (args.output / lane).mkdir(mode=0o700)
            (args.private / lane).mkdir(mode=0o700)
            env = worker_environment(spec, args.output / lane)
            env["RADIANCE_VERIFY_HEAD"] = "1" if lane == "old-fast" else "0"
            env["PYTHONPATH"] = (
                str(Path(__file__).resolve().parent) + os.pathsep + env["PYTHONPATH"]
            )
            env["QWEN_OPTIMIZED_STARTUP_RECEIPT"] = str(args.output / lane / "before-compile.json")
            env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
            env.pop("QWEN_OPTIMIZED_REPAIR", None)
            env.pop("QWEN_OPTIMIZED_PERFORMANCE", None)
            if lane == "fixed-bf16":
                require(args.repair_manifest is not None, "fixed lane requires the repair bundle")
                env["QWEN_OPTIMIZED_REPAIR"] = str(args.repair_manifest)
                if args.performance_manifest:
                    env["QWEN_OPTIMIZED_PERFORMANCE"] = str(args.performance_manifest)
            argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--lane", lane]
            for key in (
                "spec",
                "fixture",
                "private",
                "output",
                "tokens",
                "repeats",
                "execution_mode",
            ):
                argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            for flag in ("profile", "correctness", "with_correctness", "m1", "isolated_capture"):
                if getattr(args, flag):
                    argv += ["--" + flag.replace("_", "-")]
            with OwnedProcess(
                argv, args.private / lane / "process", env=env, timeout=7200
            ) as process:
                code = process.wait()
            write_private(args.output / lane / "process-result.json", {"returncode": code})
            require(code == 0, "optimized worker failed; private startup/request log retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    for key in ("spec", "fixture", "private", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path)
    parser.add_argument("--performance-manifest", type=Path)
    parser.add_argument("--lane", choices=("old-bf16", "old-fast", "fixed-bf16"))
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=("old-bf16", "old-fast", "fixed-bf16"),
        default=["old-bf16", "fixed-bf16"],
    )
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--execution-mode",
        choices=("compiled", "compiled-no-graphs", "eager"),
        default="compiled",
        help="Replay correctness with the same repair bindings; non-default modes are untimed.",
    )
    for key in (
        "profile",
        "correctness",
        "with-correctness",
        "m1",
        "allow-gpu",
        "isolated-capture",
    ):
        parser.add_argument("--" + key, action="store_true")
    args = parser.parse_args()
    require(
        args.tokens >= 128
        and args.repeats >= 0
        and (args.repeats > 0 or args.profile or args.correctness),
        "invalid timing budget",
    )
    require(
        not args.isolated_capture or (args.correctness and not args.profile),
        "isolated capture is an untimed correctness diagnostic, not a release-speed measurement",
    )
    require(
        args.execution_mode == "compiled" or (args.correctness and not args.profile),
        "execution-mode comparison requires an untimed forced correctness replay",
    )
    os.umask(0o077)
    (run if args.command == "run" else worker)(args)


if __name__ == "__main__":
    main()
