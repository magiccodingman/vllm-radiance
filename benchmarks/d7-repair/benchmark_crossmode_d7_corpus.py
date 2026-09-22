"""Compare eager M1 with compiled M8 before or after both D7 repairs on a frozen corpus.

Private tokens and ranked IDs stay in owner-only tmpfs. This is a correctness
measurement: forced-token replay and logit capture do not measure serving speed.
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.conformance_topk import aggregate, compare_rows, require
from qwen_r9700_lab.diagnostic_contract import (
    authenticate,
    digest,
    private_json,
    seal,
    write_private,
)


def load_corpus(path, expected_sha256, positions):
    manifest = private_json(path / "manifest.json")
    authenticate(manifest)
    require(manifest["sha256"] == expected_sha256, "wrong frozen corpus")
    require(manifest["positions"] == positions > 0, "wrong position budget")
    seen, total = set(), 0
    for row in manifest["continuations"]:
        name = row["name"]
        require(Path(name).name == name and name not in seen, "invalid or duplicate fixture")
        seen.add(name)
        fixture = private_json(path / name)
        authenticate(fixture)
        require(fixture["sha256"] == row["sha256"], "frozen fixture changed")
        count = row["evaluate_positions"]
        require(type(count) is int and count > 0 and count % 8 == 0, "incomplete M8 group")
        require(len(fixture["output"]) == count + 1, "forced stream length changed")
        require(len(fixture["prefix"]) == row["prefix_tokens"], "prefix length changed")
        for key, data in (("prefix_sha256", "prefix"), ("output_sha256", "output")):
            require(digest(fixture[data]) == row[key], "fixture token identity changed")
        total += count
    require(total == positions, "corpus does not cover the requested budget")
    return manifest


def compare_pair(reference, candidate, row):
    for report, width in ((reference, 1), (candidate, 8)):
        authenticate(report)
        require(report["schema"] == "urn:qwen:d7-equivalence-private-rows:v1", "wrong row schema")
        require(report["continuation"] == row["sha256"], "replay fixture changed")
        require(len(report["rows"]) == row["evaluate_positions"], "incomplete replay")
        for position, item in enumerate(report["rows"]):
            require(item["position"] == position, "replay position changed")
            require(
                item["absolute_position"] == row["prefix_tokens"] + position,
                "wrong absolute position",
            )
            require(item["target_rows"] == width, "wrong target execution width")
    compared = []
    for left, right in zip(reference["rows"], candidate["rows"], strict=True):
        require(left["absolute_position"] == right["absolute_position"], "position mismatch")
        compared.append(compare_rows(left["logits"], right["logits"]))
    return compared, compare_rows(reference["prefill"], candidate["prefill"])


def worker(args):
    from benchmark_optimized_d7 import make_config, validate_execution_metadata
    from vllm import LLM, SamplingParams

    from qwen_r9700_lab.conformance_cli import reject_dead_native_rpcs
    from qwen_r9700_lab.conformance_radiance import verify_sources

    corpus = load_corpus(args.corpus, args.corpus_sha256, args.positions)
    spec = private_json(args.spec)
    package = Path(importlib.util.find_spec("vllm").origin).parent.parent
    verify_sources(package, spec["binding"])
    root = args.output / args.arm
    mode = "eager" if args.arm == "m1" else "compiled"
    lane = "old-bf16" if args.revision == "before" else "fixed-bf16"
    config = make_config(spec, lane, speculation=args.arm == "m8", execution_mode=mode)
    if args.revision == "final" and args.arm == "m1":
        require(
            config["worker_cls"] == "execution_mode_d7_worker.ExecutionModeWorker",
            "unexpected eager worker",
        )
        config["worker_cls"] = "rotary_mode_d7_worker.RotaryRneWorker"
    write_private(root / "requested-config.json", seal(config))
    llm = LLM(**config)
    engine = llm.llm_engine
    all_rows, all_prefills, receipts = [], [], []
    completed = 0
    with reject_dead_native_rpcs(engine.engine_core):
        metadata = llm.collective_rpc("qwen_optimized_metadata")[0]
        validate_execution_metadata(metadata, execution_mode=mode, isolated_capture=False)
        final = args.revision == "final"
        require((metadata["repair"] is not None) == final, "wrong repair presence")
        require((metadata["performance"] is not None) == final, "wrong performance presence")
        if final:
            require(
                metadata["repair"]["bundle"] == private_json(args.repair_manifest)["sha256"],
                "wrong repair bundle",
            )
            require(
                metadata["performance"]["manifest"]
                == private_json(args.performance_manifest)["sha256"],
                "wrong performance bundle",
            )
        require(
            metadata["runtime"]["compiler_settings"]["emulate_precision_casts"] is final,
            "wrong rounding contract",
        )
        require(
            metadata["runtime"]["flags"]["RADIANCE_VERIFY_HEAD"] == "0", "approximate head enabled"
        )
        require(
            (metadata.get("rotary_intervention") is not None) == (final and args.arm == "m1"),
            "wrong rotary repair",
        )
        write_private(root / "actual-runtime.json", seal(metadata))
        for index, row in enumerate(corpus["continuations"]):
            fixture_path = args.corpus / row["name"]
            fixture = private_json(fixture_path)
            private = args.private / args.arm / f"{index:03d}"
            reference_path = args.private / "m1" / f"{index:03d}" / "correctness" / "rows.json"
            task = seal(
                {
                    "continuation": str(fixture_path),
                    "continuation_sha256": row["sha256"],
                    "private_output": str(private / "correctness"),
                    "binding": spec["binding"],
                    "speculation": args.arm == "m8",
                    "arm": args.arm,
                    "index": index,
                    "report_root": str(root),
                    "trace_rows": False,
                    "repair_manifest": None,
                    "reference_rows": str(reference_path) if args.arm == "m8" else None,
                    "reference_rows_sha256": private_json(reference_path)["sha256"]
                    if args.arm == "m8"
                    else None,
                }
            )
            task_path = root / f"task-{index:03d}.json"
            write_private(task_path, task)
            params = SamplingParams(
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=True,
                max_tokens=len(fixture["output"]),
                detokenize=False,
            )
            request = f"crossmode-corpus-{args.revision}-{args.arm}-{index}"
            started = last_progress = time.perf_counter()
            received = 0
            llm.collective_rpc("qwen_optimized_begin", args=(str(private), False, str(task_path)))
            try:
                engine.add_request(
                    request,
                    {
                        "prompt_token_ids": fixture["prefix"],
                        "cache_salt": digest({"corpus": corpus["sha256"], "response": index}),
                    },
                    params,
                )
                last = None
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        require(output.request_id == request, "unexpected request")
                        if output.outputs:
                            last = output.outputs[0]
                            received = len(last.token_ids)
                    if time.perf_counter() - last_progress > 10:
                        replace_private(
                            root,
                            "controller-progress.json",
                            seal(
                                {
                                    "arm": args.arm,
                                    "response": index,
                                    "completed": completed,
                                    "current_positions": max(received - 1, 0),
                                    "phase": "decode" if received else "prefill",
                                }
                            ),
                        )
                        last_progress = time.perf_counter()
                require(
                    last is not None and list(last.token_ids) == fixture["output"],
                    "forced output changed or stopped early",
                )
                observation = llm.collective_rpc("qwen_optimized_finish")[0]
                record = seal(
                    {
                        "arm": args.arm,
                        "index": index,
                        "fixture": row["sha256"],
                        "positions": row["evaluate_positions"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "output_sha256": digest(list(last.token_ids)),
                        "observation": observation,
                    }
                )
                write_private(root / f"receipt-{index:03d}.json", record)
                receipts.append(record["sha256"])
                completed += row["evaluate_positions"]
                if args.arm == "m8":
                    paired, prefill = compare_pair(
                        private_json(reference_path),
                        private_json(private / "correctness" / "rows.json"),
                        row,
                    )
                    all_rows.extend(paired)
                    all_prefills.append(prefill)
                    replace_private(
                        args.output,
                        "comparison-progress.json",
                        seal(
                            {
                                "corpus": corpus["sha256"],
                                "completed_responses": index + 1,
                                "decode": aggregate(all_rows),
                                "prefill": aggregate(all_prefills),
                            }
                        ),
                    )
                replace_private(
                    root,
                    "checkpoint.json",
                    seal(
                        {
                            "arm": args.arm,
                            "completed_responses": index + 1,
                            "completed_positions": completed,
                            "receipts": receipts,
                        }
                    ),
                )
            finally:
                engine.abort_request([request])
        require(completed == args.positions, "incomplete arm")
        write_private(
            root / "runtime-after.json", seal(llm.collective_rpc("qwen_optimized_metadata")[0])
        )
    write_private(
        root / "complete.json",
        seal(
            {
                "corpus": corpus["sha256"],
                "arm": args.arm,
                "positions": completed,
                "responses": len(receipts),
                "receipts": receipts,
            }
        ),
    )
    if args.arm == "m8":
        write_private(
            args.output / "summary.json",
            seal(
                {
                    "status": "MEASURED",
                    "revision": args.revision,
                    "corpus": corpus["sha256"],
                    "responses": len(receipts),
                    "decode": aggregate(all_rows),
                    "prefill": aggregate(all_prefills),
                    "scope": "eager M1 versus compiled GPU-graph M8; full BF16 target head; "
                    "full-vocabulary hashes and top-k; not arbitrary-input or latent-state proof",
                }
            ),
        )


def run(args):
    from benchmark_d7_equivalence import private_root

    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    private_root(args.corpus)
    corpus = load_corpus(args.corpus, args.corpus_sha256, args.positions)
    args.output.mkdir(mode=0o700)
    args.private.mkdir(mode=0o700)
    private_root(args.private)
    spec = private_json(args.spec)
    manifests = {}
    for key in ("repair_manifest", "performance_manifest"):
        value = private_json(getattr(args, key))
        authenticate(value)
        manifests[key] = value["sha256"]
    write_private(
        args.output / "measurement.json",
        seal(
            {
                "corpus": corpus["sha256"],
                "positions": args.positions,
                "binding": spec["binding"]["sha256"],
                **manifests,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "revision": args.revision,
                "reference": "fresh eager M1",
                "candidate": "fresh compiled M8 with piecewise GPU graphs",
            }
        ),
    )
    with gpu_lease(args.output / "gpu-lease"):
        for arm in ("m1", "m8"):
            (args.output / arm).mkdir(mode=0o700)
            (args.private / arm).mkdir(mode=0o700)
            env = worker_environment(spec, args.output / arm)
            env.update(
                {
                    "RADIANCE_VERIFY_HEAD": "0",
                    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1"
                    if args.revision == "final"
                    else "0",
                    "QWEN_OPTIMIZED_STARTUP_RECEIPT": str(
                        args.output / arm / "before-compile.json"
                    ),
                    "PYTHONPATH": os.pathsep.join(
                        (
                            str(Path(__file__).resolve().parent),
                            env["PYTHONPATH"],
                            str(args.runtime_source),
                        )
                    ),
                }
            )
            env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
            env.pop("QWEN_OPTIMIZED_REPAIR", None)
            env.pop("QWEN_OPTIMIZED_PERFORMANCE", None)
            if args.revision == "final":
                env["QWEN_OPTIMIZED_REPAIR"] = str(args.repair_manifest)
                env["QWEN_OPTIMIZED_PERFORMANCE"] = str(args.performance_manifest)
            argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--arm", arm]
            for key in (
                "spec",
                "corpus",
                "corpus_sha256",
                "positions",
                "revision",
                "private",
                "output",
                "repair_manifest",
                "performance_manifest",
                "runtime_source",
            ):
                argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            with OwnedProcess(
                argv, args.private / arm / "process", env=env, timeout=10800
            ) as process:
                code = process.wait()
            write_private(args.output / arm / "process-result.json", {"returncode": code})
            require(code == 0, "cross-mode replay failed; private log retained")
    write_private(
        args.output / "completed.json",
        seal(
            {
                "status": "MEASURED",
                "summary": private_json(args.output / "summary.json")["sha256"],
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    for key in (
        "spec",
        "corpus",
        "private",
        "output",
        "repair-manifest",
        "performance-manifest",
        "runtime-source",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--positions", type=int, default=10000)
    parser.add_argument("--arm", choices=("m1", "m8"))
    parser.add_argument("--revision", choices=("before", "final"), required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    # The runtime directory also contains an older qwen_r9700_lab package.
    # Keep the pinned diagnostic source first in parent and spawned workers.
    sys.path.append(str(args.runtime_source))
    (run if args.command == "run" else worker)(args)


if __name__ == "__main__":
    main()
