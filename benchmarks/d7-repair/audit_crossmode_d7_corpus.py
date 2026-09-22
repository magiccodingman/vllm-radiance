"""Audit before/final eager-M1 versus compiled-M8 corpus replays without exporting tokens."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_crossmode_d7_corpus import compare_pair, load_corpus

from qwen_r9700_lab.conformance_topk import aggregate, require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def read(path):
    value = private_json(path)
    authenticate(value)
    return value


def validate_mode(config, runtime, *, arm, revision, measurement):
    eager, final = arm == "m1", revision == "final"
    require(config["enforce_eager"] is eager, "requested execution mode changed")
    require(runtime["enforce_eager"] is eager, "actual execution mode changed")
    require(runtime["compilation_mode"] == (0 if eager else 3), "compiler mode changed")
    require(runtime["graph_mode"] == ("NONE" if eager else "PIECEWISE"), "graph mode changed")
    if not eager:
        require(runtime["capture_sizes"] == [1, 2, 4, 8], "graph capture domain changed")
    require(("speculative_config" in config) == (not eager), "speculation mode changed")
    if not eager:
        require(config["speculative_config"]["num_speculative_tokens"] == 7, "D7 depth changed")
    require((runtime["repair"] is not None) == final, "wrong repair revision")
    require((runtime["performance"] is not None) == final, "wrong performance revision")
    if final:
        require(runtime["repair"]["bundle"] == measurement["repair_manifest"], "repair drift")
        require(
            runtime["performance"]["manifest"] == measurement["performance_manifest"],
            "performance drift",
        )
    rt = runtime["runtime"]
    authenticate(rt)
    require(rt["compiler_settings"]["emulate_precision_casts"] is final, "rounding setting changed")
    require(
        rt["flags"]["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] == str(int(final)), "cast flag changed"
    )
    require(rt["flags"]["RADIANCE_VERIFY_HEAD"] == "0", "approximate vocabulary head enabled")
    rotary = runtime.get("rotary_intervention")
    require((rotary is not None) == (eager and final), "wrong rotary contract")
    if rotary:
        authenticate(rotary["identity"])
        require(rotary["identity"]["installed_before_load"], "rotary fix installed too late")
    capacity = runtime["effective_capacity"]
    require(
        (capacity["block_size"], capacity["num_gpu_blocks"])
        == ((1568, 194) if eager else (1648, 370)),
        "hybrid cache layout changed",
    )
    flags = {k: v for k, v in rt["flags"].items() if k != "TORCHINDUCTOR_EMULATE_PRECISION_CASTS"}
    return {
        "configuration": {
            k: v
            for k, v in config.items()
            if k
            not in {
                "sha256",
                "enforce_eager",
                "compilation_config",
                "worker_cls",
                "speculative_config",
                "async_scheduling",
            }
        },
        "logical_capacity": {
            k: v for k, v in capacity.items() if k not in {"block_size", "num_gpu_blocks"}
        },
        "sources": runtime["diagnostic_sources"],
        "packages": rt["packages"],
        "kernel": rt["kernel"],
        "flags_except_casts": flags,
    }


def audit(public, private, corpus, prefix):
    common = None
    results, runs, measurements = {}, {}, {}
    for revision in ("before", "final"):
        root = public / (prefix + revision)
        private_root = private / ("qwen-" + prefix + revision)
        m, summary, completed = (
            read(root / name) for name in ("measurement.json", "summary.json", "completed.json")
        )
        require(m["revision"] == summary["revision"] == revision, "wrong comparison revision")
        require(m["reference"] == "fresh eager M1", "wrong reference")
        require(m["candidate"] == "fresh compiled M8 with piecewise GPU graphs", "wrong candidate")
        require(m["corpus"] == summary["corpus"] == corpus["sha256"], "corpus changed")
        require(completed["status"] == summary["status"] == "MEASURED", "run incomplete")
        require(completed["summary"] == summary["sha256"], "completion receipt changed")
        identity = {
            k: m[k]
            for k in (
                "corpus",
                "positions",
                "binding",
                "driver_sha256",
                "repair_manifest",
                "performance_manifest",
            )
        }
        measurements[revision] = {
            "measurement": m["sha256"],
            "summary": summary["sha256"],
            "completed": completed["sha256"],
        }
        rows_by_arm = {}
        for arm, width in (("m1", 1), ("m8", 8)):
            arm_root = root / arm
            complete = read(arm_root / "complete.json")
            require(
                private_json(arm_root / "process-result.json")["returncode"] == 0, "worker failed"
            )
            require(
                complete["positions"] == corpus["positions"]
                and complete["corpus"] == corpus["sha256"],
                "incomplete corpus",
            )
            require(
                complete["responses"] == len(complete["receipts"]) == len(corpus["continuations"]),
                "missing continuation",
            )
            config, runtime, after = (
                read(arm_root / name)
                for name in ("requested-config.json", "actual-runtime.json", "runtime-after.json")
            )
            normalized = validate_mode(config, runtime, arm=arm, revision=revision, measurement=m)
            require(
                validate_mode(config, after, arm=arm, revision=revision, measurement=m)
                == normalized,
                "runtime changed during replay",
            )
            current = {**identity, **normalized}
            if common is None:
                common = current
            require(current == common, "undeclared source or configuration difference")
            rows_by_arm[arm] = []
            graphs, seconds, shapes, hashes = 0, 0.0, {}, []
            for index, fixture in enumerate(corpus["continuations"]):
                receipt = read(arm_root / f"receipt-{index:03d}.json")
                require(receipt["sha256"] == complete["receipts"][index], "receipt changed")
                require(
                    receipt["fixture"] == fixture["sha256"]
                    and receipt["positions"] == fixture["evaluate_positions"],
                    "wrong history",
                )
                require(
                    receipt["output_sha256"] == fixture["output_sha256"], "forced tokens changed"
                )
                rows = read(private_root / arm / f"{index:03d}/correctness/rows.json")
                observation = receipt["observation"]
                forced = observation["forced"]
                require(rows["sha256"] == forced["sha256"], "unbound saved logits")
                require(
                    forced["positions"] == len(rows["rows"]) == fixture["evaluate_positions"],
                    "missing decode positions",
                )
                count = observation["observation"]["counts"].get("target_graph_replays", 0)
                require((count > 0) == (arm == "m8"), "graph execution contradicts mode")
                if revision == "final" and arm == "m1":
                    rotary = observation["rotary_intervention"]
                    require(rotary["calls"] > 0, "repaired rotary did not execute")
                    require(
                        rotary["identity"] == runtime["rotary_intervention"]["identity"],
                        "rotary identity changed",
                    )
                graphs += count
                seconds += receipt["elapsed_seconds"]
                for shape, count in forced["head_shapes"].items():
                    shapes[shape] = shapes.get(shape, 0) + count
                hashes.append({"receipt": receipt["sha256"], "rows": rows["sha256"]})
                rows_by_arm[arm].append(rows)
            require(
                shapes.get(f"({width}, 248320)", 0) >= corpus["positions"] // width,
                "wrong head domain",
            )
            runs[f"{revision}-{arm}"] = {
                "mode": "eager" if arm == "m1" else "compiled",
                "target_rows": width,
                "positions": corpus["positions"],
                "responses": len(hashes),
                "config": config["sha256"],
                "runtime": runtime["sha256"],
                "runtime_after": after["sha256"],
                "completion": complete["sha256"],
                "receipts": hashes,
                "repair": m["repair_manifest"] if revision == "final" else None,
                "performance": m["performance_manifest"] if revision == "final" else None,
                "precision_casts": revision == "final",
                "rotary_intervention": runtime.get("rotary_intervention"),
                "target_graph_replays": graphs,
                "head_shapes": shapes,
                "instrumented_replay_seconds": seconds,
            }
        decode, prefills = [], []
        for index, fixture in enumerate(corpus["continuations"]):
            paired, initial = compare_pair(
                rows_by_arm["m1"][index], rows_by_arm["m8"][index], fixture
            )
            decode.extend(paired)
            prefills.append(initial)
        result = {"decode": aggregate(decode), "prefill": aggregate(prefills)}
        require(
            result["decode"] == summary["decode"] and result["prefill"] == summary["prefill"],
            "recomputed results differ",
        )
        results[revision] = result
    return seal(
        {
            "schema": "qwen.crossmode-corpus-before-final.v1",
            "status": "AUDITED",
            "scope": (
                "Fresh eager M1 versus compiled M8 before and after both fixes on the same "
                "forced-token Pi corpus; full BF16 target head. Empirical output agreement, "
                "not arbitrary-input or latent-state proof. Replay durations include "
                "instrumentation and prefill and are not serving speed."
            ),
            "common": common,
            "measurements": measurements,
            "runs": runs,
            "comparisons": results,
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("public", "private", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--positions", type=int, default=10000)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus, args.corpus_sha256, args.positions)
    result = audit(args.public, args.private, corpus, args.prefix)
    write_private(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "sha256": result["sha256"],
                "comparisons": result["comparisons"],
            }
        )
    )
