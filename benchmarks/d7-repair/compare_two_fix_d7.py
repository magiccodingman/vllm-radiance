"""Audit seven native runs and export four transcript-free 320-position comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

from qwen_r9700_lab.conformance_topk import aggregate, compare_rows, require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

CASES = {
    "original-m1": ("old-bf16", 0, "compiled", 1, False),
    "original-m8": ("old-bf16", 0, "compiled", 8, False),
    "fix1-m1": ("fixed-bf16", 0, "compiled", 1, False),
    "fix1-m8": ("fixed-bf16", 0, "compiled", 8, False),
    "fix1-eager-m8": ("fixed-bf16", 0, "eager", 8, False),
    "final-m8": ("fixed-bf16", 1, "compiled", 8, False),
    "final-eager-m8": ("fixed-bf16", 1, "eager", 8, True),
}
PAIRS = {
    "original_compiled_m8_m1": ("original-m1", "original-m8"),
    "fix1_compiled_m8_m1": ("fix1-m1", "fix1-m8"),
    "fix1_compiled_eager_m8": ("fix1-eager-m8", "fix1-m8"),
    "final_compiled_eager_m8": ("final-eager-m8", "final-m8"),
}


def analyze(public_root, private_root, prefix):
    records, private = {}, {}
    common = None
    for case, (lane, casts, mode, width, rotary) in CASES.items():
        name = prefix + case
        root = public_root / name
        directory = root / lane
        paths = {
            "measurement": root / "measurement.json",
            "config": directory / "requested-config.json",
            "runtime": directory / "actual-runtime.json",
            "pass": directory / "pass-00.json",
            "summary": directory / "summary.json",
            "rows": private_root / ("qwen-" + name) / lane / "pass-00/correctness/rows.json",
        }
        data = {key: private_json(path) for key, path in paths.items()}
        for doc in data.values():
            authenticate(doc)
        m, c, r, p, rows = (data[k] for k in ("measurement", "config", "runtime", "pass", "rows"))
        rt = r["runtime"]
        authenticate(rt)
        require(m["correctness"] and m["m1"] == (width == 1), "wrong replay mode")
        require(m["lanes"] == [lane] and not m["isolated_capture"], "wrong replay lane")
        require(m["execution_mode"] == p["execution_mode"] == mode, "wrong execution mode")
        require(m["prefix_tokens"] == 60000 and len(rows["rows"]) == 320, "wrong fixture domain")
        require(p["fixture"] == rows["continuation"] == m["fixture"], "different token histories")
        require(p["observation"]["forced"]["sha256"] == rows["sha256"], "unbound row evidence")
        require(data["summary"]["passes"] == [p["sha256"]], "incomplete or extra passes")
        eager = mode == "eager"
        require(c["enforce_eager"] == r["enforce_eager"] == eager, "eager mode changed")
        require((r["compilation_mode"] == 0) == eager, "compiler setting changed")
        require(r["graph_mode"] == ("NONE" if eager else "PIECEWISE"), "graph mode changed")
        counts = p["observation"]["observation"]["counts"]
        require(counts["target_forward_calls"] > 0, "no observed model execution")
        require((counts.get("target_graph_replays", 0) > 0) == (not eager), "graph replay differs")
        require(rt["compiler_settings"]["emulate_precision_casts"] is bool(casts), "wrong casts")
        flags = dict(rt["flags"])
        require(flags.pop("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == str(casts), "wrong cast flag")
        require(flags["RADIANCE_VERIFY_HEAD"] == "0", "approximate head enabled")
        require((r["repair"] is None) == (lane == "old-bf16"), "wrong repair integration")
        if lane == "fixed-bf16":
            require(r["performance"] is not None, "performance repairs absent")
        configuration = {
            k: v
            for k, v in c.items()
            if k
            not in {
                "sha256",
                "enforce_eager",
                "compilation_config",
                "worker_cls",
                "speculative_config",
                "async_scheduling",
            }
        }
        capacity = r["effective_capacity"]
        require(
            (capacity["block_size"], capacity["num_gpu_blocks"])
            == ((1568, 194) if width == 1 else (1648, 370)),
            "physical hybrid-cache layout is outside the pinned M1/M8 domain",
        )
        identity = {
            "fixture": m["fixture"],
            "binding": m["binding"],
            "logical_capacity": {
                k: v for k, v in capacity.items() if k not in {"block_size", "num_gpu_blocks"}
            },
            "sources": r["diagnostic_sources"],
            "packages": rt["packages"],
            "kernel": rt["kernel"],
            "flags_except_casts": flags,
            "configuration": configuration,
        }
        if common is None:
            common = identity
        require(identity == common, "an undeclared configuration/source difference occurred")
        rotary_receipt = p["observation"].get("rotary_intervention")
        require((rotary_receipt is not None) == rotary, "unexpected rotary intervention")
        if rotary:
            require(rotary_receipt["calls"] > 0, "RNE rotary kernel did not execute")
            authenticate(rotary_receipt["identity"])
            require(
                rotary_receipt["identity"]["installed_before_load"], "rotary fix installed too late"
            )
        for index, row in enumerate(rows["rows"]):
            require(
                row["position"] == index and row["absolute_position"] == 60000 + index,
                "misaligned positions",
            )
            require(row["target_rows"] == width, "wrong observed target width")
        records[case] = {
            "receipts": {k: v["sha256"] for k, v in data.items()},
            "execution_mode": mode,
            "target_rows": width,
            "precision_casts": bool(casts),
            "repair": r["repair"]["bundle"] if r["repair"] else None,
            "performance": r["performance"]["manifest"] if r["performance"] else None,
            "rotary_intervention": rotary_receipt,
            "graph_replays": counts.get("target_graph_replays", 0),
            "physical_capacity": capacity,
        }
        private[case] = rows
    require(len({r["repair"] for r in records.values() if r["repair"]}) == 1, "repair bundle drift")
    require(
        len({r["performance"] for r in records.values() if r["performance"]}) == 1,
        "performance bundle drift",
    )
    comparisons = {}
    for name, pair in PAIRS.items():
        left, right = (private[case] for case in pair)
        comparisons[name] = {
            "cases": list(pair),
            "decode": aggregate(
                [
                    compare_rows(a["logits"], b["logits"])
                    for a, b in zip(left["rows"], right["rows"], strict=True)
                ]
            ),
            "prefill": aggregate([compare_rows(left["prefill"], right["prefill"])]),
        }
    return seal(
        {
            "schema": "qwen.two-fix-native-comparisons.v1",
            "status": "MEASURED",
            "common": common,
            "runs": records,
            "comparisons": comparisons,
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": (
                "Four whole-model comparisons on 320 aligned decode positions and one prefill "
                "prediction. These are not isolated-stage measurements or arbitrary-input proofs."
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.public_root, args.private_root, args.prefix)
    write_private(args.output, result)
    print(json.dumps({name: value["decode"] for name, value in result["comparisons"].items()}))
