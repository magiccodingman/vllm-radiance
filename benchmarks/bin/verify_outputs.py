#!/usr/bin/env python3
"""Verify greedy outputs match across two benchmark configurations.

This is primarily a speculative-decoding correctness gate.  The benchmark
client records generated text but not token IDs, so we require both the text
and reported output lengths to match for every common raw case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def raw_files(path: Path) -> dict[str, Path]:
    raw = path / "raw" if path.is_dir() else path
    if not raw.is_dir():
        raise SystemExit(f"raw result directory does not exist: {raw}")
    return {item.name: item for item in sorted(raw.glob("*.json"))}


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require identical greedy text/lengths for common raw cases."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = raw_files(args.baseline)
    candidate = raw_files(args.candidate)
    common = sorted(baseline.keys() & candidate.keys())
    if not common:
        raise SystemExit("no common raw benchmark cases")

    mismatches: list[str] = []
    comparisons = 0
    for name in common:
        base = load(baseline[name])
        cand = load(candidate[name])
        if str(base.get("temperature")) not in {"0", "0.0"}:
            mismatches.append(f"{name}: baseline is not greedy")
            continue
        if str(cand.get("temperature")) not in {"0", "0.0"}:
            mismatches.append(f"{name}: candidate is not greedy")
            continue
        for field in ("seed", "input_lens"):
            if base.get(field) != cand.get(field):
                mismatches.append(f"{name}: {field} differs")
        base_text = base.get("generated_texts")
        cand_text = cand.get("generated_texts")
        if not isinstance(base_text, list) or not isinstance(cand_text, list):
            mismatches.append(f"{name}: detailed generated text is missing")
            continue
        comparisons += max(len(base_text), len(cand_text))
        if base.get("output_lens") != cand.get("output_lens"):
            mismatches.append(f"{name}: output token lengths differ")
        if base_text != cand_text:
            differing = sum(
                left != right
                for left, right in zip(base_text, cand_text, strict=False)
            ) + abs(len(base_text) - len(cand_text))
            mismatches.append(f"{name}: generated text differs for {differing} request(s)")

    missing = sorted(baseline.keys() - candidate.keys())
    report = {
        "status": "failed" if mismatches or missing else "passed",
        "common_cases": len(common),
        "compared_requests": comparisons,
        "missing_candidate_cases": missing,
        "mismatches": mismatches,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if report["status"] != "passed":
        raise SystemExit("greedy output equivalence failed")


if __name__ == "__main__":
    main()
