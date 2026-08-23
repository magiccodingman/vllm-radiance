#!/usr/bin/env python3
"""Compare two Radiance summary.json files on their common workload keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = (
    "config",
    "workload",
    "input_tokens",
    "output_tokens",
    "concurrency",
    "tp",
    "spec",
    "cpu_offload_gb",
)


def load(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    if path.is_dir():
        path = path / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(row.get(key) for key in KEYS): row for row in rows}


def number(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def delta(base: float | None, candidate: float | None, *, lower_is_better: bool) -> float | None:
    if base is None or candidate is None or base == 0:
        return None
    raw = (candidate / base - 1.0) * 100.0
    return -raw if lower_is_better else raw


def shown(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-below",
        type=float,
        help="Exit nonzero if a common decode case has a TPS improvement below this percent.",
    )
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    common = sorted(baseline.keys() & candidate.keys(), key=lambda key: tuple(map(str, key)))
    missing_candidate = sorted(baseline.keys() - candidate.keys(), key=lambda key: tuple(map(str, key)))
    candidate_only = sorted(candidate.keys() - baseline.keys(), key=lambda key: tuple(map(str, key)))

    lines = [
        "# Benchmark comparison",
        "",
        "Positive percentages are improvements: higher TPS or lower latency.",
        "",
        "| Config | Workload | In/out | C | Output TPS | TPS delta | TTFT delta | TPOT delta | Accept % | Accept delta | Accept len |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    failed: list[str] = []
    for key in common:
        base = baseline[key]
        cand = candidate[key]
        tps = number(cand, "median_output_throughput")
        tps_delta = delta(
            number(base, "median_output_throughput"), tps, lower_is_better=False
        )
        ttft_delta = delta(
            number(base, "median_p50_ttft_ms"),
            number(cand, "median_p50_ttft_ms"),
            lower_is_better=True,
        )
        tpot_delta = delta(
            number(base, "median_p50_tpot_ms"),
            number(cand, "median_p50_tpot_ms"),
            lower_is_better=True,
        )
        accept = number(cand, "median_spec_decode_acceptance_rate")
        base_accept = number(base, "median_spec_decode_acceptance_rate")
        accept_delta = (
            None if accept is None or base_accept is None else accept - base_accept
        )
        accept_len = number(cand, "median_spec_decode_acceptance_length")
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]}/{key[3]} | {key[4]} "
            f"| {'' if tps is None else f'{tps:.2f}'} | {shown(tps_delta)} "
            f"| {shown(ttft_delta)} | {shown(tpot_delta)} "
            f"| {'' if accept is None else f'{accept:.2f}'} "
            f"| {'' if accept_delta is None else f'{accept_delta:+.2f} pp'} "
            f"| {'' if accept_len is None else f'{accept_len:.2f}'} |"
        )
        if (
            args.fail_below is not None
            and key[1] == "decode"
            and tps_delta is not None
            and tps_delta < args.fail_below
        ):
            failed.append(f"{key[0]} c={key[4]}: {tps_delta:+.2f}%")

    if missing_candidate:
        lines.extend(("", "## Missing from candidate", ""))
        lines.extend(f"- `{key}`" for key in missing_candidate)
    if candidate_only:
        lines.extend(("", "## Candidate-only cases", ""))
        lines.extend(f"- `{key}`" for key in candidate_only)

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    if failed:
        raise SystemExit("Regression gate failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
