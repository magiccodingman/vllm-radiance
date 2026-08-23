#!/usr/bin/env python3
"""Build measurement and aggregate tables from vLLM benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "duration",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p50_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p50_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p99_e2el_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def numeric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.run_root.glob("**/raw/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row: dict[str, object] = {
            "file": str(path.relative_to(args.run_root)),
            "config": data.get("config", path.parent.parent.name),
            "workload": data.get("workload", "unknown"),
            "input_tokens": data.get("input_tokens", ""),
            "output_tokens": data.get("output_tokens", ""),
            "concurrency": data.get("max_concurrency", ""),
            "repetition": data.get("repetition", ""),
            "tp": data.get("tp", ""),
            "spec": data.get("spec", ""),
            "cpu_offload_gb": data.get("cpu_offload_gb", ""),
            "temperature": data.get("temperature", ""),
            "num_prompts": data.get("num_prompts", ""),
        }
        for metric in METRICS:
            row[metric] = data.get(metric, "")
        rows.append(row)

    columns = [
        "file", "config", "workload", "input_tokens", "output_tokens",
        "concurrency", "repetition", "tp", "spec", "cpu_offload_gb", "temperature",
        "num_prompts", *METRICS,
    ]
    csv_path = args.run_root / "measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["config"], row["workload"], row["input_tokens"],
            row["output_tokens"], row["concurrency"], row["tp"], row["spec"],
            row["cpu_offload_gb"],
            row["temperature"],
        )
        grouped[key].append(row)

    aggregates: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate: dict[str, object] = {
            "config": key[0], "workload": key[1], "input_tokens": key[2],
            "output_tokens": key[3], "concurrency": key[4], "tp": key[5],
            "spec": key[6], "cpu_offload_gb": key[7], "samples": len(samples),
            "temperature": key[8],
        }
        for metric in METRICS:
            values = [value for row in samples if (value := numeric(row.get(metric))) is not None]
            if values:
                aggregate[f"median_{metric}"] = statistics.median(values)
                if metric == "output_throughput":
                    mean = statistics.mean(values)
                    aggregate["output_throughput_cv_percent"] = (
                        statistics.stdev(values) / mean * 100 if len(values) > 1 and mean else 0.0
                    )
        aggregates.append(aggregate)

    (args.run_root / "summary.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        aggregate_columns = sorted({key for row in aggregates for key in row})
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        writer.writerows(aggregates)

    lines = [
        "# Benchmark summary",
        "",
        "Medians across repetitions. TPS is output-token throughput; TPOT and TTFT are milliseconds.",
        "",
        "| Config | Workload | In/out | C | Temp | N | Output TPS | Total TPS | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Spec accept % | CV % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def shown(name: str) -> str:
            value = row.get(name)
            return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.2f}"

        lines.append(
            f"| {row['config']} | {row['workload']} | {row['input_tokens']}/{row['output_tokens']} "
            f"| {row['concurrency']} | {row['temperature']} | {row['samples']} | {shown('median_output_throughput')} "
            f"| {shown('median_total_token_throughput')} | {shown('median_p50_ttft_ms')} "
            f"| {shown('median_p99_ttft_ms')} | {shown('median_p50_tpot_ms')} "
            f"| {shown('median_p99_tpot_ms')} | {shown('median_spec_decode_acceptance_rate')} "
            f"| {shown('output_throughput_cv_percent')} |"
        )
    (args.run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
