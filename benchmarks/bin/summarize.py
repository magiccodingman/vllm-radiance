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
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def summarize_telemetry(run_root: Path) -> None:
    """Promote raw telemetry into per-case/per-device headroom records."""

    records: list[dict[str, object]] = []
    fields = {
        "GPU use (%)": "peak_gpu_use_percent",
        "Average Graphics Package Power (W)": "peak_average_power_w",
        "Temperature (Sensor edge) (C)": "peak_edge_temperature_c",
        "Temperature (Sensor junction) (C)": "peak_junction_temperature_c",
        "Temperature (Sensor memory) (C)": "peak_memory_temperature_c",
    }
    for path in sorted(run_root.glob("**/telemetry/*.jsonl")):
        devices: dict[str, dict[str, object]] = {}
        min_host_available: float | None = None
        visible_devices: set[str] | None = None
        manifest_path = path.parent.parent / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            visible = manifest.get("environment", {}).get("HIP_VISIBLE_DEVICES")
            if isinstance(visible, str) and visible.strip():
                visible_devices = {
                    f"card{index.strip()}"
                    for index in visible.split(",")
                    if index.strip().isdigit()
                }
        except (OSError, json.JSONDecodeError):
            pass
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            available = numeric(sample.get("host_memory", {}).get("available_bytes"))
            if available is not None:
                min_host_available = (
                    available
                    if min_host_available is None
                    else min(min_host_available, available)
                )
            for gpu in sample.get("gpus", []):
                device = str(gpu.get("device", "unknown"))
                if visible_devices is not None and device not in visible_devices:
                    continue
                row = devices.setdefault(
                    device,
                    {
                        "file": str(path.relative_to(run_root)),
                        "device": device,
                        "samples": 0,
                    },
                )
                row["samples"] = int(row["samples"]) + 1
                total = numeric(gpu.get("VRAM Total Memory (B)"))
                used = numeric(gpu.get("VRAM Total Used Memory (B)"))
                if total is not None:
                    row["vram_total_bytes"] = int(total)
                if total is not None and used is not None:
                    row["peak_vram_used_bytes"] = max(
                        int(row.get("peak_vram_used_bytes", 0)), int(used)
                    )
                    free = int(total - used)
                    row["minimum_vram_free_bytes"] = min(
                        int(row.get("minimum_vram_free_bytes", free)), free
                    )
                    used_percent = used / total * 100 if total else 0.0
                    row["peak_vram_used_percent"] = max(
                        float(row.get("peak_vram_used_percent", 0.0)), used_percent
                    )
                for source, target in fields.items():
                    value = numeric(gpu.get(source))
                    if value is not None:
                        row[target] = max(float(row.get(target, value)), value)
        for row in devices.values():
            if min_host_available is not None:
                row["minimum_host_available_bytes"] = int(min_host_available)
            records.append(row)

    (run_root / "telemetry-summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_root / "telemetry-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = sorted({key for row in records for key in row})
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--telemetry-only", action="store_true")
    args = parser.parse_args()

    if args.telemetry_only:
        summarize_telemetry(args.run_root)
        return

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
    summarize_telemetry(args.run_root)


if __name__ == "__main__":
    main()
