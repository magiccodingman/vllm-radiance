#!/usr/bin/env python3
"""Sample AMD GPU and host memory telemetry into JSON Lines."""

from __future__ import annotations

import argparse
import csv
import io
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts:
                values[key] = int(parts[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


def gpu_metrics() -> list[dict[str, str]]:
    command = [
        "rocm-smi",
        "--showuse",
        "--showmeminfo",
        "vram",
        "--showpower",
        "--showtemp",
        "--showclocks",
        "--csv",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.startswith(("device,", "card"))]
    if not lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        while running:
            sample = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch_seconds": time.time(),
                "host_memory": meminfo(),
                "gpus": gpu_metrics(),
            }
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            deadline = time.monotonic() + args.interval
            while running and time.monotonic() < deadline:
                time.sleep(min(0.2, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
