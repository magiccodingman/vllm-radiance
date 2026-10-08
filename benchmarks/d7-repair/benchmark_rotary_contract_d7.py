"""Controlled eager M8 diagnostic with explicit nearest-even RoPE products."""

import hashlib
import sys
from pathlib import Path

import benchmark_optimized_d7 as benchmark

from qwen_r9700_lab.conformance_topk import require

BASE_DRIVER_SHA256 = "45afe12e4bffadad2e39c94b2b0955bea547269646c45f0bb6a181ec7f6037b6"
BASE_CONFIG = benchmark.make_config


def make_config(spec, lane, **kwargs):
    require(lane == "fixed-bf16", "rotary intervention requires fixed full-head lane")
    require(kwargs.get("execution_mode") == "eager", "rotary intervention requires eager")
    require(kwargs.get("speculation", True), "rotary intervention currently admits M8 only")
    result = BASE_CONFIG(spec, lane, **kwargs)
    capture = kwargs.get("isolated_capture", False)
    expected = "execution_mode_d7_worker.ExecutionMode" + ("CaptureWorker" if capture else "Worker")
    require(result["worker_cls"] == expected, "underlying diagnostic worker changed")
    result["worker_cls"] = "rotary_mode_d7_worker.RotaryRne" + (
        "CaptureWorker" if capture else "Worker"
    )
    return result


def main():
    require("--correctness" in sys.argv and "--profile" not in sys.argv, "forced replay required")
    require(
        hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest() == BASE_DRIVER_SHA256,
        "underlying benchmark driver changed",
    )
    benchmark.make_config = make_config
    benchmark.__file__ = __file__
    benchmark.main()


if __name__ == "__main__":
    main()
