"""Run an untimed native-call replay pilot on the pinned, repaired M8 path."""

import hashlib
import sys
from pathlib import Path

import benchmark_optimized_d7 as benchmark

from qwen_r9700_lab.conformance_topk import require

BASE_DRIVER_SHA256 = "45afe12e4bffadad2e39c94b2b0955bea547269646c45f0bb6a181ec7f6037b6"
BASE_CONFIG = benchmark.make_config


def make_config(spec, lane, **kwargs):
    require(lane in ("old-bf16", "fixed-bf16"), "full-head catalog required")
    require(kwargs.get("execution_mode") == "compiled-no-graphs", "compiled tape required")
    require(not kwargs.get("isolated_capture"), "tape and storage capture are separate modes")
    result = BASE_CONFIG(spec, lane, **kwargs)
    result["worker_cls"] = "native_d7_catalog_worker.CatalogWorker"
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
