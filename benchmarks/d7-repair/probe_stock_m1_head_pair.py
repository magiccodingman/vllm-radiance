"""Small, stage-only exactness and graph-timing gate for the M4-pair head."""

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import private_json, seal, write_private


def qualify(args):
    spec = private_json(args.spec)
    os.environ.update(spec["environment"])
    import torch
    from safetensors import safe_open
    from stock_m1_head_pair import StockM1HeadPair
    from vllm import _custom_ops as ops

    torch.set_num_threads(2)
    properties = torch.cuda.get_device_properties(0)
    require("gfx1201" in properties.gcnArchName, "candidate requires gfx1201")
    cu_count = properties.multi_processor_count
    model = Path(spec["native_config"]["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    with safe_open(str(model / index["weight_map"]["lm_head.weight"]), framework="pt") as source:
        weight = source.get_tensor("lm_head.weight").to("cuda")
    candidate = StockM1HeadPair(args.build)

    def grouped(x, size):
        return torch.cat(
            [ops.wvSplitK(weight, x[i : i + size], cu_count, None) for i in range(0, 8, size)]
        )

    samples, sources = [], []
    for path in sorted(args.captures.glob("head-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        hidden = saved["hidden"]
        if hidden.shape != (8, 5120):
            continue
        samples.append(hidden.to("cuda"))
        sources.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if len(samples) == args.samples:
            break
    require(len(samples) == args.samples, "insufficient consecutive M8 captures")
    torch.manual_seed(6137)
    synthetic = [torch.zeros_like(samples[0]), torch.randn_like(samples[0]), samples[0] * 2]
    checks = []

    def mismatch(a, b):
        return int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))

    for number, hidden in enumerate(samples + synthetic):
        expected = grouped(hidden, 1)
        baseline = grouped(hidden, 4)
        actual = candidate(weight, hidden)
        record = {
            "sample": number,
            "rows": 8,
            "elements": expected.numel(),
            "private_pi": number < len(samples),
            "baseline_mismatches": mismatch(baseline, expected),
            "candidate_mismatches": mismatch(actual, expected),
        }
        checks.append(record)
        write_private(args.output / f"check-{number:02d}.json", seal(record))
        require(record["baseline_mismatches"] == 0, "grouped baseline differs from M1")
        require(record["candidate_mismatches"] == 0, "head candidate differs from M1")
    broken = actual.clone()
    broken.view(torch.int16)[0, 0] ^= 1
    require(mismatch(broken, expected) == 1, "negative control was not detected")

    hidden = samples[0]
    functions = {
        "two_m4_launches": lambda: grouped(hidden, 4),
        "interleaved_pair": lambda: candidate(weight, hidden),
    }
    timings = {}
    graph_checks = {}
    for name, function in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        require(mismatch(output, grouped(hidden, 1)) == 0, "captured graph output differs from M1")
        graph_checks[name] = True
        functions[name] = (graph, output)
        timings[name] = []
    randomizer = random.Random(9237)
    for _ in range(30):
        order = list(functions)
        randomizer.shuffle(order)
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            functions[name][0].replay()
            end.record()
            end.synchronize()
            timings[name].append(begin.elapsed_time(end))
    return {
        "checks": checks,
        "private_capture_hashes": sources,
        "private_pi_rows": 8 * len(samples),
        "synthetic_rows": 8 * len(synthetic),
        "negative_control_detected": True,
        "graph_checks": graph_checks,
        "timings": {
            k: {"median_ms": statistics.median(v), "samples_ms": v} for k, v in timings.items()
        },
        "build": candidate.manifest["sha256"],
        "compute_units": cu_count,
        "native_operator": "vllm._rocm_C.wvSplitK",
        "reference": "eight serial M1 calls",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["spec", "captures", "build", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(
        args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU admission required"
    )
    require(1 <= args.samples <= 32, "small stage probe accepts at most 32 captures")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = qualify(args)
    report = seal(
        {
            "status": "SAMPLE_CHECKED",
            **result,
            "scope": "small head-only sample; no full-model or universal-equivalence claim",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "result.json", report)
    print(
        json.dumps(
            {"status": report["status"], "timings": report["timings"], "sha256": report["sha256"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
