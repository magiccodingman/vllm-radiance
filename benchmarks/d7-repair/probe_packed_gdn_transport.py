"""Small all-state check for removing convolution/GDN packing copies."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


def qualify(args):
    os.environ.update(private_json(args.spec)["environment"])
    import torch
    from packed_gdn_transport import load
    from stock_gdn_convolution_adapter import StockConvolutionAdapter
    from stock_gdn_indexed_adapter import StockIndexedAdapter
    from stock_gdn_runtime import validate_manifest

    torch.set_num_threads(2)
    torch.manual_seed(71971)
    repair = validate_manifest(args.repair)
    spec = importlib.util.spec_from_file_location("transport_conv_reference", repair["convolution"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fast_conv, fast_recur = load(args.build, torch, module.causal_conv1d_update)
    slow_conv = StockConvolutionAdapter(torch, module.causal_conv1d_update)
    slow_recur = StockIndexedAdapter(torch)
    x = torch.randn((8, 10240), device="cuda").bfloat16()
    weight = (torch.randn((10240, 4), device="cuda") * 0.1).bfloat16()
    original_history = torch.randn((2, 10240, 10), device="cuda").bfloat16()
    original_state = torch.randn((10, 48, 128, 128), device="cuda") * 0.05
    histories = [original_history.clone() for _ in range(2)]
    states = [original_state.clone() for _ in range(2)]
    a, b = [torch.randn((8, 48), device="cuda").bfloat16() for _ in range(2)]
    log = torch.randn(48, device="cuda")
    bias = torch.randn(48, device="cuda").bfloat16().float()
    conv_index = torch.tensor([1], dtype=torch.int32, device="cuda")
    indices = (torch.randperm(8, device="cuda", dtype=torch.int32) + 1).reshape(1, 8)
    cu = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    accepted = torch.ones(1, dtype=torch.int32, device="cuda")
    outputs = [torch.empty((8, 48, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def step(number):
        conv, recur = (slow_conv, slow_recur) if number == 0 else (fast_conv, fast_recur)
        qkv = conv(
            x, weight, None, histories[number], 10, conv_index, accepted, cu, 1, 8, 48, 16, 8
        )
        recur(
            *qkv,
            a,
            b,
            log,
            bias,
            states[number],
            outputs[number],
            cu,
            indices,
            accepted,
            1,
            48,
            16,
            128**-0.5,
        )
        return qkv

    def reset():
        for h, s in zip(histories, states, strict=True):
            h.copy_(original_history)
            s.copy_(original_state)

    def mismatch(left, right):
        dtype = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
        return int(torch.count_nonzero(left.view(dtype) != right.view(dtype)))

    checks = []
    for previous in range(1, 9):
        accepted.fill_(previous)
        reset()
        old = step(0)
        new = step(1)
        record = {
            "previous_acceptance": previous,
            "rows": 8,
            "qkv_mismatches": sum(mismatch(a, b) for a, b in zip(old, new, strict=True)),
            "output_mismatches": mismatch(*outputs),
            "history_mismatches": mismatch(*histories),
            "whole_state_pool_mismatches": mismatch(*states),
        }
        checks.append(record)
        write_private(args.output / f"check-{previous}.json", seal(record))
        require(
            not any(v for k, v in record.items() if k.endswith("mismatches")),
            "packed GDN transport changed numerical output/state",
        )
    broken = states[1].clone()
    broken.view(torch.int32)[1, 0, 0, 0] ^= 1
    require(mismatch(broken, states[0]) == 1, "state corruption control failed")
    separate_k = torch.empty_strided(
        new[1].shape, new[1].stride(), device=new[1].device, dtype=new[1].dtype
    )
    separate_k.copy_(new[1])
    try:
        fast_recur(
            new[0],
            separate_k,
            new[2],
            a,
            b,
            log,
            bias,
            states[1],
            outputs[1],
            cu,
            indices,
            accepted,
            1,
            48,
            16,
            128**-0.5,
        )
    except DiagnosticError:
        rejected = True
    else:
        rejected = False
    require(rejected, "independent Q/K/V storage must not masquerade as packed storage")

    graphs = []
    for number in range(2):
        reset()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step(number)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(number)
        graphs.append(graph)
    reset()
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    require(
        mismatch(*outputs) == mismatch(*histories) == mismatch(*states) == 0,
        "captured transport graph differs",
    )
    samples = [[], []]
    rng = random.Random(73161)
    for _ in range(50):
        order = [0, 1]
        rng.shuffle(order)
        for number in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[number].replay()
            end.record()
            end.synchronize()
            samples[number].append(start.elapsed_time(end))
    return {
        "checks": checks,
        "graph_checks": True,
        "negative_control_detected": True,
        "independent_storage_rejected": rejected,
        "build": private_json(args.build / "build.json")["sha256"],
        "reference_repair": repair["sha256"],
        "reference": "qualified stock-M1 arithmetic adapters; unchanged numerical kernels",
        "timings": {
            name: {"median_ms": statistics.median(v), "samples_ms": v}
            for name, v in zip(("split_copy_repack", "shared_packed_views"), samples, strict=True)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "repair", "build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = qualify(args)
    report = seal(
        {
            "status": "SAMPLE_CHECKED",
            **result,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "64 speculative positions; all 8 prior acceptance widths; not universal proof",
        }
    )
    write_private(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sha256": report["sha256"],
                "timings": {k: v["median_ms"] for k, v in report["timings"].items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
