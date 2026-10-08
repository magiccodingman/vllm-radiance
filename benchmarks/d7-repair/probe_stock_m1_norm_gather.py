"""Small norm-only M1 exactness gate and captured-graph timings."""

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
    os.environ.update(private_json(args.spec)["environment"])
    import torch
    from stock_m1_norm import StockM1Norm
    from vllm.ir.ops import fused_add_rms_norm, rms_norm

    torch.set_num_threads(2)
    torch.manual_seed(96173)
    native = (rms_norm.impls["native"].impl_fn, fused_add_rms_norm.impls["native"].impl_fn)
    methods = {
        "distributed_barriers": StockM1Norm(args.baseline),
        "gathered_tree": StockM1Norm(args.build),
    }
    checks, timings = [], {}
    inputs = []
    for count, amplitude, groups in [
        (1, 1, 1),
        (2, 1, 1),
        (8, 0, 1),
        (8, 0.001, 1),
        (8, 1, 1),
        (8, 10000, 1),
        (8, 1, 4),
        (8, 1, 24),
    ]:
        width = 5120 if groups == 1 else 256
        shape = (count, width) if groups == 1 else (count, groups, width)
        inputs.append((torch.randn(shape, device="cuda").bfloat16() * amplitude, groups))
    # A padded row stride exercises the actual noncontiguous input contract.
    inputs.append((torch.randn((8, 5120 * 2), device="cuda").bfloat16()[:, :5120], 1))

    def mismatch(a, b):
        kind = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        return int(torch.count_nonzero(a.view(kind) != b.view(kind)))

    for index, (x, groups) in enumerate(inputs):
        weight = (torch.randn(x.shape[-1], device="cuda") * 0.1).bfloat16()
        for residual in (None, torch.randn_like(x)) if groups == 1 else (None,):
            expected, carries, expected_moments = [], [], []
            for row in range(x.shape[0]):
                single = x[row : row + 1]
                z = single.float()
                if residual is None:
                    expected.append(native[0](single, weight.float() + 1, 1e-6))
                else:
                    out, carry = native[1](
                        single, residual[row : row + 1], weight.float() + 1, 1e-6
                    )
                    expected.append(out)
                    carries.append(carry)
                    z = z + residual[row : row + 1].float()
                variance = z.pow(2).mean(-1).reshape(-1)
                expected_moments.append(
                    torch.stack([variance, torch.rsqrt(variance + 1e-6)], dim=-1)
                )
            expected = torch.cat(expected)
            expected_moments = torch.cat(expected_moments)
            for name, method in methods.items():
                moments = torch.empty_like(expected_moments)
                actual = method(x, residual, weight, 1e-6, moments=moments)
                out = actual if residual is None else actual[0]
                record = {
                    "input": index,
                    "method": name,
                    "rows": x.shape[0],
                    "groups": groups,
                    "residual": residual is not None,
                    "output_mismatches": mismatch(out, expected),
                    "moment_mismatches": mismatch(moments, expected_moments),
                    "carry_mismatches": 0
                    if residual is None
                    else mismatch(actual[1], torch.cat(carries)),
                }
                checks.append(record)
                write_private(
                    args.output / f"check-{index}-{int(residual is not None)}-{name}.json",
                    seal(record),
                )
                require(
                    not any(
                        record[k]
                        for k in ("output_mismatches", "moment_mismatches", "carry_mismatches")
                    ),
                    "normalization differs from native M1",
                )
            broken = out.clone()
            broken.view(torch.int16).reshape(-1)[0] ^= 1
            require(mismatch(broken, expected) == 1, "negative control failed")

    for residual_on in (False, True):
        x = inputs[4][0]
        residual = torch.randn_like(x) if residual_on else None
        weight = torch.randn(5120, device="cuda").bfloat16()
        graphs, samples = {}, {name: [] for name in methods}
        for name, method in methods.items():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(4):
                    method(x, residual, weight, 1e-6)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(64):
                    out = method(x, residual, weight, 1e-6)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graphs[name] = (graph, out)
        a, b = [v[1] for v in graphs.values()]
        require(
            mismatch(a[0] if residual_on else a, b[0] if residual_on else b) == 0,
            "captured norm graph changed output",
        )
        rng = random.Random(81613)
        for _ in range(30):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 64)
        timings[str(residual_on)] = {
            k: {"median_ms": statistics.median(v), "samples_ms": v} for k, v in samples.items()
        }
    return {
        "checks": checks,
        "timings": timings,
        "build": methods["gathered_tree"].manifest["sha256"],
        "baseline": methods["distributed_barriers"].manifest["sha256"],
        "negative_control_detected": True,
        "graph_checks": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "baseline", "build", "output"):
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
            "scope": "small native-M1 normalization sample; no universal-equivalence claim",
        }
    )
    write_private(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sha256": report["sha256"],
                "timings": {
                    k: {n: v["median_ms"] for n, v in t.items()}
                    for k, t in report["timings"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
