#!/usr/bin/env python3
"""CPU arithmetic check for bounded draft-head packing; requires installed Torch.

Runs actual repository functions without importing their Triton kernels. An
independent whole-matrix reference detects packing/scale drift across chunks.
This is not a native kernel or speculative model qualification.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import types
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    tree = ast.parse((ROOT / "radiance_drafthead.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_head_is_empty", "_quantize_head_now")]
    ns = {"torch": torch, "types": types, "GROUP": 128, "BLOCK_N": 64,
          "KCAND": 8, "RERANK": 64, "_apply_head_int2": lambda *a: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "drafthead-functions", "exec"), ns)
    empty = ns["_head_is_empty"]
    zeros = torch.zeros((8193, 2))
    zeros[0, 0] = -0.0
    assert empty(zeros)
    zeros[-1, -1] = 1.0
    assert not empty(zeros)
    zeros[-1, -1] = float("nan")
    assert not empty(zeros)
    torch.manual_seed(33)
    original_capture = torch.cuda.is_current_stream_capturing
    original_empty = torch.cuda.empty_cache
    # No HIP calls: only allocator control is stubbed; real CPU arithmetic runs.
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.empty_cache = lambda: None
    try:
        for bits in (2, 4):
            ns.update(BITS=bits, _HEAD_CACHE={})
            w = torch.randn((2051, 256), dtype=torch.bfloat16)
            w[0] = 0
            w[1024] = 1
            lp = SimpleNamespace()
            ns["_quantize_head_now"](lp, SimpleNamespace(weight=w))
            grouped = w.float().reshape(2051, 2, 128)
            lo, hi = grouped.amin(-1), grouped.amax(-1)
            levels = (1 << bits) - 1
            scale = ((hi - lo) / levels).clamp(min=1e-8)
            zero = (-lo / scale).round().clamp(0, levels)
            q = (grouped / scale[..., None] + zero[..., None]).round().clamp(0, levels)
            q = q.to(torch.uint8).reshape_as(w)
            # Independent lane assembly rather than the optimized slicing code.
            lanes = 8 // bits
            packed = torch.zeros((2051, 256 // lanes), dtype=torch.uint8)
            for lane in range(lanes):
                packed |= q[:, lane * packed.shape[1]:(lane + 1) * packed.shape[1]] << (lane * bits)
            bias = 4 if bits == 2 else 16
            assert torch.equal(lp._radiance_wq, packed)
            assert torch.equal(lp._radiance_scale, (bias * scale).bfloat16())
            assert torch.equal(lp._radiance_zs, (bias * scale + zero * scale).bfloat16())
            reused = SimpleNamespace()
            ns["_quantize_head_now"](reused, SimpleNamespace(weight=w))
            assert reused._radiance_wq.data_ptr() == lp._radiance_wq.data_ptr()
    finally:
        torch.cuda.is_current_stream_capturing = original_capture
        torch.cuda.empty_cache = original_empty
    print("INT2/INT4 draft-head bounded packing and reuse: PASS (CPU only)")


if __name__ == "__main__":
    main()
