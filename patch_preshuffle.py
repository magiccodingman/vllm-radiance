#!/usr/bin/env python3
"""Preshuffle output-shape fix. BlockScaledMMLinearKernel.apply_weights computes
   output_shape = [*x.shape[:-1], weight.shape[0]]
but the preshuffle weight-shuffle-at-load rewrites the weight to [N//16, K*16], so weight.shape[0]
becomes N//16 and the final .view() fails. Use the true N stashed on the layer (_radiance_N) when the
layer was preshuffled; otherwise fall back to weight.shape[0] (baseline behaviour unchanged).
Idempotent. Run once pre-serve (in addition to the RADIANCE_PRESHUFFLE=1 load hook)."""
import ast
import sysconfig
from pathlib import Path

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py"
OLD = "        output_shape = [*x.shape[:-1], weight.shape[0]]"
NEW = '        output_shape = [*x.shape[:-1], getattr(layer, "_radiance_N", weight.shape[0])]  # radiance preshuffle: true N'


def apply(path, anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new` in `path`. Skips if
    `sentinel` is already present; a missing file or non-unique anchor is fatal."""
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({path})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    path.write_text(s)
    print(f"  OK    {label}")


def main():
    apply(F, OLD, NEW, "_radiance_N", "preshuffle output-shape fix")


if __name__ == "__main__":
    main()
