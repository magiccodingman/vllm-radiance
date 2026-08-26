"""RADIANCE skinny bf16 GEMM dispatch. Routes small unquantized projections to the R4D gfx1201
kernel `gemm_bf16_nt_m64` (C[M,N] = x[M,K] @ W[N,K]^T, one 16x16x16 WMMA per row tile per 16 of K,
split-K reduced in LDS) from vLLM's ROCm unquantized-linear chokepoint. Enabled with the rest of the
library by RADIANCE_USE_R4D.

Two shapes qualify on this box, and they are the same problem twice: a projection whose weight is
under a megabyte, so rocBLAS lays it out as a handful of workgroups and leaves the GPU idle while
each of them walks the whole of K.

  * the MoE gate, W[256, 2048] -- Qwen3.6-35B-A3B;
  * the gated-delta-net b/a projection, W[96, 5120] -- every linear-attention layer of Qwen3.8-27B.
    It is bf16 in an otherwise fp8 checkpoint because 96 output columns cannot carry a 128-wide
    block scale, and there are 48 of them per step: 19.2 us each under a CUDA graph, against a
    960 KiB weight that the last-level cache holds outright.

M is capped at 64 by the kernel and floored at 6 by vLLM: below that its own wvSplitK already
handles the shape, and taking those away from it would be a regression, not a win. 64 covers the
whole decode band this box runs -- a batch of 8 streams verifying a block of 8.

A shape that is not in _CFG is declined. Each entry is a measured optimum for that (N, K), and a
split-K config carried over to a shape it was not measured on is a guess. The kernel is 2.4x to 5.8x
the torch path on the shapes below and, unlike the dot2 kernel it replaces, its cost is nearly flat
in M -- 3.4 us at M=8 and 3.6 us at M=64 on the b/a projection -- so one config per shape holds
across the band instead of one per M.
"""
import os
import sys

import torch

# RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py): with
# it off this module behaves exactly as it would in an image built without the library.
USE_R4D = os.environ.get("RADIANCE_USE_R4D", "1") == "1"
# RADIANCE_SKINNY_GEMM: 0 takes just this dispatch out without disabling the rest of the library,
# which is what an A/B of it against rocBLAS needs; "all" adds the shapes in _CFG_TAXED.
_MODE = os.environ.get("RADIANCE_SKINNY_GEMM", "1")
ENABLED = USE_R4D and _MODE != "0"
# Each distinct (N, K, M) this dispatch claims is logged once; a shape that never appears stayed on
# the path it would have taken without R4D.
_seen: set = set()

M_MIN = 6          # below this vLLM's own wvSplitK serves the shape

# (N, K) -> (WV, SK, MB): column tiles per block, k-splits per column tile, row tiles per block.
# Measured under a CUDA graph over M = 6..64; the optimum did not move with M on any of them.
_CFG = {
    # The MoE gate (Qwen3.6-35B-A3B): 9.6 us -> 3.2 us.
    (256, 2048): (1, 16, 1),
}

# Shapes where the kernel is faster and the SERVE is not, because a speculative drafter is reading
# what it perturbs. The gated-delta-net b/a projection is 18.7 us -> 3.4 us and -3.3% on the decode
# step, but acceptance falls further than that: 4.205 -> 3.892 at c1, and below the baseline in all
# three independent text samples it was checked against.
#
# It is not a correctness problem. Checked inside the serve against the linear it replaces, on real
# weights and real activations, the kernel matches bit for bit on most calls and differs on ONE
# element in 432 on the rest, at a bf16 ULP. Three different accumulation orders (no split-K, 8
# splits, 20 splits) all land at 3.78-3.93, so it is not this particular rounding either: a
# perturbation of the target's gates -- any perturbation, however small and however unbiased --
# moves the hidden states the drafter was fitted to, and that loss is one-sided. The control is the
# act-scale change in radiance_kernels, which alters the schedule but not one output bit and leaves
# acceptance identical to the digit.
#
# RADIANCE_SKINNY_GEMM=all enables them anyway -- for a serve with no drafter, where the tax does
# not exist, the step-time win is real.
_CFG_TAXED = {
    # in_proj_ba. TP-sharded (the replicate-instead path upstream is gated on is_cuda()), so N is 48
    # per rank at TP=2 and 96 at TP=1; both are measured.
    (48, 5120): (1, 8, 1),
    (96, 5120): (1, 8, 1),
    # The DFlash2 drafter's two unquantized projections -- ReplicatedLinear with quant_config=None
    # upstream, so bf16 and NOT TP-sharded: every rank reads all of them, ten kernel_projections
    # and one hidden_projection per draft pass, 131 MiB per rank per step. Timed against a POOLED
    # working set (one copy of either fits the 64 MiB last level, so a single-buffer bench measures
    # cache residency rather than the serve):
    #   [1280, 5120]  35.6 us -> 23.3 us  (369 -> 562 GB/s)  1.52x at M=8, 1.50x at M=64
    #   [ 256, 5120]  26.5 us ->  6.9 us  ( 99 -> 381 GB/s)  3.85x at M=8, 3.41x at M=64
    # In the serve that is a clean and reproducible -0.41% / -0.40% on the step over two text
    # samples -- and acceptance came in BELOW the control on both (4.044 -> 3.879, 4.150 -> 4.137).
    # Same shape of result as in_proj_ba above: the kernel differs from rocBLAS on 6 elements in
    # 10240 at a bf16 ULP, and this drafter is ULP-sensitive. Whether that is a real tax or the
    # trajectory re-roll any numeric change buys (a change that alters the text lands anywhere in a
    # ~6% acceptance band on this stack) cannot be told apart without a paired acceptance harness,
    # so these stay behind RADIANCE_SKINNY_GEMM=all with the other taxed shapes.
    (1280, 5120): (1, 8, 1),
    (256, 5120): (1, 8, 1),
}
if _MODE == "all":
    _CFG.update(_CFG_TAXED)

try:
    import r4d as _r4d                     # the compiled R4D kernel library
except Exception as e:                     # library missing or failed to load: stay on rocBLAS
    _r4d = None
    ENABLED = False
    sys.stderr.write(f"[radiance.gemm] r4d import failed, disabled: {e!r}\n")

_GEMM = None
_M_MAX = 0
if ENABLED:
    # The entry point carries the shape it is compiled for, so ask the library which kernel it has
    # rather than naming one here. M=64 excludes the older dot2 kernel, whose accumulators cap it at
    # 16; None means this build predates the WMMA one and every shape below stays on the path it
    # would have taken without R4D.
    _name = _r4d.select("gemm_nt", M=64, K=5120, dtype="bf16")
    if _name is None:
        ENABLED = False
        sys.stderr.write("[radiance.gemm] no gemm_nt kernel for M<=64 bf16, disabled\n")
    else:
        _GEMM = getattr(_r4d, _name)
        _M_MAX = int(_r4d.GEMM64_MAX_M)
        sys.stderr.write(
            "[radiance.gemm] skinny GEMM kernel ENABLED for M in [%d,%d]: %s\n"
            % (M_MIN, _M_MAX, ", ".join("%dx%d" % nk for nk in sorted(_CFG))))


def config_for(n: int, k: int, m: int):
    """(WV, SK, MB) for this shape, or None when nothing was measured for it."""
    if not (M_MIN <= m <= _M_MAX):
        return None
    return _CFG.get((n, k))


def maybe_gemm(x: torch.Tensor, weight: torch.Tensor):
    """C = x @ weight^T for a measured skinny shape, or None to leave the caller's path alone."""
    k = int(weight.shape[1])
    if x.shape[-1] != k:
        return None
    n = int(weight.shape[0])
    m = x.numel() // k
    cfg = config_for(n, k, m)
    if cfg is None:
        return None
    x2 = x.reshape(m, k)
    if not (x2.is_contiguous() and weight.is_contiguous()):
        return None
    if (n, k, m) not in _seen:
        _seen.add((n, k, m))
        sys.stderr.write("[radiance.gemm] claimed N=%d K=%d M=%d  WV=%d SK=%d MB=%d\n"
                         % (n, k, m, cfg[0], cfg[1], cfg[2]))
    c = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _GEMM(x2.data_ptr(), weight.data_ptr(), c.data_ptr(), m, k, n, cfg[0], cfg[1], cfg[2],
          torch.cuda.current_stream().cuda_stream)
    return c.reshape(*x.shape[:-1], n)
