#!/usr/bin/env python3
"""Lower the row count at which top-k/top-p uses the Triton kernel instead of a vocab-wide sort.

`apply_top_k_top_p` routes to Triton only at `logits.shape[0] >= 8` and otherwise runs
`logits.sort(dim=-1)` across the whole vocabulary, on the reasoning that sorting is cheap for a
few rows. On gfx1201 that reasoning does not hold: the sort is slower at *every* row count.
Measured in-image at vocab 248320, k=20, p=0.95, median of 5 interleaved passes (us/call):

    rows      1      2      4      5      8     10     16     20     40
    sort  235.3  775.0 1067.4  685.4  824.2  955.6 1525.0 1903.7 3995.5
  triton  209.0  209.1  209.4  210.9  199.8  210.1  214.7  222.9  508.0

Triton is flat to 20 rows because one program handles a row and the vocabulary dimension is what
costs; the sort's cost is not even monotonic in rows, since torch switches sort algorithm around
8 rows. Below the gate it is 3.3-5.1x slower.

The two paths are **bit-identical**, not merely close: same -inf mask and 0.000e+00 maximum
difference on the kept entries at every row count above. This is a scheduling choice with no
numerics attached, which is why it is defaulted on rather than put behind an off-by-default flag.

It matters on this build because speculative decode lands squarely under the gate. The spec path
calls `apply_top_k_top_p` once per step from `rejection_sampler.apply_sampling_constraints()` on
`batch x (SPEC+1)` rows -- 5 rows single-stream at the shipped SPEC=4 -- and the bonus-token
sampler calls it again on 1 row. Both took the sort. Note the gate is also the reason batch 8
profiled *faster* than batch 4 before this patch.

`RADIANCE_TOPK_TRITON_MIN_ROWS` keeps the threshold reachable; set it to 8 to restore upstream
behaviour, or higher to force the sort. Unrelated to the AITER sampler kernels, which stay off
here -- `TopKTopPSampler` gates those to MI3xx, and they cannot serve the rejection sampler
anyway because they return sampled ids rather than a masked logits tensor.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
TS = SP / "vllm" / "v1" / "sample" / "ops" / "topk_topp_sampler.py"

ANCHOR = (
    "    if HAS_TRITON and logits.shape[0] >= 8:\n"
    "        return apply_top_k_top_p_triton(logits, k, p)\n"
    "\n"
    "    # Use pytorch sort implementation for small batch sizes.\n"
    "    return apply_top_k_top_p_pytorch(logits, k, p)\n"
)

NEW = (
    "    # --- radiance (patch_topk_triton_rows.py): the >= 8 gate is wrong on gfx1201 ---\n"
    "    # Upstream sorts the whole vocabulary below 8 rows on the assumption that sorting a few\n"
    "    # rows is cheap. Measured here (vocab 248320, k=20, p=0.95) the Triton kernel is flat at\n"
    "    # ~210 us to 20 rows while the sort runs 235-1900 us and is not monotonic in rows, so the\n"
    "    # sort loses 3.3-5.1x exactly where speculative decode lives: the rejection sampler calls\n"
    "    # this once per step on batch x (SPEC+1) = 5 rows. The paths are bit-identical (same -inf\n"
    "    # mask, 0.0 max difference on kept entries), so this is scheduling only.\n"
    "    if HAS_TRITON and logits.shape[0] >= _RADIANCE_TRITON_MIN_ROWS:\n"
    "        return apply_top_k_top_p_triton(logits, k, p)\n"
    "\n"
    "    # Use pytorch sort implementation for small batch sizes.\n"
    "    return apply_top_k_top_p_pytorch(logits, k, p)\n"
)

# Module-level constant, inserted just above the function. `os` is not imported in this file and
# the surrounding code already uses inline __import__ for exactly this reason.
CONST_ANCHOR = "def apply_top_k_top_p(\n"
CONST_NEW = (
    "# radiance (patch_topk_triton_rows.py): row count at or above which apply_top_k_top_p uses the\n"
    "# Triton kernel rather than a vocab-wide sort. Upstream hardcodes 8; see the patch for the\n"
    "# measurements. Set to 8 to restore upstream behaviour.\n"
    "_RADIANCE_TRITON_MIN_ROWS = int(\n"
    "    __import__(\"os\").environ.get(\"RADIANCE_TOPK_TRITON_MIN_ROWS\") or 1\n"
    ")\n"
    "\n"
    "\n"
    "def apply_top_k_top_p(\n"
)

SENTINEL = "_RADIANCE_TRITON_MIN_ROWS"


def main():
    apply(TS, CONST_ANCHOR, CONST_NEW, SENTINEL, "top-k/top-p: Triton min-rows constant")
    apply(TS, ANCHOR, NEW, "the >= 8 gate is wrong on gfx1201",
          "top-k/top-p: route <8 rows to Triton")


if __name__ == "__main__":
    main()
