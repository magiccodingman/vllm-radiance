#!/usr/bin/env python3
"""Build-time source patch for AITER's unified attention on RDNA (gfx1201 / R9700). Three idempotent
string replacements on the installed site-packages copy of unified_attention.py.

1. LDS fit (correctness, unconditional, both config selectors). The attention kernels stage a
   TILE_SIZE x next_pow2(head_size) K/V tile num_stages deep in shared memory, at the KV cache element
   size, plus ~256 B. AITER sizes that tile for CDNA's much larger LDS; the R9700 has 64 KiB, so
   several of its own picks do not fit and Triton raises OutOfResources at cudagraph capture:
     head_size 256, 2-byte KV : 64*256*2*2 + 256 = 65792
     head_size 512, fp8    KV : 64*512*1*2 + 256 = 65792   (Gemma4's global-attention layers)
   Both selectors now step num_stages, then TILE_SIZE, down until the tile fits. This is a hard
   requirement, not a preference, so it lives in source and applies whether or not the runtime
   RADIANCE_ATTN_TUNE hook is installed (that hook is a tune and must stay disableable).

2. bf16/fp16 (2-byte, incl. --kv-cache-dtype auto) 3D-decode tune. do_bench-optimal at head_size 256:
   TILE 16, warps 4, stages 2, waves 2, reduce warps 4 (warps=4 is the lever: +14% decode, 4-7x
   prefill). This must be a source patch rather than the RADIANCE_ATTN_TUNE wrapper because that
   wrapper is bypassed for the bf16 3D path in-serve.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
F = SP / "aiter/ops/triton/attention/unified_attention.py"


def _fit(tag, stages_var):
    """Clamp source for one selector: shrink pipeline depth first, then the tile."""
    return (
        f"    # --- RADIANCE LDS fit ({tag}): shrink the staged K/V tile into the R9700's 64 KiB LDS ---\n"
        "    _rad_el = 2 if kv_cache_dtype in (torch.bfloat16, torch.float16) else 1\n"
        "    _rad_hs = triton.next_power_of_2(head_size)\n"
        f"    while {stages_var} > 1 and TILE_SIZE * _rad_hs * _rad_el * {stages_var} + 256 > 65536:\n"
        f"        {stages_var} -= 1\n"
        f"    while TILE_SIZE > 16 and TILE_SIZE * _rad_hs * _rad_el * {stages_var} + 256 > 65536:\n"
        "        TILE_SIZE //= 2\n"
        "\n"
    )


# 1a. select_3d_config. Inserted before the gather-mode block so that block's
# `NUM_BLOCKS_GATHER_PER_TILE = TILE_SIZE // block_size` is recomputed from the clamped tile.
A3 = (
    "    if NUM_BLOCKS_GATHER_PER_TILE > 1:\n"
    "        # force gather mode\n"
)

# 1b. select_2d_config returns its config dict directly; nothing downstream depends on the tile.
A2 = (
    "    return {\n"
    '        "BLOCK_M": BLOCK_M,\n'
    '        "BLOCK_Q": BLOCK_Q,\n'
)

# 2. The 8/12-space indent uniquely targets select_3d_config's RDNA branch (select_2d_config's
# identical elif is at 4/8-space, so it is not matched; do NOT rely on replace-count alone).
ANCHOR = (
    "        elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:\n"
    "            TILE_SIZE = max(32, TILE_SIZE)\n"
)
INSERT = (
    "        elif kv_cache_dtype in (torch.bfloat16, torch.float16):\n"
    "            # --- RADIANCE 2-byte (bf16/fp16, incl. --kv-cache-dtype auto) KV, gfx1201 ---\n"
    "            # do_bench-optimal at head_size 256: TILE16 warps4 stages2 waves2 (warps4 = +14%\n"
    "            # decode, 4-7x prefill), reduce warps4. The LDS fit above keeps this in bounds.\n"
    "            TILE_SIZE = 16\n"
    "            attn_warps = 4\n"
    "            attn_stages = 2\n"
    "            waves_per_eu = 2\n"
    "            reduce_num_warps = 4\n"
)


def main():
    apply(F, A3, _fit("3D", "attn_stages") + A3, "RADIANCE LDS fit (3D)", "unified_attention LDS fit (3D)")
    apply(F, A2, _fit("2D", "num_stages_2d") + A2, "RADIANCE LDS fit (2D)", "unified_attention LDS fit (2D)")
    apply(F, ANCHOR, ANCHOR + INSERT, "RADIANCE 2-byte", "unified_attention bf16 3D-decode tune")


if __name__ == "__main__":
    main()
