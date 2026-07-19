#!/usr/bin/env python3
"""Build-time source patch: RADIANCE bf16/fp16 (2-byte) KV attention config for aiter's unified
attention 3D decode kernel on RDNA (gfx1201 / R9700). Idempotent string-replacement on the installed
site-packages copy of unified_attention.py.

Why a source patch, not the RADIANCE_ATTN_TUNE runtime wrapper: that wrapper on select_3d_config is
silently bypassed for the bf16 3D decode path in-serve (it fires for fp8 and the 2D path, not this
one), so the fix must live in source. LDS-critical, not just a tune: a 2-byte KV cache stages a
TILE_SIZE x next_pow2(head_size) K+V tile num_stages-deep in shared memory, so at head_size 256 the
aiter default (TILE 64, stages 2) needs 64*256*2B*2 + 256 = 65792 B > the R9700's 64 KiB (65536 B)
LDS -> Triton OutOfResources at cudagraph capture. do_bench-optimal (head-256): TILE=16 warps=4
stages=2 waves=2 (warps=4 the key lever); fits (16*256*2*2 = 16384 B).
"""
import ast
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
F = SP / "aiter/ops/triton/attention/unified_attention.py"

# The 8/12-space indent uniquely targets select_3d_config's RDNA branch (select_2d_config's identical
# elif is at 4/8-space, so it is not matched; do NOT rely on replace-count alone).
ANCHOR = (
    "        elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:\n"
    "            TILE_SIZE = max(32, TILE_SIZE)\n"
)
INSERT = (
    "        elif kv_cache_dtype in (torch.bfloat16, torch.float16):\n"
    "            # --- RADIANCE 2-byte (bf16/fp16, incl. --kv-cache-dtype auto) KV, gfx1201 ---\n"
    "            # aiter default (TILE 64, stages 2) needs 64*256*2B*2 + 256 = 65792 B > 64 KiB LDS at\n"
    "            # head_size 256 -> Triton OutOfResources at cudagraph capture. do_bench-optimal:\n"
    "            # TILE16 warps4 stages2 waves2 (warps4 = +14%), reduce warps4; fits (16384 B).\n"
    "            TILE_SIZE = 16\n"
    "            attn_warps = 4\n"
    "            attn_stages = 2\n"
    "            waves_per_eu = 2\n"
    "            reduce_num_warps = 4\n"
    "            if TILE_SIZE * triton.next_power_of_2(head_size) * 2 * attn_stages + 256 > 65536:\n"
    "                attn_stages = 1  # LDS-fit fallback for head_size > 256\n"
)


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
    apply(F, ANCHOR, ANCHOR + INSERT, "RADIANCE 2-byte", "unified_attention bf16 3D config")


if __name__ == "__main__":
    main()
