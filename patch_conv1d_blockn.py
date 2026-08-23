#!/usr/bin/env python3
"""Widen the gated-delta-net prefill causal-conv1d channel block from 256 to 1024 on gfx1201.

`causal_conv1d_fn` hardcodes `BLOCK_N=256` for the prefill kernel. With 128 threads that is two
bf16 per lane -- a 4-byte access -- and on this part it leaves the kernel at 37% of the sustained
DRAM bandwidth. At 1024 each lane moves eight bf16 (one 16-byte access, the widest the ISA has) and
the kernel reaches 82%.

The effect is much larger than a normal tiling win because of the caller. The linear-attention
layer builds the conv input as a view:

    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)

so the tensor handed to the conv has a row pitch of qkv_dim + z_dim, not qkv_dim. On this model at
TP2 that is 8192 bf16 = 16384 bytes -- exactly 2**14 -- and a narrow block makes every program walk
its tokens at that stride, which aliases onto one DRAM channel set. Measured in isolation at the
serve's own chunk length (3296 tokens, 5120 channels per rank): 304 us at BLOCK_N=256 against 138 us
at 1024, a 2.22x. The same sweep with the pitch padded by 128 elements shows where the penalty comes
from: 0.196 ms at pitch 8192 versus 0.094 ms at 8320. A wide block spreads each program's accesses
across enough channels that the aliasing stops mattering, and it is still 1.21x on a pitch that does
not alias at all, so this is a win either way.

Bit-identical: the 4-tap accumulation is per channel, so retiling the channel axis cannot change a
single result. Verified exactly equal to the stock kernel at BLOCK_N 512/1024/2048.

The decode `causal_conv1d_update` kernel is deliberately left alone. Its grid is
(batch, ceil(dim/BLOCK_N)) -- 40 workgroups at 1024 -- and widening it measured *worse*
(8.5 -> 12.0 us at eight sequences).

The block falls back to 256 when the channel count is too small to fill two blocks, so a model with
a narrow conv dim keeps a usable grid.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

CONV = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
)

# `BLOCK_N=256,` appears twice (prefill fwd and decode update); this triple is unique to the
# prefill launch, which is the only one being changed.
ANCHOR = "        BLOCK_M=BLOCK_M,\n        BLOCK_N=256,\n        num_stages=2,\n"

NEW = (
    "        BLOCK_M=BLOCK_M,\n"
    "        BLOCK_N=(\n"
    "            _RADIANCE_CONV1D_BLOCKN\n"
    "            if dim >= 2 * _RADIANCE_CONV1D_BLOCKN\n"
    "            else 256\n"
    "        ),\n"
    "        num_stages=2,\n"
)

# Injected next to the module's own imports so the width is a module constant rather than a value
# recomputed per launch.
IMPORT_ANCHOR = (
    "from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID\n"
)
IMPORT_NEW = (
    IMPORT_ANCHOR
    + "\n"
    "# RADIANCE: prefill conv1d channel-block width. 256 (the stock value) gives each lane a\n"
    "# 4-byte access and lands at 37% of DRAM bandwidth; 1024 gives a 16-byte access and 82%,\n"
    "# and it also defuses the 2**14-byte row pitch the qkvz split() view hands this kernel.\n"
    "# Bit-identical either way -- the 4-tap accumulation is per channel.\n"
    "_RADIANCE_CONV1D_BLOCKN = 1024\n"
)


def main():
    apply(CONV, IMPORT_ANCHOR, IMPORT_NEW, "_RADIANCE_CONV1D_BLOCKN",
          "conv1d BLOCK_N constant")
    apply(CONV, ANCHOR, NEW, "_RADIANCE_CONV1D_BLOCKN\n            if dim",
          "conv1d BLOCK_N 256 -> 1024 (prefill)")


if __name__ == "__main__":
    main()
