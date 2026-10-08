"""Pinned gated GDN norm with the same one-row workgroups as native M1.

The native kernel changes its row tile with the total number of heads. Keep
the M1 tile while still launching all speculative rows together.
"""

import hashlib
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError

NORM_SOURCE = "0ca5e46aab408b80d673561b91cf7bff7987c71d798d3bdacac75b7e29416f49"


class StockM1GdnNorm:
    def __init__(self):
        from vllm.third_party.flash_linear_attention.ops import layernorm_guard as native

        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NORM_SOURCE:
            raise DiagnosticError("GDN norm requires the pinned native kernel")
        self.native = native

    def __call__(self, x, z, weight, eps):
        import torch

        if (
            x.ndim != 2
            or x.shape[1] != 128
            or not 48 <= x.shape[0] <= 384
            or x.shape[0] % 48
            or x.dtype != torch.bfloat16
            or z.shape != x.shape
            or z.dtype != x.dtype
            or weight.shape != (128,)
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or x.device.type != "cuda"
            or z.device != x.device
            or weight.device != x.device
            or any(t.stride(-1) != 1 for t in (x, z, weight))
        ):
            raise DiagnosticError("GDN norm inputs differ from the admitted M1/D7 contract")
        if self.native.calc_rows_per_block(48, x.device) != 1:
            raise DiagnosticError("native GDN M1 row layout changed")
        out = torch.empty_like(x)
        rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        self.native.layer_norm_fwd_kernel[(x.shape[0], 1)](
            x,
            out,
            weight,
            None,
            z,
            None,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0),
            x.shape[0],
            128,
            eps,
            BLOCK_N=128,
            ROWS_PER_BLOCK=1,
            HAS_BIAS=False,
            HAS_Z=True,
            NORM_BEFORE_GATE=True,
            IS_RMS_NORM=True,
            ACTIVATION="silu",
            num_warps=1,
        )
        return out
