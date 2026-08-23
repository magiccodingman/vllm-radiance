#!/usr/bin/env python3
"""Install the RADIANCE router-GEMM hook: route the bf16 MoE-gate GEMM (x[n,2048] @ W[256,2048]^T,
n in [6,16]) to the custom gfx1201 kernel via radiance_router, gated by RADIANCE_MOE_ROUTER=1.
Idempotent source patch of vllm's rocm_unquantized_gemm_impl (the rocm unquantized-linear chokepoint)."""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
UTIL = SP / "vllm/model_executor/layers/utils.py"

ANCHOR = (
    "def rocm_unquantized_gemm_impl(\n"
    "    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None\n"
    ") -> torch.Tensor:\n"
    "    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250\n"
)
NEW = (
    "try:\n"
    "    import radiance_router as _radiance_router\n"
    "except Exception:\n"
    "    _radiance_router = None\n"
    "\n"
    "def rocm_unquantized_gemm_impl(\n"
    "    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None\n"
    ") -> torch.Tensor:\n"
    "    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250\n"
    "    # --- RADIANCE router GEMM kernel (patch_router_gemm.py) ---\n"
    "    if (_radiance_router is not None and _radiance_router.ENABLED and bias is None\n"
    "            and weight.dim() == 2 and weight.shape[0] == 256 and weight.shape[1] == 2048\n"
    "            and x.shape[-1] == 2048 and x.dtype == torch.bfloat16):\n"
    "        _n = x.numel() // 2048\n"
    "        if 6 <= _n <= 16 and x.is_contiguous() and weight.is_contiguous():\n"
    "            return _radiance_router.router_gemm(x.reshape(_n, 2048), weight)\n"
)


def main():
    apply(UTIL, ANCHOR, NEW, "RADIANCE router GEMM kernel (patch_router_gemm.py)",
          "route bf16 MoE-gate GEMM n in [6,16] -> custom gfx1201 kernel")


if __name__ == "__main__":
    main()
