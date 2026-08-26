#!/usr/bin/env python3
"""Install the RADIANCE skinny-GEMM hook on vLLM's ROCm unquantized-linear chokepoint.

Any bf16 projection whose weight is small enough that rocBLAS lays it out as a handful of
workgroups goes to the R4D split-K kernel instead. radiance_gemm decides -- it holds the measured
(N, K) -> config table and declines everything else -- so adding a shape is an edit to that module,
not a re-patch. Gated by RADIANCE_USE_R4D through the same module.

The band starts at M = 6 because vLLM's own wvSplitK serves M <= 5, and it ends at 16 because that
is where the kernel holds its accumulators.
"""
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
    "    import radiance_gemm as _radiance_gemm\n"
    "except Exception:\n"
    "    _radiance_gemm = None\n"
    "\n"
    "def rocm_unquantized_gemm_impl(\n"
    "    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None\n"
    ") -> torch.Tensor:\n"
    "    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250\n"
    "    # --- RADIANCE skinny GEMM kernel (patch_skinny_gemm.py) ---\n"
    "    if (_radiance_gemm is not None and _radiance_gemm.ENABLED and bias is None\n"
    "            and weight.dim() == 2 and x.dtype == torch.bfloat16\n"
    "            and weight.dtype == torch.bfloat16):\n"
    "        _out = _radiance_gemm.maybe_gemm(x, weight)\n"
    "        if _out is not None:\n"
    "            return _out\n"
)


def main():
    apply(UTIL, ANCHOR, NEW, "RADIANCE skinny GEMM kernel (patch_skinny_gemm.py)",
          "route measured skinny bf16 GEMMs -> custom gfx1201 kernel")


if __name__ == "__main__":
    main()
