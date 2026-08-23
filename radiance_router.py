"""RADIANCE MoE-router GEMM dispatch. Wraps the R4D gfx1201 kernel gemm_bf16_nt_m16 for the
bf16 gate GEMM C[n,256] = x[n,2048] @ W[256,2048]^T. Enabled with the rest of the library by RADIANCE_USE_R4D. Called from the
patched rocm_unquantized_gemm_impl for n in [6,16], the band rocBLAS serves poorly (wvSplitK covers n<=5)."""
import os
import sys
import torch

# RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py): with
# it off this module behaves exactly as it would in an image built without the library.
USE_R4D = os.environ.get("RADIANCE_USE_R4D", "1") == "1"
ENABLED = USE_R4D
_WV, _SK = 8, 4                            # columns per block, k-splits (best config)

try:
    import r4d as _rg                      # the compiled R4D kernel library
except Exception as e:                     # library missing or failed to load: stay on rocBLAS
    _rg = None
    ENABLED = False
    sys.stderr.write(f"[radiance.router] r4d import failed, disabled: {e!r}\n")

# The entry point carries the shape it is compiled for, so ask the library which kernel it has for
# this GEMM rather than naming one here. M is the top of the band this dispatch serves and K the
# router's hidden size; both are constraints the kernel rejects on, and None means stay on rocBLAS.
_GEMM = None
if ENABLED:
    _name = _rg.select("gemm_nt", M=16, K=2048, dtype="bf16")
    if _name is None:
        ENABLED = False
        sys.stderr.write("[radiance.router] no gemm_nt kernel for M<=16 K=2048 bf16, disabled\n")
    else:
        _GEMM = getattr(_rg, _name)
        sys.stderr.write(
            "[radiance.router] router GEMM kernel ENABLED (n in [6,16], WV=8 SK=4, wave32)\n")


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, K = int(x.shape[0]), int(x.shape[1])
    N = int(weight.shape[0])
    c = torch.empty((n, N), device=x.device, dtype=torch.bfloat16)
    _GEMM(x.data_ptr(), weight.data_ptr(), c.data_ptr(), n, K, N, _WV, _SK,
          torch.cuda.current_stream().cuda_stream)
    return c
