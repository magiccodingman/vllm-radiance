"""RADIANCE MoE-router GEMM dispatch. Wraps the custom gfx1201 HIP kernel (router_gemm ext) for the
bf16 gate GEMM C[n,256] = x[n,2048] @ W[256,2048]^T. Enabled by RADIANCE_MOE_ROUTER=1. Called from the
patched rocm_unquantized_gemm_impl for n in [6,16], the band rocBLAS serves poorly (wvSplitK covers n<=5)."""
import os
import sys
import torch

ENABLED = os.environ.get("RADIANCE_MOE_ROUTER", "0") == "1"
_WV, _SK = 8, 4                            # columns per block, k-splits (best config)

try:
    import router_gemm as _rg              # the compiled pybind11 HIP extension
except Exception as e:                     # ext missing or failed to load: stay on rocBLAS
    _rg = None
    ENABLED = False
    sys.stderr.write(f"[radiance.router] ext import failed, disabled: {e!r}\n")

if ENABLED and _rg is not None:
    sys.stderr.write("[radiance.router] router GEMM kernel ENABLED (n in [6,16], WV=8 SK=4, wave32)\n")


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, K = int(x.shape[0]), int(x.shape[1])
    N = int(weight.shape[0])
    c = torch.empty((n, N), device=x.device, dtype=torch.bfloat16)
    _rg.launch(x.data_ptr(), weight.data_ptr(), c.data_ptr(), n, K, N, _WV, _SK,
               torch.cuda.current_stream().cuda_stream)
    return c
