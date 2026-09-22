"""Source/binary-bound wrapper for the experimental batch-invariant norm."""

import ctypes
import hashlib
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, private_json


class StockM1Norm:
    def __init__(self, build):
        build = Path(build)
        self.manifest = private_json(build / "build.json")
        authenticate(self.manifest)
        binary = build / "candidate.so"
        if (
            self.manifest["status"] != "BUILT_UNTESTED"
            or self.manifest.get("kernel_abi") != "qwen-stock-m1-norm-v3"
            or hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]
        ):
            raise DiagnosticError("stock M1 normalization binary binding mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.launch = self.library.qwen_stock_m1_gemma_norm
        self.launch.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.launch.restype = ctypes.c_int

    def __call__(self, x, residual, weight, epsilon, *, moments=None):
        import torch

        hidden = x.ndim == 2 and x.shape[-1] == 5120
        attention = x.ndim == 3 and x.shape[1:] in ((4, 256), (24, 256))
        if not (hidden or attention):
            raise DiagnosticError("stock normalization shape is outside the pinned model")
        width = x.shape[-1]
        threads = 512 if hidden else (64 if x.shape[1] == 4 else 32)
        if (
            x.dtype != torch.bfloat16
            or x.stride(-1) != 1
            or x.device.type != "cuda"
            or weight.shape != (width,)
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or weight.stride(0) != 1
        ):
            raise DiagnosticError("stock M1 normalization requires the declared BF16 layout")
        if attention and residual is not None:
            raise DiagnosticError("attention normalization does not admit a residual")
        if residual is not None and (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or residual.device != x.device
            or residual.stride(-1) != 1
        ):
            raise DiagnosticError("stock M1 normalization residual representation changed")
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        flat = x.reshape(-1, width)
        if moments is not None and (
            moments.shape != (flat.shape[0], 2)
            or moments.dtype != torch.float32
            or moments.device != x.device
            or not moments.is_contiguous()
        ):
            raise DiagnosticError("stock normalization moment buffer has the wrong layout")
        residual_output = torch.empty_like(output) if residual is not None else None
        status = self.launch(
            flat.data_ptr(),
            residual.data_ptr() if residual is not None else None,
            weight.data_ptr(),
            output.data_ptr(),
            residual_output.data_ptr() if residual_output is not None else None,
            flat.shape[0],
            flat.stride(0),
            residual.stride(0) if residual is not None else 0,
            epsilon,
            torch.cuda.current_stream().cuda_stream,
            moments.data_ptr() if moments is not None else None,
            width,
            threads,
        )
        if status:
            raise DiagnosticError(f"stock M1 normalization launch failed: {status}")
        return (output, residual_output) if residual is not None else output
