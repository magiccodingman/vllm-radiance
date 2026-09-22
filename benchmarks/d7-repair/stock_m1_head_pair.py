"""Strict tensor and binary binding for the experimental M4-pair head."""

import ctypes
import hashlib
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json


class StockM1HeadPair:
    def __init__(self, build):
        build = Path(build)
        self.manifest = private_json(build / "build.json")
        authenticate(self.manifest)
        binary = build / "candidate.so"
        require(
            self.manifest["status"] == "BUILT_UNTESTED"
            and self.manifest.get("kernel_abi") == "qwen-stock-m1-head-pair-v1"
            and hashlib.sha256(binary.read_bytes()).hexdigest() == self.manifest["binary_sha256"],
            "head candidate binary binding mismatch",
        )
        self.library = ctypes.CDLL(str(binary))
        self.launch = self.library.qwen_stock_m1_head_pair
        self.launch.argtypes = [ctypes.c_void_p] * 4
        self.launch.restype = ctypes.c_int

    def __call__(self, weight, hidden, output=None):
        import torch

        require(
            tuple(weight.shape) == (248320, 5120)
            and tuple(hidden.shape) == (8, 5120)
            and weight.dtype == hidden.dtype == torch.bfloat16
            and weight.device == hidden.device
            and hidden.device.type == "cuda"
            and weight.is_contiguous()
            and hidden.is_contiguous(),
            "head input is outside the admitted contract",
        )
        if output is None:
            output = torch.empty((8, 248320), device=hidden.device, dtype=hidden.dtype)
        require(
            tuple(output.shape) == (8, 248320)
            and output.dtype == hidden.dtype
            and output.device == hidden.device
            and output.is_contiguous(),
            "head output is outside the admitted contract",
        )
        require(
            output.untyped_storage().data_ptr()
            not in (weight.untyped_storage().data_ptr(), hidden.untyped_storage().data_ptr()),
            "head output must not alias its inputs",
        )
        with torch.cuda.device(hidden.device):
            status = self.launch(
                weight.data_ptr(),
                hidden.data_ptr(),
                output.data_ptr(),
                torch.cuda.current_stream(hidden.device).cuda_stream,
            )
        require(status == 0, f"head candidate launch failed: {status}")
        return output
