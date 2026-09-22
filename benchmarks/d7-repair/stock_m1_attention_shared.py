"""Isolated shared-KV candidate with per-query M1 softmax decisions."""

import ctypes as c
import hashlib
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json


class R4DArgs(c.Structure):
    _fields_ = (
        [
            (n, c.c_void_p)
            for n in ["q", "kv", "table", "lengths", "out", "ks", "vs", "qs", "scratch"]
        ]
        + [
            (n, c.c_int)
            for n in ["sequences", "width", "heads", "kv_heads", "dim", "block", "max_blocks"]
        ]
        + [
            ("block_stride", c.c_long),
            ("head_stride", c.c_long),
            ("scale", c.c_float),
            ("splits", c.c_int),
            ("max_ctx", c.c_int),
        ]
    )


class SharedM1Attention:
    def __init__(self, build):
        build = Path(build)
        self.manifest = private_json(build / "build.json")
        authenticate(self.manifest)
        binary = build / "candidate.so"
        require(
            self.manifest["status"] == "BUILT_UNTESTED"
            and self.manifest.get("kernel_abi") == "qwen-stock-m1-shared-attention-v1"
            and hashlib.sha256(binary.read_bytes()).hexdigest() == self.manifest["binary_sha256"],
            "shared attention binary binding mismatch",
        )
        require(c.sizeof(R4DArgs) == 136, "R4D host ABI size changed")
        self.library = c.CDLL(str(binary))
        self.launch = self.library.qwen_stock_m1_attention_shared
        self.launch.argtypes = [c.POINTER(R4DArgs), c.c_int, c.c_void_p]
        self.launch.restype = c.c_int

    def __call__(
        self, query, kv, table, lengths, scratch, *, out=None, ks=None, vs=None, max_ctx=253792
    ):
        import torch

        tensors = [query, kv, table, lengths, scratch]
        valid = (
            query.shape == (8, 24, 256)
            and query.dtype == torch.bfloat16
            and query.device.type == "cuda"
            and query.is_contiguous()
            and kv.shape[1:] == (4, 16, 512)
            and kv.dtype in (torch.float8_e4m3fn, torch.uint8, torch.bfloat16)
            and kv.stride(3) == 1
            and kv.stride(2) == 512
            and kv.stride(1) >= 16 * 512
            and kv.stride(0) >= 4 * kv.stride(1)
            and table.ndim == 2
            and table.shape[0] == 1
            and table.dtype == lengths.dtype == torch.int32
            and lengths.shape == (1,)
            and scratch.dtype == torch.uint8
            and scratch.is_contiguous()
            and scratch.numel() >= 8 * 24 * 32 * 520
            and max_ctx >= 1024
            and all(t.device == query.device for t in tensors)
        )
        if not valid:
            require(
                False,
                "attention input contract changed: "
                + repr([(tuple(t.shape), str(t.dtype), t.stride()) for t in tensors]),
            )
        for scales in (ks, vs):
            require(
                scales is None
                or (
                    scales.shape == (4,)
                    and scales.dtype == torch.float32
                    and scales.device == query.device
                    and scales.is_contiguous()
                ),
                "attention scale contract changed",
            )
        if out is None:
            out = torch.empty_like(query)
        require(
            out.shape == query.shape
            and out.dtype == query.dtype
            and out.device == query.device
            and out.is_contiguous(),
            "attention output contract changed",
        )
        require(
            out.untyped_storage().data_ptr()
            not in [t.untyped_storage().data_ptr() for t in tensors],
            "attention output aliases an input",
        )
        args = R4DArgs(
            query.data_ptr(),
            kv.data_ptr(),
            table.data_ptr(),
            lengths.data_ptr(),
            out.data_ptr(),
            0 if ks is None else ks.data_ptr(),
            0 if vs is None else vs.data_ptr(),
            0,
            scratch.data_ptr(),
            1,
            8,
            24,
            4,
            256,
            16,
            table.shape[1],
            kv.stride(0),
            kv.stride(1),
            256**-0.5,
            32,
            max_ctx,
        )
        with torch.cuda.device(query.device):
            status = self.launch(
                c.byref(args),
                int(kv.dtype == torch.bfloat16),
                torch.cuda.current_stream().cuda_stream,
            )
        require(status == 0, f"shared attention launch failed: {status}")
        return out
