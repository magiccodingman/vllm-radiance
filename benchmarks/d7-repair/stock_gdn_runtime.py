"""Explicit, source-bound installation of the experimental D7 arithmetic repairs.

Used by isolated qualification workers. It does not change installed sources,
production profiles or snapshot compatibility on import.
"""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, private_json, seal

RADIANCE_SOURCE = "e4b4a735804c490f52bee2a11735a7a3b29a9c5209bdc0fa34f452f96595aafb"
CONV_SOURCE = "da2d1183c29f68497d0166960c3ca7bedd143b90eead1de3d98e90af0c0f8a4a"
REQUIRED_FILES = (
    "stock_gdn_runtime.py",
    "stock_gdn_scan.py",
    "stock_gdn_scan_kernel.py",
    "stock_gdn_convolution_adapter.py",
    "stock_gdn_indexed_adapter.py",
    "stock_gdn_prefill_adapter.py",
    "stock_gdn_sequence.py",
    "stock_target_arithmetic.py",
    "stock_m1_norm.py",
    "stock_m1_gdn_norm.py",
    "stock_m1_attention.py",
)


def validate_separate_gdn_dispatch(native):
    # The pinned model has 48 value heads. FUSED_MAX_ITEMS=32 disables the
    # fused branch for every legal nonempty batch even when its feature flag
    # is enabled. Match the native dispatch predicate, not just that flag.
    fused_reachable = native.FUSED_UPDATE_ON and native.FUSED_MAX_ITEMS >= 48
    if fused_reachable or native.NORM_FUSE:
        raise DiagnosticError("D7 qualification requires separate convolution, recurrence and norm")


def validate_manifest(path):
    manifest = private_json(path)
    authenticate(manifest)
    if (
        manifest.get("schema") != "urn:qwen:d7-stock-repair-bundle:v4"
        or type(manifest.get("prefill")) is not bool
        or type(manifest.get("stock_norm")) is not bool
        or type(manifest.get("stock_head")) is not bool
        or type(manifest.get("stock_attention")) is not bool
        or type(manifest.get("head_group_size")) is not int
        or manifest.get("head_group_size") not in (1, 2, 4, 5)
        or set(manifest.get("sources", {})) != set(REQUIRED_FILES)
    ):
        raise DiagnosticError("unsupported D7 repair bundle")
    root = Path(__file__).resolve().parent
    for name, expected in manifest["sources"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise DiagnosticError("D7 repair source differs from the qualified bundle")
    conv = Path(manifest["convolution"])
    if not conv.is_absolute() or hashlib.sha256(conv.read_bytes()).hexdigest() != CONV_SOURCE:
        raise DiagnosticError("D7 repair convolution source mismatch")
    if manifest.get("norm_build") is not None:
        build = Path(manifest["norm_build"])
        record = private_json(build / "build.json")
        authenticate(record)
        if (
            not build.is_absolute()
            or not manifest["stock_norm"]
            or record["sha256"] != manifest.get("norm_build_sha256")
            or hashlib.sha256((build / "candidate.so").read_bytes()).hexdigest()
            != record["binary_sha256"]
        ):
            raise DiagnosticError("D7 normalization binary differs from the repair bundle")
    return manifest


class RuntimeRepairs:
    def __init__(self, manifest_path, *, target_model=None):
        manifest = validate_manifest(Path(manifest_path))
        import torch  # isort: skip

        import radiance_gdn as native
        from stock_gdn_convolution_adapter import StockConvolutionAdapter
        from stock_gdn_indexed_adapter import StockIndexedAdapter
        from stock_gdn_prefill_adapter import StockPrefillAdapter
        from stock_gdn_scan import StockScan

        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != RADIANCE_SOURCE:
            raise DiagnosticError("D7 repair requires the pinned Radiance source")
        validate_separate_gdn_dispatch(native)
        if torch.cuda.is_current_stream_capturing():
            raise DiagnosticError("D7 repairs must be installed before execution")
        model = importlib.import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        spec = importlib.util.spec_from_file_location(
            "qwen_stock_runtime_convolution", manifest["convolution"]
        )
        conv = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = conv
        spec.loader.exec_module(conv)
        self.manifest = manifest
        self.hooks = HookSet()
        self.convolution = StockConvolutionAdapter(torch, conv.causal_conv1d_update)
        self.recurrent = StockIndexedAdapter(torch)
        self.prefill = None
        self.target = None
        self.attention = None
        try:
            self.hooks.replace(model, "causal_conv1d_update", conv.causal_conv1d_update)
            self.hooks.replace(native, "conv_update", self.convolution.__call__)
            self.hooks.replace(native, "recurrent_update", self.recurrent.__call__)
            if manifest["stock_attention"]:
                from stock_m1_attention import StockM1Attention

                self.attention = StockM1Attention(self.hooks)
            if manifest["prefill"]:
                self.prefill = StockPrefillAdapter(
                    torch,
                    native,
                    conv.causal_conv1d_update,
                    StockScan(torch, torch.device("cuda", torch.cuda.current_device())),
                    native.forward_core_fused,
                )
                self.hooks.replace(native, "forward_core_fused", self.prefill.__call__)
            if manifest["stock_norm"] or manifest["stock_head"]:
                from stock_target_arithmetic import TargetArithmetic

                if target_model is None:
                    raise DiagnosticError("target arithmetic repair requires the loaded target")
                self.target = TargetArithmetic(
                    target_model,
                    self.hooks,
                    norm=manifest["stock_norm"],
                    head=manifest["stock_head"],
                    norm_build=manifest.get("norm_build"),
                    head_group_size=manifest["head_group_size"],
                )
        except BaseException:
            self.close()
            raise

    def receipt(self):
        return seal(
            {
                "bundle": self.manifest["sha256"],
                "convolution_calls": self.convolution.calls,
                "recurrent_calls": self.recurrent.calls,
                "recurrent_rows": self.recurrent.rows,
                "prefill_calls": self.prefill.calls if self.prefill else 0,
                "prefill_rows": self.prefill.rows if self.prefill else 0,
                "target_arithmetic": self.target.receipt() if self.target else None,
                "target_attention": self.attention.receipt() if self.attention else None,
                "installed_sources_changed": False,
            }
        )

    def close(self):
        self.hooks.close()
