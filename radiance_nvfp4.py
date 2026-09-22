"""Bounded load-time NVFP4 to MXFP4 compatibility; this is requantization.

E2M1 quantizer adapted from GGZ14/vllm-mxfp4, commit
20652ec (frozen donor 31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663).
Radiance adds NVFP4-only policy, row-bounded dequantization, validation,
transactional installation and checkpoint/policy/layout provenance.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import Parameter

DONOR = "31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663"
LAYOUT = "radiance-mxfp4-e2m1-low-nibble-e8m0-group32-v1"
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def policy():
    """Only NVFP4 conversion is currently supported; unrelated precision stays put."""
    enabled = os.environ.get("RADIANCE_NVFP4_MXFP4", "0")
    if enabled not in ("0", "1"):
        raise ValueError("RADIANCE_NVFP4_MXFP4 must be 0 or 1")
    for name, default in (("RADIANCE_NVFP4_FP8_LAYERS", "fp8"),
                          ("RADIANCE_NVFP4_BF16_LAYERS", ""),
                          ("RADIANCE_NVFP4_LMHEAD", "preserve")):
        if os.environ.get(name, default) != default:
            raise ValueError(f"{name}: extra precision conversion is not qualified")
    mode = os.environ.get("RADIANCE_NVFP4_EXP", "mse")
    rows = int(os.environ.get("RADIANCE_NVFP4_CHUNK_ROWS", "128"))
    if mode not in ("mse", "ocp", "noclip") or not 1 <= rows <= 4096:
        raise ValueError("invalid NVFP4 exponent mode or chunk rows (1..4096)")
    return {"enabled": enabled == "1", "exponent": mode, "chunk_rows": rows,
            "extra_fp8": False, "extra_bf16": False, "lm_head": "preserve",
            "execution_layout": LAYOUT}


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 -> [N, K] fp32. Low nibble is element 2i (vLLM's break_fp4_bytes order)."""
    n = packed.shape[0]
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(n, -1)
    grid = _E2M1.to(packed.device)
    mag = grid[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def dequant_nvfp4(packed: torch.Tensor, scale_e4m3: torch.Tensor, global_divisor: float) -> torch.Tensor:
    """W[n, k] = e2m1 * fp32(scale_e4m3[n, k//16]) / global_divisor.

    global_divisor is the checkpoint's weight_global_scale AS STORED (compressed-tensors stores
    (6 * 448) / amax, a divisor; vLLM's own scheme inverts it the same way)."""
    n = packed.shape[0]
    k = packed.shape[1] * 2
    v = unpack_e2m1(packed).reshape(n, k // 16, 16)
    s = scale_e4m3.to(torch.float32) / float(global_divisor)
    return (v * s.unsqueeze(-1)).reshape(n, k)


# --------------------------------------------------------------------------------------------
# fp32 -> MXFP4 (e2m1 + e8m0 per 32)
# --------------------------------------------------------------------------------------------
def e2m1_index(a: torch.Tensor) -> torch.Tensor:
    """|value| (already divided by the block scale, clamped to 6) -> code index 0..7 with
    round-to-nearest, ties to the EVEN code, matching the argmin+tie rule of the offline
    quantizer (quantize_dflash_mxfp4.py). torch.round is half-to-even on the integers it sees:
      a < 2      : step 0.5 -> round(2a)        (indices 0..4)
      2 <= a < 4 : step 1   -> round(a) + 2     (indices 4..6)
      a >= 4     : step 2   -> round(a/2) + 4   (indices 6..7)"""
    a = a.clamp(max=6.0)
    lo = torch.round(a * 2.0)
    mid = torch.round(a) + 2.0
    hi = torch.round(a * 0.5) + 4.0
    idx = torch.where(a < 2.0, lo, torch.where(a < 4.0, mid, hi))
    return idx.to(torch.uint8)


def _block_exponent(amax: torch.Tensor, mode: str) -> torch.Tensor:
    """Unbiased block exponent e (scale = 2^e) for each 32-block from its amax. amax == 0 -> 0."""
    safe = amax.clamp(min=torch.finfo(torch.float32).tiny)
    if mode == "ocp":
        e = torch.floor(torch.log2(safe)) - 2.0          # amax lands in [4, 8): values above 6 clip
    else:
        e = torch.ceil(torch.log2(safe / 6.0))           # amax lands in (3, 6]: never clips
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    return e.clamp(-127.0, 127.0)


def _encode_blocks(wb: torch.Tensor, e: torch.Tensor):
    """wb [R, G, 32] fp32, e [R, G] exponent -> (idx uint8 [R, G, 32], sq err [R, G])."""
    scale = torch.exp2(e).unsqueeze(-1)
    v = wb / scale
    idx = e2m1_index(v.abs())
    grid = _E2M1.to(wb.device)
    q = grid[idx.long()] * torch.sign(v) * scale
    err = ((q - wb) ** 2).sum(-1)
    return idx, err


def quant_mxfp4(w: torch.Tensor, mode: str = "mse"):
    """[N, K] fp32 -> (packed [N, K/2] uint8 low-nibble-first, e8m0 [N, K/32] uint8, sq err)."""
    n, k = w.shape
    assert k % 32 == 0, k
    wb = w.float().reshape(n, k // 32, 32)
    amax = wb.abs().amax(-1)
    if mode == "mse":
        e0 = _block_exponent(amax, "noclip")
        i0, r0 = _encode_blocks(wb, e0)
        e1 = (e0 - 1.0).clamp(min=-127.0)                                   # finer grid, top of block clips at 6
        i1, r1 = _encode_blocks(wb, e1)
        take1 = r1 < r0
        e = torch.where(take1, e1, e0)
        idx = torch.where(take1.unsqueeze(-1), i1, i0)
        err = torch.where(take1, r1, r0)
    else:
        e = _block_exponent(amax, mode)
        idx, err = _encode_blocks(wb, e)
    sign = (torch.signbit(wb) & (idx != 0)).to(torch.uint8) << 3   # -0 stays code 0
    code = (idx | sign).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    e8m0 = (e + 127.0).to(torch.uint8).contiguous()
    return packed, e8m0, err.sum()


def dequant_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    n = packed.shape[0]
    k = packed.shape[1] * 2
    v = unpack_e2m1(packed).reshape(n, k // 32, 32)
    return (v * torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)).reshape(n, k)



def _hash_tensor(hasher, value):
    hasher.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())


@torch.no_grad()
def convert(packed, scales, divisors, widths, *, chunk_rows=128, mode="mse",
            source_id, metadata):
    """Convert each partition without materializing it in FP32.

    Input and output are complete packed tensors on their original device.
    All arithmetic temporaries are CPU row chunks. Full outputs are allocated
    once; no list/cat duplication of the converted model is used.
    """
    if not source_id or not isinstance(metadata, dict):
        raise ValueError("checkpoint identity and original quantization metadata required")
    if mode not in ("mse", "ocp", "noclip") or not 1 <= chunk_rows <= 4096:
        raise ValueError("unsupported conversion mode/chunk bound")
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("NVFP4 requires a two-dimensional uint8 packed weight")
    n, half_k = packed.shape
    k = half_k * 2
    if n <= 0 or k <= 0 or k % 32:
        raise ValueError("MXFP4 requires nonempty rows and K divisible by 32")
    if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != (n, k // 16):
        raise ValueError("NVFP4 requires e4m3 scales with group size 16")
    if packed.device != scales.device:
        raise ValueError("packed weight and scales must share a device")
    if not widths or any(type(w) is not int or w <= 0 for w in widths) or sum(widths) != n:
        raise ValueError("merged partition widths do not cover the weight exactly")
    gs = divisors.detach().float().cpu().flatten().tolist()
    if len(gs) != len(widths) or not all(math.isfinite(g) and g > 0 for g in gs):
        raise ValueError("each partition requires one finite positive global divisor")
    # Conservative 96 bytes per scalar for overlapping FP32/index temporaries.
    # This bounds the selected algorithm's chunks; measured RSS is reported
    # separately because the CPU allocator may retain its arenas.
    rows = min(chunk_rows, max(1, (32 * 1024 * 1024) // (96 * k)))
    if 96 * k > 32 * 1024 * 1024:
        raise ValueError("one conversion row exceeds the temporary budget")
    out = torch.empty_like(packed)
    out_scales = torch.empty((n, k // 32), dtype=torch.uint8, device=packed.device)
    source_hash, result_hash = hashlib.sha256(), hashlib.sha256()
    source_p, source_s = hashlib.sha256(), hashlib.sha256()
    result_p, result_s = hashlib.sha256(), hashlib.sha256()
    source_hash.update(json.dumps({"source": source_id, "metadata": metadata,
                                  "widths": widths, "divisors": gs},
                                 sort_keys=True).encode())
    err, ssq, offset = 0.0, 0.0, 0
    max_chunk = 0
    for width, divisor in zip(widths, gs):
        for start in range(offset, offset + width, rows):
            stop = min(offset + width, start + rows)
            pc = packed[start:stop].detach().cpu()
            sc = scales[start:stop].detach().cpu()
            if not torch.isfinite(sc.float()).all() or (sc.float() < 0).any():
                raise ValueError("NVFP4 group scales must be finite and nonnegative")
            wc = dequant_nvfp4(pc, sc, divisor)
            if not torch.isfinite(wc).all():
                raise ValueError("NVFP4 dequantization overflow")
            p, s, error = quant_mxfp4(wc, mode)
            out[start:stop].copy_(p)
            out_scales[start:stop].copy_(s)
            _hash_tensor(source_p, pc)
            _hash_tensor(source_s, sc)
            _hash_tensor(result_p, p)
            _hash_tensor(result_s, s)
            err += float(error.double())
            ssq += float(wc.double().square().sum())
            max_chunk = max(max_chunk, wc.numel() * 4)
            del pc, sc, wc, p, s, error
        offset += width
    source_hash.update(source_p.digest() + source_s.digest())
    result_hash.update(result_p.digest() + result_s.digest())
    receipt = {"source_id": source_id, "source_quantization": metadata,
               "source_sha256": source_hash.hexdigest(),
               "converted_sha256": result_hash.hexdigest(), "widths": widths,
               "global_divisors": gs, "mode": mode, "chunk_rows": rows,
               "peak_fp32_dequant_tensor_bytes": max_chunk,
               "temporary_budget_bytes": 32 * 1024 * 1024,
               "output_bytes": out.numel() + out_scales.numel(),
               "device_arithmetic_temporary_bytes": 0,
               "relative_rms_error": math.sqrt(err / ssq) if ssq else 0.0,
               "layout": LAYOUT, "donor": DONOR,
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return out, out_scales, receipt


def scheme_class():
    """Use v0.30's original parameter/loader ABI, with an explicit MXFP4 backend."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
    from radiance_mxfp4 import kernel_class

    class RadianceNvfp4ToMxfp4(CompressedTensorsW4A4Fp4):
        def __init__(self, *, use_a16, source_id, metadata):
            if use_a16:
                raise ValueError("NVFP4A16 is not qualified for Radiance W4A8 conversion")
            from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
            selected = kernel_class()
            if selected is None:
                raise RuntimeError("NVFP4 conversion requires Radiance native W4A8")
            config = MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
            supported, reason = selected.is_supported()
            if not supported:
                raise RuntimeError(f"NVFP4 W4A8 backend unavailable: {reason}")
            implement, reason = selected.can_implement(config)
            if not implement:
                raise RuntimeError(f"NVFP4 W4A8 layout unsupported: {reason}")
            self.kernel = selected(config)
            self.use_a16 = use_a16
            self.group_size = 16
            self.source_id = source_id
            self.source_metadata = metadata

        @torch.no_grad()
        def process_weights_after_loading(self, layer):
            cfg = policy()
            packed, scales, receipt = convert(
                layer.weight_packed, layer.weight_scale, layer.weight_global_scale,
                list(layer.logical_widths), chunk_rows=cfg["chunk_rows"],
                mode=cfg["exponent"], source_id=self.source_id,
                metadata=self.source_metadata)
            # Processing occurs on a temporary module/parameter dictionary.
            # If native repacking fails, the source module remains intact.
            staged = copy.copy(layer)
            staged._parameters = dict(layer._parameters)
            staged.weight = Parameter(packed, requires_grad=False)
            staged.weight_scale = Parameter(scales, requires_grad=False)
            del staged.weight_packed, staged.weight_global_scale
            if hasattr(staged, "input_global_scale"):
                del staged.input_global_scale
            self.kernel.process_weights_after_loading(staged)
            if not getattr(staged, "radiance_w4a8_ok", False):
                raise RuntimeError("NVFP4 converted shape did not select native W4A8")
            receipt["policy"] = cfg
            staged._radiance_nvfp4_receipt = receipt
            layer.__dict__.update(staged.__dict__)
            print("[radiance.nvfp4] " + json.dumps(receipt, sort_keys=True), flush=True)

    return RadianceNvfp4ToMxfp4


def select_scheme(weight_quant, input_quant, layer_name):
    cfg = policy()
    if not cfg["enabled"]:
        return None
    if not layer_name:
        raise ValueError("NVFP4 conversion requires an explicit layer identity")
    if layer_name.split(".")[-1] == "lm_head":
        return None
    if input_quant is None:
        raise ValueError("NVFP4A16 is not qualified for Radiance W4A8 conversion")
    source_id = os.environ.get("RADIANCE_NVFP4_SOURCE_ID", "").strip()
    if not source_id:
        raise ValueError("RADIANCE_NVFP4_SOURCE_ID must identify the checkpoint revision")
    metadata = {"weight": weight_quant.model_dump(mode="json"),
                "input": input_quant.model_dump(mode="json") if input_quant else None,
                "layer": layer_name}
    return scheme_class()(use_a16=False, source_id=source_id, metadata=metadata)
