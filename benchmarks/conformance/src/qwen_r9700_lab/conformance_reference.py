"""Independent CPU finite-precision operators for the declared reference.

FP32 reductions are explicitly left-to-right with separate rounded multiply and
add operations. NumPy scalar ufuncs/libm are trusted, versioned dependencies.
This is intentionally slow and does not call vLLM, Torch, HIP, R4D or BLAS GEMM.
"""

from __future__ import annotations

import numpy as np

from qwen_r9700_lab.conformance_state import values
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

LEVELS = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)
FP8_POSITIVE = values(bytes(range(127)), "fp8_e4m3fn").astype(np.float32)


def low_nibble(value):
    return value & 15


def high_nibble(value):
    return (value >> 4) & 15


def canonical_slot(block, offset, block_size):
    return block * block_size + offset


def bf16_round_bits(bits):
    """RNE bit expression for non-NaN FP32; also executed symbolically by Z3."""
    return (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000


def bf16(value):
    x = np.asarray(value, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = bf16_round_bits(bits)
    # Preserve NaN as NaN, including small NaN payloads that would round to Inf.
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    rounded = np.where(nan, bits | 0x00400000, rounded).astype(np.uint32)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_encode(value):
    """Saturating finite E4M3FN, round-to-nearest, ties to an even code."""
    x = np.asarray(value, dtype=np.float32)
    if not np.isfinite(x).all():
        raise DiagnosticError("reference FP8 quantizer rejects nonfinite inputs")
    magnitude = np.minimum(np.abs(x), np.float32(448))
    hi = np.minimum(np.searchsorted(FP8_POSITIVE, magnitude), 126)
    lo = np.maximum(hi - 1, 0)
    lower, upper = magnitude - FP8_POSITIVE[lo], FP8_POSITIVE[hi] - magnitude
    choose_hi = (upper < lower) | ((upper == lower) & ((hi & 1) == 0))
    code = np.where(choose_hi, hi, lo).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 7)


def fp8_decode(code):
    a = np.asarray(code, dtype=np.uint8)
    return values(a.tobytes(), "fp8_e4m3fn").astype(np.float32).reshape(a.shape)


def activation_quantize(value):
    x = np.asarray(value, dtype=np.float32)
    if not x.size or not np.isfinite(x).all():
        raise DiagnosticError("invalid activation quantizer input")
    scale = np.maximum(
        np.max(np.abs(x), axis=-1, keepdims=True) / np.float32(448), np.float32(1 / (448 * 512))
    )
    code = fp8_encode(np.divide(x, scale, dtype=np.float32))
    return code, scale


def unpack_mxfp4(packed, scales):
    packed, scales = np.asarray(packed), np.asarray(scales)
    if packed.dtype != np.uint8 or scales.dtype != np.uint8 or packed.ndim != 2:
        raise DiagnosticError("MXFP4 storage must be a packed matrix with E8M0 scales")
    n, half_k = packed.shape
    if half_k * 2 % 32 or scales.shape != (n, half_k * 2 // 32) or np.any(scales == 255):
        raise DiagnosticError("unsupported MXFP4 shape or nonfinite E8M0 scale")
    codes = np.stack((low_nibble(packed), high_nibble(packed)), axis=-1).reshape(n, -1)
    expanded = np.repeat(
        np.ldexp(np.ones_like(scales, dtype=np.float32), scales.astype(np.int32) - 127), 32, axis=-1
    )
    return np.multiply(LEVELS[codes], expanded, dtype=np.float32)


def ordered_sum(value, axis=-1):
    x = np.moveaxis(np.asarray(value, dtype=np.float32), axis, -1)
    result = np.zeros(x.shape[:-1], dtype=np.float32)
    for i in range(x.shape[-1]):
        result = np.add(result, x[..., i], dtype=np.float32)
    return result


def linear(value, weight, *, quantize_activation=False, output_bf16=True):
    x, w = np.asarray(value, dtype=np.float32), np.asarray(weight, dtype=np.float32)
    if x.shape[-1] != w.shape[-1] or w.ndim != 2:
        raise DiagnosticError("linear dimensions do not agree")
    if quantize_activation:
        code, scale = activation_quantize(x)
        x = np.multiply(fp8_decode(code), scale, dtype=np.float32)
    result = np.zeros((*x.shape[:-1], w.shape[0]), dtype=np.float32)
    for k in range(w.shape[-1]):
        product = np.multiply(x[..., k, None], w[:, k], dtype=np.float32)
        result = np.add(result, product, dtype=np.float32)
    return bf16(result) if output_bf16 else result


def rms_norm(value, weight, epsilon, *, weight_offset=0.0, output_bf16=True):
    x = np.asarray(value, dtype=np.float32)
    variance = ordered_sum(np.multiply(x, x, dtype=np.float32)) / np.float32(x.shape[-1])
    inverse = np.reciprocal(np.sqrt(variance + np.float32(epsilon), dtype=np.float32))
    normalized = np.multiply(x, inverse[..., None], dtype=np.float32)
    output = np.multiply(
        normalized,
        np.asarray(weight, dtype=np.float32) + np.float32(weight_offset),
        dtype=np.float32,
    )
    return bf16(output) if output_bf16 else output


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(-np.abs(x), dtype=np.float32)
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e)).astype(np.float32)


def silu(x):
    return np.multiply(np.asarray(x, dtype=np.float32), sigmoid(x), dtype=np.float32)


def softplus(x):
    x = np.asarray(x, dtype=np.float32)
    return np.add(
        np.maximum(x, np.float32(0)),
        np.log1p(np.exp(-np.abs(x), dtype=np.float32)),
        dtype=np.float32,
    )


def rope(value, position, rotary_dim, theta):
    x = np.asarray(value, dtype=np.float32).copy()
    if rotary_dim % 2 or rotary_dim > x.shape[-1] or position < 0:
        raise DiagnosticError("invalid text rotary position geometry")
    frequencies = np.float32(position) / np.power(
        np.float32(theta), np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    )
    cosine, sine = bf16(np.cos(frequencies)), bf16(np.sin(frequencies))
    half = rotary_dim // 2
    a, b = x[..., :half].copy(), x[..., half:rotary_dim].copy()
    x[..., :half] = np.subtract(np.multiply(a, cosine), np.multiply(b, sine), dtype=np.float32)
    x[..., half:rotary_dim] = np.add(np.multiply(b, cosine), np.multiply(a, sine), dtype=np.float32)
    return bf16(x)


def convolution_step(value, history, weight):
    x, old, w = (np.asarray(v, dtype=np.float32) for v in (value, history, weight))
    if w.ndim != 2 or old.shape != (w.shape[0], w.shape[1] - 1) or x.shape != (w.shape[0],):
        raise DiagnosticError("invalid causal convolution geometry")
    window = np.concatenate((old, x[:, None]), axis=1)
    output = bf16(silu(ordered_sum(np.multiply(window, w, dtype=np.float32))))
    return output, window[:, 1:].copy()


def gdn_step(q, k, v, decay_log, beta, state, *, scale=None):
    q, k, v, decay_log, beta, state = (
        np.asarray(x, dtype=np.float32) for x in (q, k, v, decay_log, beta, state)
    )
    heads, value_width, key_width = state.shape
    if (
        heads % q.shape[0]
        or q.shape != k.shape
        or q.shape[1] != key_width
        or v.shape != (heads, value_width)
    ):
        raise DiagnosticError("invalid GDN geometry")
    if decay_log.shape != (heads,) or beta.shape != (heads,):
        raise DiagnosticError("invalid GDN gates")
    q, k = (np.repeat(x, heads // x.shape[0], axis=0) for x in (q, k))
    decayed = np.multiply(
        state, np.exp(decay_log, dtype=np.float32)[:, None, None], dtype=np.float32
    )
    prediction = ordered_sum(np.multiply(decayed, k[:, None, :], dtype=np.float32))
    residual = np.subtract(v, prediction, dtype=np.float32)
    update = np.multiply(
        np.multiply(beta[:, None], residual, dtype=np.float32)[:, :, None],
        k[:, None, :],
        dtype=np.float32,
    )
    new_state = np.add(decayed, update, dtype=np.float32)
    output = ordered_sum(np.multiply(new_state, q[:, None, :], dtype=np.float32))
    output = np.multiply(
        output, np.float32(key_width**-0.5 if scale is None else scale), dtype=np.float32
    )
    return bf16(output), new_state


def dense_attention(q, keys, vals, *, scale=None):
    q, keys, vals = (np.asarray(v, dtype=np.float32) for v in (q, keys, vals))
    if q.ndim != 2 or keys.ndim != 3 or keys.shape != vals.shape:
        raise DiagnosticError("invalid attention geometry")
    heads, width = q.shape
    if width != keys.shape[-1] or heads % keys.shape[1] or keys.shape[0] == 0:
        raise DiagnosticError("invalid grouped attention geometry")
    mapping = np.arange(heads) // (heads // keys.shape[1])
    scores = ordered_sum(keys[:, mapping, :] * q[None, :, :])
    scores = np.multiply(scores, np.float32(width**-0.5 if scale is None else scale))
    probabilities = np.exp(scores - np.max(scores, axis=0), dtype=np.float32)
    probabilities /= ordered_sum(probabilities, axis=0)[None, :]
    output = ordered_sum(probabilities[:, :, None] * vals[:, mapping, :], axis=0)
    return bf16(output)


OPERATORS = {
    "rms_norm": rms_norm,
    "linear": linear,
    "rope": rope,
    "convolution_step": convolution_step,
    "gdn_step": gdn_step,
    "dense_attention": dense_attention,
    "activation_quantize": activation_quantize,
    "unpack_mxfp4": unpack_mxfp4,
}


REFERENCE_PROFILES = {
    "weight-only-bf16": (False, False),
    "weight-fp8-activations": (True, False),
    "weight-fp8-kv": (False, True),
    "radiance-fp8": (True, True),
}


def reference_precision(profile):
    if not isinstance(profile, str) or profile not in REFERENCE_PROFILES:
        raise DiagnosticError("unsupported reference precision profile")
    activation_fp8, kv_fp8 = REFERENCE_PROFILES[profile]
    return {
        "profile": profile,
        "activation_fp8": activation_fp8,
        "kv_fp8": kv_fp8,
        "kv_encoding": "fp8_e4m3fn" if kv_fp8 else "bf16",
        "additional_quantization": [
            name for name, enabled in (("activations", activation_fp8), ("KV", kv_fp8)) if enabled
        ],
    }


def kv_scaling(config, kv_scales, profile):
    precision = reference_precision(profile)
    expected = {str(i) for i, kind in enumerate(config["layer_types"]) if kind == "full_attention"}
    if not isinstance(kv_scales, dict):
        raise DiagnosticError("KV scales must be an explicit layer mapping")
    if not precision["kv_fp8"]:
        units = {i: [1.0, 1.0] for i in expected}
        if kv_scales and kv_scales != units:
            raise DiagnosticError("BF16 KV has no quantizer; scales must be absent or unit")
        return units
    if set(kv_scales) != expected or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(
            type(value) not in (int, float) or not np.isfinite(value) or value <= 0 for value in row
        )
        for row in kv_scales.values()
    ):
        raise DiagnosticError("all KV scale values must be declared explicitly")
    return kv_scales


def reference_contract(profile="radiance-fp8"):
    return {
        "implementation": "numpy-serial-fp32-v1",
        "numpy": np.__version__,
        "reductions": "left-to-right separate FP32 multiply/add; no reassociation or FMA",
        "transcendentals": "pinned NumPy float32 ufuncs; libm and compiler trusted",
        "bf16": "round-to-nearest ties-even at explicit boundaries",
        "fp8": "E4M3FN saturating nearest-even; per-row max(abs(x))/448; scale floor 1/(448*512)",
        "mxfp4": "low nibble first; E2M1; E8M0/32; exponent bias127; reject scale255",
        "attention": "native dense causal GQA; logical positions only; no Quest approximation",
        "sampler": "greedy, first vocabulary index wins an exact tie",
        "reference_implementation": "TESTED",
        "native_equivalence": "UNPROVED",
        "precision": reference_precision(profile),
        "stock_implementation_equivalence": "UNPROVED; canonical arithmetic is not stock dispatch",
    }


def reference_semantics(files, config, kv_scales, profile):
    """Bind model choices independently of a caller-supplied contract digest."""
    arithmetic = reference_contract(profile)
    precision = arithmetic["precision"]
    return {
        "weights": {"files": files, "config": config},
        "weight_quantization": {
            "format": "MXFP4 E2M1/E8M0",
            "group": 32,
            "nibble_order": "low_first",
        },
        "activation_quantization": (
            {"format": "E4M3FN", "policy": arithmetic["fp8"]}
            if precision["activation_fp8"]
            else {"format": "BF16", "extra_quantizer": False}
        ),
        "attention": {"method": "native dense causal GQA", "sparse_approximation": False},
        "kv_representation": {
            "format": precision["kv_encoding"],
            "scales": kv_scaling(config, kv_scales, profile),
        },
        "recurrence": {
            "state": "FP32",
            "conv_history": "BF16 values",
            "algorithm": "serial gated delta",
        },
        "position_encoding": {"text_rope": config.get("rope_parameters")},
        "tokenizer": {"domain": "explicit token IDs; tokenization itself outside this replay"},
        "chat_template": {
            "domain": "supplied rendered prefix; template construction outside this replay"
        },
        "sampler": {
            "method": "greedy full vocabulary",
            "tie_break": "lowest token ID",
            "forced_replay": "diagnostic inputs only; never published as model decisions",
        },
        "numerical_contract": arithmetic,
    }
