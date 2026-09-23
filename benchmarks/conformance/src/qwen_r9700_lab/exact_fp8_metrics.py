"""Exact, bounded integer diagnostics for pairs of E4M3 cache bytes.

This does not decide tensor equality or authenticate files. Those checks remain
the caller's responsibility. The histogram replaces repeated floating-point
expansion only for the diagnostic sums, with an explicit overflow bound.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MAX_ELEMENTS = 16 * 1024**2
DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fnuz"})


@lru_cache(maxsize=2)
def tables(dtype):
    if dtype not in DTYPES:
        raise ValueError("unsupported FP8 representation")
    # Decode independently in integer units: FN's unit is 2^-9 and FNUZ's
    # is 2^-10. Both have the same integer exponent/significand expression.
    code = np.arange(256, dtype=np.int64)
    exponent, mantissa = (code >> 3) & 15, code & 7
    units = np.where(exponent == 0, mantissa, (8 + mantissa) << np.maximum(exponent - 1, 0))
    units = np.where(code & 128, -units, units)
    finite = code != 128 if dtype.endswith("fnuz") else ~((exponent == 15) & (mantissa == 7))
    valid = finite[:, None] & finite[None, :]
    absolute = np.where(valid, np.abs(units[:, None] - units[None, :]), 0).reshape(-1)
    square = absolute * absolute
    ref_square = np.where(valid, units[:, None] ** 2, 0).reshape(-1)
    if int(square.max()) * MAX_ELEMENTS > np.iinfo(np.int64).max:
        raise ValueError("FP8 integer accumulation bound is insufficient")
    result = {
        "absolute": absolute,
        "square": square,
        "reference_square": ref_square,
        "valid": valid.reshape(-1).astype(np.int64),
        "nonfinite_a": np.repeat(~finite, 256).astype(np.int64),
        "nonfinite_b": np.tile(~finite, 256).astype(np.int64),
        "scale": 1024 if dtype.endswith("fnuz") else 512,
    }
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return result


def chunk_metrics(left: bytes, right: bytes, dtype: str) -> dict:
    """Match the existing finite-pair long-double diagnostic sums exactly.

    At most 2^24 elements, each squared difference at most 491520^2 integer
    units, gives a nonnegative sum below 2^62. Conversion to an extended
    significand of at least 64 bits is exact. Power-of-two scaling is exact.
    """
    if len(left) != len(right) or len(left) > MAX_ELEMENTS:
        raise ValueError("FP8 metric chunk lengths differ or exceed the proved integer bound")
    if np.finfo(np.longdouble).nmant < 63:
        raise ValueError("exact FP8 metrics require at least a 64-bit significand")
    table = tables(dtype)
    a, b = np.frombuffer(left, np.uint8), np.frombuffer(right, np.uint8)
    pair = (a.astype(np.uint16) << 8) | b
    histogram = np.bincount(pair, minlength=65536).astype(np.int64, copy=False)
    scale = table["scale"]
    present = histogram != 0
    maximum = int(table["absolute"][present].max()) if present.any() else 0
    return {
        "max_abs": np.longdouble(maximum) / scale,
        "squared": np.longdouble(histogram @ table["square"]) / (scale * scale),
        "reference_squared": np.longdouble(histogram @ table["reference_square"]) / (scale * scale),
        "count": int(histogram @ table["valid"]),
        "nonfinite": [
            int(histogram @ table["nonfinite_a"]),
            int(histogram @ table["nonfinite_b"]),
        ],
    }
