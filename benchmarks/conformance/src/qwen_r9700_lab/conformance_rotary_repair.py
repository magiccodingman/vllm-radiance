"""Construct an isolated, pinned MRoPE product-rounding intervention.

This never installs a production override. A native probe must validate the
generated module against the recorded input/output contract before using it.
"""

import hashlib

from qwen_r9700_lab.conformance_topk import require

SOURCE_SHA256 = "f795e2a347715d1c8e2b9953fcf6fba578ca9a648858ed9d26d27eb6bdd61d19"

HELPER = """@triton.jit
def _diagnostic_bf16_product_rne(a, b):
    return (a.to(tl.float32) * b.to(tl.float32)).to(
        a.dtype, fp_downcast_rounding="rtne"
    )


"""


def patch_source(source):
    require(
        hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256,
        "native MRoPE source is outside the pinned intervention",
    )
    marker = "@triton.jit\ndef _triton_mrope_forward("
    require(source.count(marker) == 1, "missing unique native rotary entry")
    result = source.replace(marker, HELPER + marker)
    for kind in ("q", "k"):
        for index in (1, 2):
            for coefficient in ("cos_row", "sin_row"):
                expression = f"{kind}_tile_{index} * {coefficient}"
                require(result.count(expression) == 2, "native rotary arithmetic changed")
                result = result.replace(
                    expression,
                    f"_diagnostic_bf16_product_rne({kind}_tile_{index}, {coefficient})",
                )
    return result
