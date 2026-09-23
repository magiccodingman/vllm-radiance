import ast
import hashlib

import pytest

from qwen_r9700_lab import conformance_rotary_repair as repair
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def test_intervention_refuses_unpinned_source():
    with pytest.raises(DiagnosticError, match="pinned"):
        repair.patch_source("unknown source")


def test_pinned_intervention_only_changes_products_and_keeps_layout_branches(monkeypatch):
    expressions = [
        f"    new_{kind}_{i} = {kind}_tile_{i} * cos_row {op} {kind}_tile_{3 - i} * sin_row"
        for kind in ("q", "k")
        for i, op in ((1, "-"), (2, "+"))
    ]
    source = "@triton.jit\ndef _triton_mrope_forward():\n" + "\n".join(expressions * 2) + "\n"
    monkeypatch.setattr(repair, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    result = repair.patch_source(source)
    ast.parse(result)
    assert result.count("_diagnostic_bf16_product_rne(") == 17  # one definition, 16 products
    assert result.count(" - ") == 4 and result.count(" + ") == 4
    assert 'fp_downcast_rounding="rtne"' in result
    bad = source.replace("q_tile_1 * cos_row", "q_tile_1", 1)
    monkeypatch.setattr(repair, "SOURCE_SHA256", hashlib.sha256(bad.encode()).hexdigest())
    with pytest.raises(DiagnosticError, match="arithmetic changed"):
        repair.patch_source(bad)
