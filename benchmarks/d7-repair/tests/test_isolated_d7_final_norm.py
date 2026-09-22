"""CPU checks of the expanded compiled norm cut; numerical execution is native."""

import ast
import re
from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal


@pytest.fixture
def cuts():
    path = (
        Path(__file__).resolve().parents[1]
        / "probe_isolated_d7_final_norm.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cuts"
    ]
    namespace = {"re": re, "require": require, "authenticate": authenticate}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["cuts"]


def event(owner, index):
    return {
        "operation": "qwen_d7_qualified.gemma_residual.default",
        "index": index,
        "logical_identities": [owner],
    }


def test_final_norm_includes_its_retained_residual_producer(cuts):
    previous = event("model.layers.63.post_attention_layernorm", 688)
    final = event("model.norm", 694)
    assert cuts(seal({"events": [previous, final]})) == (final, previous)


def test_cannot_use_another_layers_carry(cuts):
    with pytest.raises(DiagnosticError, match="ambiguous"):
        cuts(
            seal(
                {
                    "events": [
                        event("model.layers.62.post_attention_layernorm", 688),
                        event("model.norm", 694),
                    ]
                }
            )
        )


def test_duplicate_or_reversed_cuts_are_rejected(cuts):
    previous = event("model.layers.63.post_attention_layernorm", 698)
    final = event("model.norm", 694)
    with pytest.raises(DiagnosticError, match="precede"):
        cuts(seal({"events": [previous, final]}))
    with pytest.raises(DiagnosticError, match="ambiguous"):
        cuts(seal({"events": [previous, final, final]}))
