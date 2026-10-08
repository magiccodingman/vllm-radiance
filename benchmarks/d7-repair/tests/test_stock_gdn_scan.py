"""CPU admission checks for the ordered native scan candidate."""

import importlib
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    return importlib.import_module("stock_gdn_scan")


@pytest.mark.parametrize("keep", [(True,), (-1,), (8,), (1.0,), (0, 0)])
def test_invalid_state_selection_rejected(module, keep):
    with pytest.raises(DiagnosticError):
        module.retained_rows(8, keep)


def test_all_accept_widths_and_empty_sequence(module):
    assert module.retained_rows(8, range(8)) == tuple(range(8))
    assert module.retained_rows(0, ()) == ()
    with pytest.raises(DiagnosticError):
        module.retained_rows(0, (0,))


def test_empty_transition_is_independent_identity_and_cpu_does_not_launch(module):
    torch = pytest.importorskip("torch")
    scan = module.StockScan(torch, "cpu")
    state = torch.zeros((48, 128, 128), dtype=torch.float32)
    qkv = torch.empty((0, 10240), dtype=torch.bfloat16)
    gates = torch.empty((0, 48), dtype=torch.bfloat16)
    a_log = torch.zeros(48, dtype=torch.float32)
    bias = torch.zeros(48, dtype=torch.bfloat16)
    result = scan.run(state, qkv, gates, gates, a_log, bias)
    assert result.rows == 0 and result.after_rows == {}
    assert result.outputs.shape == (0, 48, 128)
    result.final_state[0, 0, 0] = 1
    assert state[0, 0, 0] == 0
    with pytest.raises(DiagnosticError, match="requires a GPU"):
        scan.run(
            state,
            torch.zeros((1, 10240), dtype=torch.bfloat16),
            torch.zeros((1, 48), dtype=torch.bfloat16),
            torch.zeros((1, 48), dtype=torch.bfloat16),
            a_log,
            bias,
        )
