"""Fresh/continued prefill ownership checks with CPU-only stand-in operators."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError

torch = pytest.importorskip("torch", reason="requires pinned CPU-only stack image")


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    return importlib.import_module("stock_gdn_prefill_adapter")


@pytest.mark.parametrize("has_initial", [False, True])
def test_prefill_selects_only_owned_history_and_publishes_final_state(module, has_initial):
    count = 4
    state = torch.full((3, 48, 128, 128), 7.0)
    conv_state = torch.full((3, 10240, 10), 5.0, dtype=torch.bfloat16)
    before = state.clone()
    conv_before = conv_state.clone()
    md = SimpleNamespace(
        num_prefills=1,
        prefill_state_indices=torch.tensor([2], dtype=torch.int32),
        prefill_has_initial_state=torch.tensor([has_initial]),
        has_initial_state=torch.tensor([has_initial]),
        non_spec_state_indices_tensor=torch.tensor([[1]], dtype=torch.int32),
    )
    cu = torch.tensor([0, count], dtype=torch.int32)
    plan = ("prefill", count, conv_state, state, md, (None, cu))
    native = SimpleNamespace(_plan=lambda *args: plan)
    layer = SimpleNamespace(
        A_log=torch.zeros(48),
        dt_bias=torch.zeros(48, dtype=torch.bfloat16),
        conv1d=SimpleNamespace(weight=torch.ones((10240, 1, 4), dtype=torch.bfloat16), bias=None),
        _radiance_z=torch.ones(1),
    )
    raw = torch.ones((count, 10240), dtype=torch.bfloat16)
    gates = torch.zeros((count, 48), dtype=torch.bfloat16)
    output = torch.empty((count, 48, 128), dtype=torch.bfloat16)

    def convolution(x, history, weights, bias, activation, **kwargs):
        assert torch.all(history[1] == (5 if has_initial else 0))
        assert torch.equal(history[[0, 2]], conv_before[[0, 2]])
        assert kwargs.get("num_accepted_tokens") is None
        assert kwargs["out"].data_ptr() != raw.data_ptr()
        kwargs["out"].fill_(2)
        history[1, :, :3] = 3
        return kwargs["out"]

    def scan(initial, qkv, a, b, a_log, dt_bias):
        assert torch.all(initial == (7 if has_initial else 0))
        assert torch.all(qkv == 2)
        return SimpleNamespace(
            outputs=torch.full_like(output, 4), final_state=torch.full_like(initial, 9)
        )

    adapter = module.StockPrefillAdapter(
        torch, native, convolution, SimpleNamespace(run=scan), None
    )
    assert adapter(layer, raw, gates, gates, output) is True
    assert adapter.calls == 1 and adapter.rows == count
    assert torch.all(state[2] == 9) and torch.equal(state[:2], before[:2])
    assert torch.all(raw == 1) and torch.all(output == 4)
    assert not hasattr(layer, "_radiance_z")


def test_decode_is_delegated_and_unqualified_mixed_prefill_is_rejected(module):
    plan = ["decode"]
    sentinel = object()
    adapter = module.StockPrefillAdapter(
        torch, SimpleNamespace(_plan=lambda *a: plan), None, None, lambda *a: sentinel
    )
    assert adapter(None, None, None, None, None) is sentinel
    plan[:] = ["prefill+spec", 8, None, None, SimpleNamespace(num_prefills=1), (None, None)]
    with pytest.raises(DiagnosticError, match="unmixed"):
        adapter(None, None, None, None, None)
