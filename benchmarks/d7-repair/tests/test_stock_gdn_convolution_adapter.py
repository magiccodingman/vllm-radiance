"""Convolution ABI/isolation checks in CPU Torch; native checks are separate."""

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requires pinned CPU-only stack image")


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    return importlib.import_module("stock_gdn_convolution_adapter")


@pytest.mark.parametrize("previous", range(1, 9))
def test_previous_acceptance_and_history_pass_through_without_changing_input(module, previous):
    x = torch.arange(8 * 10240).reshape(8, 10240).to(torch.bfloat16)
    original = x.clone()
    state = torch.zeros((3, 10240, 10), dtype=torch.bfloat16)
    weight = torch.zeros((10240, 4), dtype=torch.bfloat16)
    indices = torch.tensor([2], dtype=torch.int32)
    accepted = torch.tensor([previous], dtype=torch.int32)
    cu = torch.tensor([0, 8], dtype=torch.int32)

    def update(raw, history, weights, bias, activation, **kwargs):
        assert raw is x and history is state and weights is weight
        assert bias is None and activation == "silu"
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["conv_state_indices"] is indices
        assert kwargs["query_start_loc"] is cu and kwargs["max_query_len"] == 8
        out = kwargs["out"]
        assert out.data_ptr() != x.data_ptr()
        out.copy_(raw)
        return out

    adapter = module.StockConvolutionAdapter(torch, update)
    result = adapter(x, weight, None, state, 10, indices, accepted, cu, 1, 8, 48, 16, 8)
    assert adapter.calls == 1 and torch.equal(x, original)
    assert [tuple(v.shape) for v in result] == [(8, 16, 128), (8, 16, 128), (8, 48, 128)]
    assert all(v.is_contiguous() for v in result)
    assert torch.equal(torch.cat([v.flatten(1) for v in result], -1), x)


def test_captured_midpoint_rounds_to_even():
    # Layer 4, value channel 3887: the FP32 SiLU is exactly a BF16
    # midpoint. R4D's add-0x8000 conversion chooses the other endpoint.
    x = torch.tensor([0.00293731689453125], dtype=torch.float32)
    assert x.to(torch.bfloat16).item() == 0.0029296875
    bits = x.view(torch.int32)
    away = ((bits + 0x8000) >> 16).to(torch.int16).view(torch.bfloat16)
    assert away.item() == 0.0029449462890625
