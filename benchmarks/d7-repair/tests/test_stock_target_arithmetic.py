"""CPU wiring checks for target arithmetic controls; numerical GPU evidence is separate."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

torch = pytest.importorskip("torch", reason="requires pinned CPU-only stack image")


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    adapter = importlib.import_module("stock_target_arithmetic")
    gdn = importlib.import_module("stock_m1_gdn_norm")
    monkeypatch.setattr(gdn, "StockM1GdnNorm", lambda: lambda x, z, weight, eps: x + z)

    class GemmaRMSNorm:
        weight = torch.zeros(5120, dtype=torch.bfloat16)
        variance_epsilon = 1e-6

        def __init__(self):
            self.rows = []

        def forward(self, x, residual=None):
            self.rows.append(x.shape[0])
            # Deliberate shape-dependent fault: routing must eliminate it for D7.
            output = x.clone() + (x.shape[0] - 1)
            return output if residual is None else (output, residual.clone())

    class Head:
        def __init__(self):
            self.rows = []

        def _apply_head(self, lm_head, hidden, bias=None):
            self.rows.append(hidden.shape[0])
            return hidden[:, :7].clone() + (hidden.shape[0] - 1) + bias

    class RMSNormGated:
        weight = torch.ones(128, dtype=torch.bfloat16)
        bias = group_size = None
        norm_before_gate = True
        activation = "silu"
        eps = 1e-6

        def __init__(self):
            self.rows = []
            self._forward_method = self.forward_hip

        def forward_hip(self, x, z):
            self.rows.append(x.shape[0])
            return x + z + x.shape[0] - 48

        def forward(self, x, z):
            return self._forward_method(x, z)

    norms, head = [GemmaRMSNorm() for _ in range(129)], Head()
    names = [
        f"language_model.model.layers.{i}.{kind}"
        for i in range(64)
        for kind in ("input_layernorm", "post_attention_layernorm")
    ] + ["language_model.model.norm"]
    attention_norms = []
    for i in range(3, 64, 4):
        for kind in ("q", "k"):
            module = GemmaRMSNorm()
            module.weight = torch.zeros(256)
            attention_norms.append(
                (f"language_model.model.layers.{i}.self_attn.{kind}_norm", module)
            )
    model = SimpleNamespace(
        named_modules=lambda: [
            *zip(names, norms, strict=True),
            ("language_model.logits_processor", head),
            *attention_norms,
            *[
                (f"language_model.model.layers.{i}.linear_attn.norm", RMSNormGated())
                for i in range(64)
                if i % 4 != 3
            ],
        ]
    )
    inventory = model.named_modules()
    model.named_modules = lambda: inventory
    return adapter, model, norms, head


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("residual", (False, True))
def test_speculative_rows_preserve_independent_m1_operator_behavior(setup, count, residual):
    adapter, model, norms, head = setup
    hooks = HookSet()
    try:
        repair = adapter.TargetArithmetic(model, hooks, norm=True, head=True)
        x = torch.randn(count, 5120).bfloat16()
        saved = x.clone()
        carry = torch.randn_like(x) if residual else None
        out = norms[7].forward(x, carry)
        assert torch.equal(out[0] if residual else out, saved)
        if residual:
            assert torch.equal(out[1], carry) and out[1].data_ptr() != carry.data_ptr()
        assert torch.equal(x, saved)
        assert norms[7].rows == [1] * count
        logits = head._apply_head(SimpleNamespace(tp_size=1), x, torch.ones(7))
        assert torch.equal(logits, saved[:, :7] + torch.ones(7))
        assert head.rows == [1] * count
        assert repair.receipt()["head_rows"] == count
        assert repair.receipt()["norm_rows"] == count
    finally:
        hooks.close()
    norms[7].forward(torch.zeros(8, 5120))
    assert norms[7].rows[-1] == 8


def test_large_prefill_remains_on_original_path(setup):
    adapter, model, norms, head = setup
    hooks = HookSet()
    try:
        repair = adapter.TargetArithmetic(model, hooks, norm=True, head=True)
        x = torch.zeros(129, 5120)
        assert torch.equal(norms[0].forward(x), torch.full_like(x, 128))
        head._apply_head(SimpleNamespace(tp_size=1), x, torch.zeros(7))
        assert norms[0].rows == [129] and head.rows == [129]
        assert repair.receipt()["head_calls"] == repair.receipt()["norm_calls"] == 0
    finally:
        hooks.close()


def test_rejects_unqualified_tensor_parallel_head(setup):
    adapter, model, _, head = setup
    hooks = HookSet()
    try:
        adapter.TargetArithmetic(model, hooks, head=True)
        with pytest.raises(DiagnosticError, match="TP1"):
            head._apply_head(SimpleNamespace(tp_size=2), torch.zeros(8, 5120))
    finally:
        hooks.close()


@pytest.mark.parametrize("heads", (4, 24))
@pytest.mark.parametrize("count", range(1, 9))
def test_attention_norm_preserves_the_m1_head_grouping(setup, count, heads):
    adapter, model, _, _ = setup
    name = f"language_model.model.layers.3.self_attn.{'k' if heads == 4 else 'q'}_norm"
    module = dict(model.named_modules())[name]
    hooks = HookSet()
    try:
        adapter.TargetArithmetic(model, hooks, norm=True)
        x = torch.randn(count, heads, 256).bfloat16()
        assert torch.equal(module.forward(x), x)
        assert module.rows == [1] * count
    finally:
        hooks.close()


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("group_size", (2, 4, 5))
def test_supported_head_groups_retain_order_bias_and_all_rows(setup, count, group_size):
    adapter, model, _, head = setup
    calls = []

    def small_batch_only(lm_head, hidden, bias=None):
        calls.append(hidden.shape[0])
        # Model the actual dispatch boundary: batches over five change arithmetic.
        if hidden.shape[0] > 5:
            return torch.full_like(hidden[:, :7], float("nan"))
        return hidden[:, :7].clone() + bias

    head._apply_head = small_batch_only
    hooks = HookSet()
    try:
        repair = adapter.TargetArithmetic(model, hooks, head=True, head_group_size=group_size)
        x = torch.arange(count * 5120).reshape(count, 5120).float()
        bias = torch.arange(7).float()
        original = x.clone()
        out = head._apply_head(SimpleNamespace(tp_size=1), x, bias)
        assert torch.equal(out, x[:, :7] + bias)
        assert torch.equal(x, original)
        assert sum(calls) == count and max(calls) <= group_size
        assert repair.receipt()["head_mode"] == f"groups-{group_size}"
    finally:
        hooks.close()


@pytest.mark.parametrize("size", (0, 3, 6, 8, True))
def test_head_rejects_unqualified_group_sizes(setup, size):
    adapter, model, _, _ = setup
    with pytest.raises(DiagnosticError, match="qualified small-batch"):
        adapter.TargetArithmetic(model, HookSet(), head=True, head_group_size=size)


@pytest.mark.parametrize("count", range(1, 9))
def test_gdn_norm_removes_row_grouping_fault_and_preserves_m1(setup, count):
    adapter, model, _, _ = setup
    norm = dict(model.named_modules())["language_model.model.layers.28.linear_attn.norm"]
    hooks = HookSet()
    try:
        repair = adapter.TargetArithmetic(model, hooks, norm=True)
        x = torch.ones(count * 48, 128)
        z = torch.full_like(x, 2)
        saved = (x.clone(), z.clone())
        assert torch.equal(norm.forward(x, z), x + z)
        assert norm.rows == ([48] if count == 1 else [])
        assert repair.receipt()["gdn_norm_rows"] == (0 if count == 1 else count)
        assert all(torch.equal(a, b) for a, b in zip(saved, (x, z), strict=True))
        large = torch.ones(9 * 48, 128)
        norm.forward(large, large)
        assert norm.rows[-1] == 9 * 48
    finally:
        hooks.close()
