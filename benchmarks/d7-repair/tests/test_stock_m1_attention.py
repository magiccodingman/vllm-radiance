"""CPU checks of causal lengths, split boundaries and scratch admission; no GPU math claims."""

import ctypes
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    return importlib.import_module("stock_m1_attention")


@pytest.mark.parametrize(
    "context,expected",
    [
        (1, 1),
        (16, 1),
        (17, 2),
        (129, 9),
        (256, 16),
        (257, 16),
        (992, 16),
        (993, 16),
        (1008, 16),
        (1009, 32),
        (1024, 32),
        (253792, 32),
    ],
)
def test_pinned_split_law_at_discontinuous_boundaries(module, context, expected):
    assert module.m1_splits(context) == expected


def test_queries_across_a_split_boundary_keep_their_own_m1_split_counts(module):
    assert module.split_groups(8, 23) == [(0, 1, 1), (1, 8, 2)]
    assert module.split_groups(8, 1012) == [(0, 4, 16), (4, 8, 32)]
    assert module.split_groups(8, 253792) == [(0, 8, 32)]


@pytest.mark.parametrize("width,bound", [(9, 100), (0, 100), (8, 7), (True, 100)])
def test_unqualified_queries_are_rejected(module, width, bound):
    with pytest.raises(DiagnosticError):
        module.split_groups(width, bound)


def test_native_dispatch_preserves_query_order_causal_lengths_and_descales(
    module, monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch", reason="requires pinned CPU-only stack image")
    source = tmp_path / "native.py"
    source.write_text("fake native ABI for CPU pointer-admission checks")
    monkeypatch.setattr(module, "ATTENTION_SOURCE", hashlib.sha256(source.read_bytes()).hexdigest())
    calls = []

    def launch(*args):
        q, _, table, lengths, out, k, v, _, count, query_length = args[:10]
        calls.append(
            {
                "lengths": list((ctypes.c_int * count).from_address(lengths)),
                "table": list((ctypes.c_int * (count * 3)).from_address(table)),
                "k": list((ctypes.c_float * (count * 4)).from_address(k)),
                "v": v,
                "splits": args[18],
                "query_length": query_length,
            }
        )
        ctypes.memmove(out, q, count * 24 * 256 * 2)

    class Impl:
        num_heads, num_kv_heads, head_size, scale = 24, 4, 256, 0.0625

        def _geometry(self, *args):
            return 0, 32768, 8192

        def forward(self, *args, **kwargs):
            raise AssertionError("unexpected original dispatch")

    native = SimpleNamespace(
        __file__=str(source),
        R4DAttentionImpl=Impl,
        _DECODE=[launch, launch],
        r4d=SimpleNamespace(attn_decode_h256_gqa6_scratch_bytes=lambda *args: 100),
    )
    monkeypatch.setitem(sys.modules, "radiance_r4d_attn", native)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    md = SimpleNamespace(
        r4d_plan=((0, 1, 8, 0),),
        causal=True,
        r4d_max_ctx=23,
        block_table=torch.tensor([[7, 3, 9]], dtype=torch.int32),
        seq_lens=torch.tensor([23], dtype=torch.int32),
        r4d_scratch=torch.empty(100, dtype=torch.uint8),
    )
    query = torch.arange(8 * 24 * 256).reshape(8, 24, 256).bfloat16()
    kv = torch.empty(1, dtype=torch.uint8)
    output = torch.empty_like(query)
    hooks = HookSet()
    try:
        repair = module.StockM1Attention(hooks)
        Impl().forward(SimpleNamespace(_k_scale_float=0.5), query, None, None, kv, md, output)
        assert torch.equal(output, query)
        assert [row["lengths"] for row in calls] == [[16], list(range(17, 24))]
        assert [row["splits"] for row in calls] == [1, 2]
        for row in calls:
            count = len(row["lengths"])
            assert row["table"] == [7, 3, 9] * count
            assert row["k"] == [0.5] * (count * 4)
            assert row["v"] == 0 and row["query_length"] == 1
        assert repair.receipt()["rows"] == 8
        md.r4d_scratch = torch.empty(99, dtype=torch.uint8)
        with pytest.raises(DiagnosticError, match="scratch capacity"):
            Impl().forward(SimpleNamespace(), query, None, None, kv, md, output)
        assert len(calls) == 2
    finally:
        hooks.close()
