import importlib.util
import types
from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def test_dummy_compatibility_cannot_bypass_live_or_graph_captured_repairs():
    path = Path(__file__).parents[1] / "optimized_d7_startup.py"
    spec = importlib.util.spec_from_file_location("optimized_startup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    md = types.SimpleNamespace(num_prefills=2)
    capturing = [False]

    def repaired(*args):
        seen.append("repaired")
        if md.num_prefills != 1:
            raise DiagnosticError("real multi-sequence request rejected")

    native = types.SimpleNamespace(
        forward_core_fused=repaired,
        conv_update=lambda *args: seen.append("synthetic convolution"),
        recurrent_update=lambda *args: seen.append("synthetic recurrent"),
    )
    installed = HookSet()
    installed.replace(native, "conv_update", repaired)
    installed.replace(native, "recurrent_update", repaired)

    def original(*args):
        native.conv_update()
        native.recurrent_update()
        seen.append("synthetic original")

    repairs = types.SimpleNamespace(
        hooks=installed,
        prefill=types.SimpleNamespace(
            native=native,
            original=original,
        ),
    )
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_current_stream_capturing=lambda: capturing[0],
        )
    )
    args = (None, types.SimpleNamespace(shape=(8, 10240)), None, None, None)

    def repaired_attention(*args):
        seen.append("repaired attention")

    repaired_attention.__wrapped__ = lambda *args: seen.append("synthetic attention")
    attention = types.SimpleNamespace(forward=repaired_attention.__wrapped__)
    installed.replace(attention, "forward", repaired_attention)
    # Performance optimizations wrap the repair, but startup must still find
    # the original native multi-sequence implementation from the binding map.
    performance = HookSet()

    def faster_attention(*args):
        seen.append("faster repaired attention")

    faster_attention.__wrapped__ = repaired_attention
    performance.replace(attention, "forward", faster_attention)
    with module.startup_prefill_compatibility(repairs, torch, attention) as receipt:
        native.forward_core_fused(*args)
        assert seen == ["synthetic convolution", "synthetic recurrent", "synthetic original"]
        assert receipt["synthetic_gdn_calls"] == 1
        assert receipt["conv_update"] == receipt["recurrent_update"] == 1
        attention.forward(None, None, args[1])
        assert seen[-1] == "synthetic attention"
        large = types.SimpleNamespace(shape=(2048, 10240))
        native.forward_core_fused(None, large, None, None, None)
        attention.forward(None, None, large)
        with pytest.raises(DiagnosticError, match="warm-up batch"):
            attention.forward(None, None, types.SimpleNamespace(shape=(2049, 10240)))
        with pytest.raises(DiagnosticError, match="warm-up batch"):
            native.forward_core_fused(
                None, types.SimpleNamespace(shape=(2049, 10240)), None, None, None
            )
        capturing[0] = True
        with pytest.raises(DiagnosticError, match="captured graph"):
            native.forward_core_fused(*args)
        with pytest.raises(DiagnosticError, match="captured graph"):
            attention.forward(None, None, args[1])
        with pytest.raises(DiagnosticError, match="captured graph"):
            native.conv_update()
        capturing[0] = False
        md.num_prefills = 1
        native.forward_core_fused(*args)
        assert seen[-1] == "synthetic original"
    assert native.forward_core_fused is repaired
    assert native.conv_update is repaired
    assert native.recurrent_update is repaired
    assert attention.forward is faster_attention
    md.num_prefills = 2
    with pytest.raises(DiagnosticError, match="real multi-sequence"):
        native.forward_core_fused(*args)
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        module.startup_prefill_compatibility(repairs, torch, attention),
    ):
        raise RuntimeError("startup failed")
    assert native.forward_core_fused is repaired
    assert native.conv_update is repaired
    assert native.recurrent_update is repaired
    assert attention.forward is faster_attention
    performance.close()
    assert attention.forward is repaired_attention
    installed.close()
