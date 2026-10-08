import functools
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_r9700_lab.conformance_instrumentation import (
    CallRecorder,
    HookSet,
    callable_identity,
    compare_calls,
)
from qwen_r9700_lab.conformance_state import load_arrays
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest


class Layer:
    def __init__(self, fault=False):
        self.fault = fault

    def inner(self, x):
        x[0] += 1 + int(self.fault)
        return x * np.float32(2)

    def forward(self, x):
        return self.inner(x) + np.float32(3)


class CustomOpDef:
    __module__ = "torch._library.custom_ops"

    def __init__(self, fn):
        self._qualname = "qualification::example"
        self._schema = "(Tensor x) -> Tensor"
        self._init_fn = fn
        self._backend_fns = {}
        self._disabled_kernel = set()
        self.register(fn)

    def register(self, fn):
        def wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        self._backend_fns["cpu"] = wrapped_fn

    def __call__(self, value):
        return self._backend_fns["cpu"](value)


def test_custom_operator_binding_tracks_registered_implementation(tmp_path):
    def initial(x):
        return x + 1

    def replacement(x):
        return x + 2

    operator = CustomOpDef(initial)
    before = callable_identity(operator)
    operator.register(replacement)
    assert callable_identity(operator) != before
    operator.register(initial)
    assert callable_identity(operator) == before
    owner, hooks = SimpleNamespace(op=operator), HookSet()
    rec = recorder(tmp_path / "custom-op", required=("custom",))
    rec.bind(owner, "op", site="custom", hooks=hooks, source_sha256=before)
    np.testing.assert_array_equal(owner.op(np.zeros(3)), np.ones(3))
    operator.register(replacement)
    with pytest.raises(DiagnosticError, match="registration changed"):
        owner.op(np.zeros(3))
    rec.finish()
    hooks.close()
    assert owner.op is operator


def recorder(path, *, mode="tensor", export=np.asarray, required=("layer", "inner")):
    return CallRecorder(
        path,
        contract=digest("contract"),
        execution=digest("execution"),
        adapter=digest("adapter"),
        tensor_export=export,
        is_tensor=lambda v: isinstance(v, np.ndarray),
        mode=mode,
        required_sites=required,
    )


def record_model(path, fault=False):
    rec, hooks, layer = recorder(path), HookSet(), Layer(fault)
    rec.bind(layer, "forward", site="layer", hooks=hooks)
    rec.bind(layer, "inner", site="inner", hooks=hooks)
    value = np.arange(4, dtype="<f4")
    result = layer.forward(value)
    rec.finish()
    hooks.close()
    assert "forward" not in vars(layer) and "inner" not in vars(layer)
    return result


def test_real_nested_calls_detect_first_causal_difference_before_parent_output(tmp_path):
    record_model(tmp_path / "reference")
    record_model(tmp_path / "candidate", True)
    result = compare_calls(tmp_path / "reference", tmp_path / "candidate", tmp_path / "diff.json")
    assert not result["equal"]
    assert result["first_difference"]["site"] == "inner"
    assert result["first_difference"]["phase"] == "after"
    _, before = load_arrays(tmp_path / "candidate" / "call-000000001" / "before")
    _, after = load_arrays(tmp_path / "candidate" / "call-000000001" / "after")
    assert before["args.0"][0] == 0  # cannot alias the mutated live input
    assert after["args.0"][0] == 2


def test_equal_calls_differ_in_timing_and_execution_but_compare_exact_tensors(tmp_path):
    record_model(tmp_path / "a")
    record_model(tmp_path / "b")
    assert compare_calls(tmp_path / "a", tmp_path / "b", tmp_path / "diff.json")["equal"]


def test_metadata_mode_never_reads_tensor_values_and_cannot_pass_numerical_comparison(tmp_path):
    def forbidden(_):
        pytest.fail("metadata observation exported a tensor")

    rec = recorder(tmp_path / "trace", mode="metadata", export=forbidden, required=("layer",))
    hooks, layer = HookSet(), Layer()
    rec.bind(layer, "forward", site="layer", hooks=hooks)
    layer.forward(np.zeros(3, dtype="<f4"))
    assert rec.finish()["mode"] == "metadata"
    hooks.close()
    with pytest.raises(DiagnosticError, match="tensor evidence"):
        compare_calls(tmp_path / "trace", tmp_path / "trace", tmp_path / "result.json")


def test_source_binding_is_checked_before_mutation_and_hooks_restore_after_errors(tmp_path):
    rec, hooks, layer = recorder(tmp_path / "calls"), HookSet(), Layer()
    with pytest.raises(DiagnosticError, match="source changed"):
        rec.bind(layer, "forward", site="layer", hooks=hooks, source_sha256=digest("wrong"))
    assert not hooks.entries and "forward" not in vars(layer)
    rec.bind(
        layer,
        "forward",
        site="working",
        hooks=hooks,
        source_sha256=callable_identity(layer.forward),
    )
    with pytest.raises(IndexError):
        layer.forward(np.array([], dtype="<f4"))
    hooks.close()
    assert "forward" not in vars(layer)
    with pytest.raises(DiagnosticError, match="incomplete"):
        rec.finish()


def test_missing_site_and_empty_scope_never_become_qualified(tmp_path):
    rec = recorder(tmp_path / "empty")
    with pytest.raises(DiagnosticError, match="incomplete"):
        rec.finish()
    rec, hooks, layer = recorder(tmp_path / "missing"), HookSet(), Layer()
    rec.bind(layer, "forward", site="layer", hooks=hooks)
    layer.forward(np.zeros(2, dtype="<f4"))
    hooks.close()
    with pytest.raises(DiagnosticError, match="never observed"):
        rec.finish()


def test_cleanup_does_not_clobber_a_later_hook_and_restores_other_sites():
    layer, hooks = Layer(), HookSet()

    def first(value):
        return value

    def second(value):
        return value * 2

    hooks.replace(layer, "forward", first)
    hooks.replace(layer, "inner", first)
    layer.forward = second
    with pytest.raises(DiagnosticError, match="changed before cleanup"):
        hooks.close()
    assert layer.forward is second and "inner" not in vars(layer)


def test_native_probe_cleanup_always_runs_even_when_validation_fails(tmp_path):
    from qwen_r9700_lab.conformance_radiance import RadianceProbe

    probe = RadianceProbe.__new__(RadianceProbe)
    layer, hooks, removed = Layer(), HookSet(), []
    hooks.replace(layer, "forward", lambda x: x)
    probe.handles = [SimpleNamespace(remove=lambda: removed.append(True))]
    probe.hooks = hooks
    probe.campaign = SimpleNamespace(root=tmp_path)
    probe.plan = {"sha256": "a" * 64}
    probe.binding = {"sha256": "b" * 64}

    def fail():
        raise DiagnosticError("injected incomplete capture")

    probe.boundaries = SimpleNamespace(finish=fail)
    with pytest.raises(DiagnosticError, match="incomplete"):
        probe.finish()
    assert removed == [True] and "forward" not in vars(layer)


def test_wrapper_cannot_claim_identity_of_the_function_it_wraps():
    @functools.wraps(Layer.forward)
    def changed(*args):
        return Layer.forward(*args) + 1

    assert callable_identity(changed) != callable_identity(Layer.forward)


def test_resealed_call_manifest_cannot_relabel_tensor_execution(tmp_path):
    from qwen_r9700_lab.diagnostic_contract import private_json, seal

    record_model(tmp_path / "a")
    record_model(tmp_path / "b")
    path = tmp_path / "b" / "calls.json"
    doc = private_json(path)
    doc["execution"] = digest("different model execution")
    doc = seal({k: v for k, v in doc.items() if k != "sha256"})
    import json

    path.write_text(json.dumps(doc))
    with pytest.raises(DiagnosticError, match="different execution"):
        compare_calls(tmp_path / "a", tmp_path / "b", tmp_path / "bad.json")
