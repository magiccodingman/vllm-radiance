"""CPU checks of the actual replay coordinator, not GPU arithmetic evidence."""

import importlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_gdn_contract import read_contract
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/profiles/gdn-stock-m1-arithmetic-v1.json"
ARTIFACTS = Path(os.environ.get(
    "RADIANCE_GDN_REFERENCE_ARTIFACTS",
    ROOT / "artifacts/conformance/20260916-gdn-arithmetic-contract-001",
))


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("stock_gdn_sequence")


@pytest.fixture
def contract():
    return read_contract(CONTRACT)


def test_changed_contract_is_not_admitted(module, contract):
    changed = deepcopy(contract)
    changed.pop("sha256")
    changed["precision"]["beta"] = "FP32"
    with pytest.raises(DiagnosticError, match="unreviewed"):
        module.check_executable(seal(changed), b"", {})


def test_unknown_binary_is_rejected_before_a_launch(module, contract):
    with pytest.raises(DiagnosticError, match="executable differs"):
        module.check_executable(contract, b"another GDN kernel", {})


def test_retained_executable_and_metadata_are_required(module, contract):
    authority = contract["arithmetic_authority"]
    base = ARTIFACTS / authority["cache_entry"]
    if not (base / (module.KERNEL + ".hsaco")).exists():
        pytest.skip("Set RADIANCE_GDN_REFERENCE_ARTIFACTS to the qualified compiler artifacts")
    binary = (base / (module.KERNEL + ".hsaco")).read_bytes()
    metadata = json.loads((base / (module.KERNEL + ".json")).read_text())
    assert (
        module.check_executable(contract, binary, metadata)
        == authority["files"][module.KERNEL + ".hsaco"]
    )
    for name, value in ("enable_fp_fusion", False), ("num_warps", 4), ("target", {}):
        changed = {**metadata, name: value}
        with pytest.raises(DiagnosticError, match="metadata changed"):
            module.check_executable(contract, binary, changed)


@pytest.fixture
def torch():
    result = pytest.importorskip("torch")
    result.set_num_threads(1)
    return result


@pytest.fixture
def inputs(torch, module):
    # Distinct rows make an off-by-one commit or replay observable. Values are
    # deliberately synthetic; this transition is not an emulation of GDN.
    initial = torch.zeros(module.STATE_SHAPE, dtype=torch.float32)
    qkv = torch.zeros((8, 10240), dtype=torch.bfloat16)
    a = torch.arange(1, 9, dtype=torch.bfloat16)[:, None].expand(8, 48).clone()
    b = torch.zeros((8, 48), dtype=torch.bfloat16)
    return initial, qkv, a, b, torch.zeros(48), torch.zeros(48, dtype=torch.bfloat16)


@pytest.fixture
def runner(module, contract, torch):
    calls = []

    def factory(workspace):
        def step():
            calls.append(float(workspace.a[0, 0]))
            increment = (workspace.a + workspace.b + workspace.qkv[:, :48]).float()
            workspace.pool[1].mul_(0.5).add_(increment[0, :, None, None])
            workspace.output[0, 0].copy_(workspace.pool[1, :, :, 0])

        return step

    result = module._Sequence(torch, contract, "cpu", factory)
    result.test_calls = calls
    return result


@pytest.mark.parametrize("accepted", range(8))
def test_every_acceptance_width_commits_only_its_processed_prefix(runner, inputs, accepted, torch):
    initial, qkv, a, b, a_log, dt = inputs
    before = [value.clone() for value in inputs]
    result = runner.d7(*inputs)
    expected = initial.clone()
    for row in range(accepted + 1):
        expected = expected * 0.5 + a[row].float()[:, None, None]
    assert torch.equal(result.accepted_state(accepted), expected)
    assert result.rows == 8
    assert runner.test_calls == list(range(1, 9))
    assert all(torch.equal(left, right) for left, right in zip(inputs, before, strict=True))
    selected = result.accepted_state(accepted)
    selected.zero_()
    assert torch.equal(result.accepted_state(accepted), expected)
    # Retained states and outputs must survive scratch reuse by another call.
    copy = result.final_state.clone()
    runner.run(initial, qkv[:1], a[:1], b[:1], a_log, dt)
    assert torch.equal(result.final_state, copy)


@pytest.mark.parametrize("accepted", range(7))
def test_rejected_suffix_does_not_contaminate_any_retained_prefix(runner, inputs, accepted, torch):
    result = runner.d7(*inputs)
    poisoned = [value.clone() for value in inputs]
    for value in poisoned[1:4]:
        value[accepted + 1 :] = -31
    other = runner.d7(*poisoned)
    assert torch.equal(result.accepted_state(accepted), other.accepted_state(accepted))
    assert torch.equal(result.outputs[: accepted + 1], other.outputs[: accepted + 1])
    assert not torch.equal(result.final_state, other.final_state)


@pytest.mark.parametrize("chunks", [(8,), (4, 4), (0, 3, 0, 5, 0), (1,) * 8, (2, 3, 3)])
def test_prefill_partitions_include_empty_identity(runner, inputs, chunks, torch):
    initial, qkv, a, b, a_log, dt = inputs
    whole = runner.run(*inputs)
    state, cursor, outputs = initial, 0, []
    for size in chunks:
        end = cursor + size
        part = runner.run(state, qkv[cursor:end], a[cursor:end], b[cursor:end], a_log, dt)
        state = part.final_state
        outputs.append(part.outputs)
        cursor = end
    assert cursor == 8
    assert torch.equal(whole.final_state, state)
    assert torch.equal(whole.outputs, torch.cat(outputs))


def test_physical_packing_matches_the_preserved_executable(runner):
    assert runner.qkv.stride() == (16384, 1)
    assert runner.a.stride() == runner.b.stride() == (96, 1)
    assert runner.pool.stride() == (802816, 16384, 128, 1)
    assert runner.indices.stride() == (1,)
    assert runner.indices.tolist() == [1]


@pytest.mark.parametrize("fault", ["guard", "index", "qkv", "gate", "nan", "missing", "error"])
def test_tentative_results_are_not_returned_after_a_fault(runner, inputs, fault):
    normal = runner._launch

    def broken():
        if fault == "missing":
            return
        if fault == "error":
            raise DiagnosticError("injected launch error")
        normal()
        if fault == "guard":
            runner.storage[0] = 0
        elif fault == "index":
            runner.indices[0] = 2
        elif fault == "qkv":
            runner.qkv[0, 0] = 1
        elif fault == "gate":
            runner.A_log[0] = 1
        elif fault == "nan":
            runner.pool[1, 0, 0, 0] = float("nan")

    runner._launch = broken
    with pytest.raises(DiagnosticError):
        runner.d7(*inputs)
    assert not runner._lock.locked()


@pytest.mark.parametrize("bad", [-1, 8, True, "1", 1.0])
def test_invalid_acceptance_counts_do_not_select_a_state(runner, inputs, bad):
    result = runner.d7(*inputs)
    with pytest.raises(DiagnosticError):
        result.accepted_state(bad)


def test_partial_sequence_cannot_be_used_as_a_d7_result(runner, inputs):
    with pytest.raises(DiagnosticError, match="complete D7"):
        runner.run(*inputs).accepted_state(0)


def test_invalid_inputs_fail_before_any_transition(runner, inputs):
    for a in (inputs[2].float(), inputs[2][:7], inputs[2] * float("nan")):
        changed = (*inputs[:2], a, *inputs[3:])
        with pytest.raises(DiagnosticError, match="tensor"):
            runner.d7(*changed)
    for keep in ((8,), (-1,), (True,), (1, 1)):
        with pytest.raises(DiagnosticError):
            runner.run(*inputs, retain_rows=keep)
    assert runner.test_calls == []


def test_scratch_cannot_be_shared_by_concurrent_replays(runner, inputs):
    with runner._lock, pytest.raises(DiagnosticError, match="already in use"):
        runner.d7(*inputs)
    assert runner.test_calls == []
