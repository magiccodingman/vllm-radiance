import numpy as np
import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.conformance_boundaries import BoundaryRecorder, compare_boundaries
from qwen_r9700_lab.conformance_replay import run_reference
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, private_json


@pytest.mark.parametrize(
    "positions,stages",
    [
        ([], [["input"]]),
        ([0, 0], [["input"]]),
        ([-1], [["input"]]),
        ([True], [["input"]]),
        ([0], [["input", "input"]]),
        ([0], [["x/../../escape"]]),
    ],
)
def test_invalid_or_empty_observation_domains_are_rejected(tmp_path, positions, stages):
    with pytest.raises(DiagnosticError):
        BoundaryRecorder(
            tmp_path / "capture",
            contract=digest("c"),
            execution=digest("e"),
            adapter=digest("a"),
            positions=positions,
            layers=1,
            layer_stages=stages,
            input_digests={p: digest(p) for p in positions},
        )


def test_detailed_reference_really_captures_all_gdn_and_attention_boundaries(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    run_reference(plan, tmp_path / "reference")
    doc = private_json(tmp_path / "reference" / "semantic" / "boundaries.json")
    assert doc["positions"] == list(range(6))  # includes early prefill, not just its final token
    stages = doc["layer_stages"]
    assert {"conv_input_state", "conv_state", "gdn_input_state", "gdn_state"} <= set(stages[0])
    assert {"attention_q_rope", "attention_key_stored", "attention_output"} <= set(stages[-1])
    assert len(doc["frames"]) == len(doc["positions"]) * sum(map(len, stages))
    result = compare_boundaries(
        tmp_path / "reference" / "semantic",
        tmp_path / "reference" / "semantic",
        tmp_path / "comparison",
    )
    assert result["equal"]


def test_missing_required_low_level_observation_cannot_finish(tmp_path):
    recorder = BoundaryRecorder(
        tmp_path / "capture",
        contract=digest("c"),
        execution=digest("e"),
        adapter=digest("a"),
        positions=[0],
        layers=1,
        layer_stages=[["gdn_input_state", "gdn_state"]],
        input_digests={0: digest("prefix")},
    )
    recorder.record(0, 0, "gdn_state", np.zeros(2))
    with pytest.raises(DiagnosticError, match="incomplete"):
        recorder.finish()


def test_prefill_and_accepted_row_domain_omits_emitted_but_pending_token(tmp_path):
    from qwen_r9700_lab.conformance_replay import observation_domain, scheduled_inputs
    from qwen_r9700_lab.diagnostic_contract import seal

    plan = tiny_plan(tmp_path / "checkpoint")
    plan["forced_tokens"] = list(range(32)) + list(range(5))
    plan["accepted_widths"] = list(range(8))
    plan = seal({k: v for k, v in plan.items() if k != "sha256"})
    positions, identities = observation_domain(plan)
    tokens = plan["prefix"] + plan["forced_tokens"][:-1]
    assert positions == list(range(len(tokens)))
    assert len(positions) == list(scheduled_inputs(plan))[-1]["consumed"]
    assert all(identities[p] == digest(tokens[: p + 1]) for p in positions)


def test_native_boundaries_include_early_prefill_but_not_rejected_speculative_suffix(tmp_path):
    from types import SimpleNamespace

    from qwen_r9700_lab.conformance_radiance import RadianceProbe

    observed = []
    probe = RadianceProbe.__new__(RadianceProbe)
    probe.index = 0
    probe.campaign = SimpleNamespace(expected=[{"consumed": 3}])
    probe.positions = [0, 1, 2, 3, 4]  # two tentative suffix rows
    probe.boundaries = SimpleNamespace(record=lambda *args: observed.append(args[:3]))
    probe.record_boundary(0, "input_norm", np.zeros((5, 2)))
    assert observed == [(0, 0, "input_norm"), (1, 0, "input_norm"), (2, 0, "input_norm")]


def test_selected_capture_window_is_explicit_and_never_claims_full_prefill_coverage(tmp_path):
    from qwen_r9700_lab.conformance_replay import observation_domain, validate_plan
    from qwen_r9700_lab.diagnostic_contract import seal

    plan = tiny_plan(tmp_path / "checkpoint")
    plan["observation_positions"] = [1, 4]
    plan = seal({k: v for k, v in plan.items() if k != "sha256"})
    validate_plan(plan)
    positions, digests = observation_domain(plan)
    assert positions == [1, 4] and set(digests) == {1, 4}
    run_reference(plan, tmp_path / "reference")
    report = private_json(tmp_path / "reference" / "boundaries" / "boundaries.json")
    assert report["positions"] == [1, 4]


@pytest.mark.parametrize("positions", [[], None, [False], [2, 1], [0, 0], [-1], [6]])
def test_empty_invalid_or_pending_observation_selection_is_rejected(tmp_path, positions):
    from qwen_r9700_lab.conformance_replay import validate_plan
    from qwen_r9700_lab.diagnostic_contract import seal

    plan = tiny_plan(tmp_path / "checkpoint")
    plan["observation_positions"] = positions
    plan = seal({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(DiagnosticError, match="observation positions"):
        validate_plan(plan)
