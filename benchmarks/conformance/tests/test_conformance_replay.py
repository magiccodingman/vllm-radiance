from copy import deepcopy

import numpy as np
import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference
from qwen_r9700_lab.conformance_reference import gdn_step
from qwen_r9700_lab.conformance_replay import replay_operator, run_reference, write_operator_capsule
from qwen_r9700_lab.conformance_session import compare_campaign
from qwen_r9700_lab.conformance_state import compare_frames
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, digest, private_json


def test_old_plan_cannot_claim_to_run_a_changed_reference(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_replay

    plan = tiny_plan(tmp_path / "checkpoint")
    monkeypatch.setattr(conformance_replay, "reference_code_identity", lambda: {"new": "code"})
    with pytest.raises(DiagnosticError, match="reference implementation changed"):
        conformance_replay.validate_plan(plan)


def test_forced_diagnostic_tokens_cannot_reach_the_publication_gate(tmp_path):
    from qwen_r9700_lab.conformance_session import CheckedSession

    plan = tiny_plan(tmp_path / "checkpoint")
    run_reference(plan, tmp_path / "reference")
    root = tmp_path / "reference/frame-000000"
    f = private_json(root / "frame.json")
    session = CheckedSession(
        tmp_path / "authority",
        contract=plan["contract"],
        required_components=f["coverage"],
        create=True,
    )
    try:
        with pytest.raises(DiagnosticError, match="forced diagnostic tokens"):
            session.commit(
                root,
                root,
                base_revision=0,
                reference_tokens=(f["pending"],),
                candidate_tokens=(f["pending"],),
                reference_stop=None,
                candidate_stop=None,
            )
        assert session.outputs_since(0) == []
    finally:
        session.close()


def model(plan):
    from pathlib import Path

    return QuantizedQwenReference(
        Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"]),
        kv_scales=plan["kv_scales"],
        contract=plan["contract"],
        execution=plan["execution"],
        adapter=plan["adapter"],
    )


def test_real_hybrid_prefill_replay_and_complete_schedule(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    a = run_reference(plan, tmp_path / "reference")
    b = run_reference(plan, tmp_path / "candidate")
    assert len(a["frames"]) == 4
    assert a["frames"][0]["phase"] == "prefill"
    assert a["frames"][-1]["consumed"] == 6
    assert a["frames"] == b["frames"]
    result = compare_campaign(
        tmp_path / "reference", tmp_path / "candidate", tmp_path / "comparison"
    )
    assert result["equal"] and result["formal_backend_equivalence"] == "UNPROVED"
    authenticate(result)


def test_snapshot_roundtrip_preserves_future_logits_and_every_layer(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    a, b = model(plan), model(plan)
    try:
        for token in plan["prefix"]:
            a.step(token)
        a.frame(tmp_path / "saved", phase="prefill")
        b.restore(tmp_path / "saved")
        for token in plan["forced_tokens"]:
            np.testing.assert_array_equal(a.step(token), b.step(token))
        a.frame(tmp_path / "a", phase="step")
        b.frame(tmp_path / "b", phase="step")
        assert compare_frames(tmp_path / "a", tmp_path / "b")["equal"]
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("width", range(8))
def test_every_d7_rejected_suffix_has_no_influence_on_restored_commit(tmp_path, width):
    plan = tiny_plan(tmp_path / "checkpoint")
    a, serial = model(plan), model(plan)
    try:
        for t in plan["prefix"]:
            a.step(t)
            serial.step(t)
        drafts = [3, 4, 5, 6, 7, 8, 9]
        # The incoming pending token is processed; a bonus emitted token is not.
        a.step(2)
        serial.step(2)
        for t in drafts[:width]:
            a.step(t)
            serial.step(t)
        a.frame(tmp_path / "accepted", phase="commit", pending=17)
        for t in drafts[width:]:
            a.step(t)
        a.restore(tmp_path / "accepted")
        expected = serial.step(17)
        actual = a.step(17)
        np.testing.assert_array_equal(actual, expected)
        a.frame(tmp_path / "a", phase="step")
        serial.frame(tmp_path / "b", phase="step")
        assert compare_frames(tmp_path / "a", tmp_path / "b")["equal"]
    finally:
        a.close()
        serial.close()


def test_captured_operator_same_inputs_find_extreme_decay_state_corruption(tmp_path):
    inputs = {
        "q": np.ones((1, 2), np.float32),
        "k": np.asarray([[1, 0]], np.float32),
        "v": np.asarray([[3, 4]], np.float32),
        "decay_log": np.asarray([-1000], np.float32),
        "beta": np.asarray([1], np.float32),
        "state": np.ones((1, 2, 2), np.float32),
    }
    out, state = gdn_step(**inputs)
    state[0, 0, 1] = 1  # latent state corruption, current output retained
    write_operator_capsule(
        tmp_path / "capsule",
        operator="gdn_step",
        inputs=inputs,
        outputs=[out, state],
        options={},
        contract=digest("c"),
        execution=digest("e"),
        adapter=digest("a"),
    )
    result = replay_operator(tmp_path / "capsule", tmp_path / "reference")
    assert not result["equal"]
    assert result["first_difference"]["boundary"] == "output.1"


def test_checkpoint_rejects_changed_artifact(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    (tmp_path / "checkpoint" / "config.json").write_text("{}")
    with pytest.raises(DiagnosticError, match="identity"):
        model(plan)


def test_no_pass_for_empty_missing_reordered_or_changed_schedule(tmp_path):
    from qwen_r9700_lab.diagnostic_contract import seal, write_private

    plan = tiny_plan(tmp_path / "checkpoint")
    run_reference(plan, tmp_path / "a")
    run_reference(plan, tmp_path / "b")
    path = tmp_path / "b" / "schedule.json"
    original = private_json(path)
    for index, frames in enumerate(
        ([], original["frames"][1:], list(reversed(original["frames"])))
    ):
        doc = deepcopy(original)
        doc.pop("sha256")
        doc["frames"] = frames
        path.unlink()
        write_private(path, seal(doc))
        with pytest.raises(DiagnosticError):
            compare_campaign(tmp_path / "a", tmp_path / "b", tmp_path / f"comparison{index}")
