from dataclasses import replace

import numpy as np
import pytest

from qwen_r9700_lab.conformance_stage_isolation import Cut, aggregate_stage, evaluate
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def example():
    values = np.arange(64, dtype=np.float32)
    return Cut(
        7,
        3,
        "test-stage",
        {"x": values.copy()},
        {"memory": np.zeros(1, np.float32)},
        {"y": values.copy()},
        {"memory": np.zeros(1, np.float32)},
        values.copy(),
    )


class SmallModel:
    def __init__(self, defect):
        self.defect = defect
        self.remainder_calls = 0

    def stage(self, arm, inputs, state):
        out = inputs["x"].copy()
        if arm == "old":
            if self.defect == "small_numeric":
                out[0] = 0.25
            elif self.defect == "state_only":
                state["memory"][0] = 1
            elif self.defect == "input_write":
                inputs["x"][0] = 100
        return {"y": out}, state

    def remainder(self, output, next_state):
        self.remainder_calls += 1
        logits = output["y"].copy()
        logits[0] += next_state["memory"][0] * 100
        return logits


def test_local_numeric_difference_is_not_automatically_a_top20_change():
    adapter = SmallModel("small_numeric")
    results = evaluate(example(), adapter)
    assert not results["old"]["stage_output_exact"]
    assert results["old"]["topk"]["20"]["set_exact"]
    assert results["old"]["topk"]["20"]["ranked_exact"]
    assert results["fixed"]["stage_output_exact"]
    # Reference check + genuinely different candidate; exact output/state
    # takes the explicit identity implication rather than an invented replay.
    assert adapter.remainder_calls == 2


def test_equal_output_cannot_hide_a_corrupt_state():
    cut = example()
    results = evaluate(cut, SmallModel("state_only"))
    assert results["old"]["stage_output_exact"]
    assert not results["old"]["stage_state_exact"]
    assert not results["old"]["topk"]["20"]["set_exact"]
    assert results["fixed"]["full_logits_exact"]
    assert cut.state["memory"][0] == 0


def test_input_mutation_is_rejected_and_reference_data_stays_unchanged():
    cut = example()
    with pytest.raises(DiagnosticError, match="modified an input"):
        evaluate(cut, SmallModel("input_write"))
    assert cut.inputs["x"][0] == 0


def test_remainder_has_to_match_independently_saved_reference():
    cut = example()
    incorrect = cut.reference_logits.copy()
    incorrect[0] += 0.125
    with pytest.raises(DiagnosticError, match="does not reproduce"):
        evaluate(replace(cut, reference_logits=incorrect), SmallModel("small_numeric"))


def test_no_missing_duplicate_or_cherry_picked_stage_instances():
    result = evaluate(example(), SmallModel("state_only"))["fixed"]
    records = [{**result, "position": p, "layer": layer} for p in range(320) for layer in (3, 7)]
    summary = aggregate_stage(records, range(320), (3, 7))
    assert summary["top20_set_exact"] == 320
    assert summary["evaluations"] == 640
    for broken in (records[:-1], [*records, records[0]]):
        with pytest.raises(DiagnosticError, match="coverage"):
            aggregate_stage(broken, range(320), (3, 7))
    failed = {**records[0], "topk": {"20": {"set_exact": False, "ranked_exact": False}}}
    summary = aggregate_stage([failed, *records[1:]], range(320), (3, 7))
    assert summary["top20_set_exact"] == summary["top20_order_exact"] == 319


def bridge_records():
    from qwen_r9700_lab.conformance_topk import summarize_logits
    from qwen_r9700_lab.diagnostic_contract import seal

    fixture = seal({"prefix": [1, 2], "output": [3, 4, 5, 6]})
    short = seal({"prefix": [1, 2], "output": [3, 4, 5]})
    rows = [
        {
            "absolute_position": p,
            "position": p - 2,
            "logits": summarize_logits(np.arange(32, dtype=np.float32) + p),
        }
        for p in range(2, 5)
    ]
    ref = seal({"continuation": fixture["sha256"], "rows": rows, "prefill": {"digest": "p"}})
    captured = seal({"continuation": short["sha256"], "rows": rows[:2], "prefill": ref["prefill"]})
    capture = seal(
        {
            "positions": 2,
            "batches": [{"positions": [2, 3]}],
            "counts": {"inductor/actual_kernel": 1},
        }
    )
    return [ref, captured, fixture, short, capture]


def test_native_output_bridge_admits_only_the_matching_saved_prefix():
    from qwen_r9700_lab.conformance_stage_isolation import validate_capture_bridge

    result = validate_capture_bridge(*bridge_records())
    assert result["positions"] == result["full_logits_exact"] == 2
    assert "not isolated-stage" in result["scope"]


@pytest.mark.parametrize("fault", ["input", "decode", "prefill", "coverage", "compiled"])
def test_native_output_bridge_rejects_mismatches_and_missing_observations(fault):
    from qwen_r9700_lab.conformance_stage_isolation import validate_capture_bridge
    from qwen_r9700_lab.diagnostic_contract import seal

    records = bridge_records()
    if fault == "input":
        records[3]["prefix"][0] = 99
    elif fault == "decode":
        records[1]["rows"][0] = {**records[1]["rows"][0], "logits": {"digest": "changed"}}
    elif fault == "prefill":
        records[1]["prefill"] = {"digest": "changed"}
    elif fault == "coverage":
        records[4]["batches"][0]["positions"] = [3, 2]
    else:
        records[4]["counts"] = {}
    records = [seal({k: v for k, v in r.items() if k != "sha256"}) for r in records]
    with pytest.raises(DiagnosticError):
        validate_capture_bridge(*records)
