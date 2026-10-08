import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import (
    ReplaySchedule,
    aggregate,
    choose_continuations,
    compare_measurements,
    compare_rows,
    compare_saved_rows,
    summarize_logits,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def row(values):
    return summarize_logits(np.asarray(values, dtype=np.float32))


def test_exact_self_comparison_covers_all_measures():
    measured = row(np.arange(64))
    result = aggregate([compare_rows(measured, measured)])
    assert result["full_logits_exact"] == 1
    for k in (1, 10, 20):
        assert result[str(k)]["set_exact_percent"] == 100
        assert result[str(k)]["ranked_exact_percent"] == 100
        assert result[str(k)]["mean_overlap_tokens"] == k
        assert result[str(k)]["retained_scores_exact"] == 1


def test_rank_change_is_distinct_from_membership_change():
    values = np.arange(64, dtype=np.float32)
    changed = values.copy()
    changed[61], changed[62] = changed[62], changed[61]
    result = compare_rows(row(values), row(changed))
    assert result["1"]["ranked_exact"]
    assert result["10"]["set_exact"] and not result["10"]["ranked_exact"]
    assert result["20"]["set_exact"] and not result["20"]["ranked_exact"]


def test_bad_top1_and_missing_twentieth_token_are_detected_independently():
    values = np.arange(64, dtype=np.float32)
    changed = values.copy()
    changed[0] = 100
    result = compare_rows(row(values), row(changed))
    assert not result["1"]["set_exact"]
    assert result["10"]["overlap"] == 9
    assert result["20"]["overlap"] == 19
    assert not result["full_logits_exact"]


def test_ties_are_retained_and_have_a_declared_deterministic_order():
    values = np.zeros(64, dtype=np.float32)
    values[5:15] = 3
    values[16:40] = 2
    measured = row(values)
    assert measured["ids"][:10] == list(range(5, 15))
    assert measured["ids"][10:] == list(range(16, 40))
    assert measured["boundary_ties"] == {"1": 10, "10": 10, "20": 24}
    result = compare_rows(measured, measured)
    assert result["20"]["inclusive_tie_set_exact"]
    assert result["20"]["reference_boundary_tied"]


def test_numerical_changes_are_not_hidden_by_equal_ranking():
    reference = row(np.arange(64))
    candidate = row(np.arange(64) + 0.125)
    result = compare_rows(reference, candidate)
    assert result["20"]["ranked_exact"]
    assert not result["20"]["retained_scores_exact"]
    assert not result["full_logits_exact"]


@pytest.mark.parametrize(
    "bad", [np.full(32, np.nan), np.full(32, np.inf), np.zeros(20), np.zeros((1, 32))]
)
def test_invalid_vocabulary_evidence_is_rejected(bad):
    with pytest.raises(DiagnosticError):
        row(bad)


def records(counts):
    return [
        {"mode": "full", "finish_reason": "stop", "trial": i, "output_tokens": n}
        for i, n in enumerate(counts)
    ]


def test_budget_counts_decode_positions_not_input_or_prefill_or_rejected_rows():
    selected = choose_continuations(records([11, 18, 100]), 32)
    assert [r["evaluate_positions"] for r in selected] == [8, 16, 8]
    assert sum(r["evaluate_positions"] for r in selected) == 32


def test_insufficient_natural_output_and_non_m8_budgets_are_rejected():
    for data, target in [(records([8, 8]), 16), (records([100]), 17)]:
        with pytest.raises(DiagnosticError):
            choose_continuations(data, target)


@pytest.mark.parametrize("field,value", [("mode", "global256"), ("finish_reason", "length")])
def test_wrong_source_generation_is_not_silently_admitted(field, value):
    data = records([100])
    data[0][field] = value
    with pytest.raises(DiagnosticError):
        choose_continuations(data, 16)


def run_schedule(speculation):
    prefix, output = [101, 102, 103], list(range(200, 217))
    schedule = ReplaySchedule(prefix, output, speculation=speculation)
    schedule.check_inputs([0, 1], prefix[:2])
    schedule.check_inputs([2], prefix[2:])
    initial = schedule.commit(0)
    assert initial["prefill"] and initial["tokens"] == [200]
    positions, predictions, emitted = [], [], initial["tokens"][:]
    while not schedule.done:
        start = schedule.cursor
        width = 8 if speculation else 1
        pending_and_drafts = [output[start]] + (schedule.proposals() if speculation else [])
        observed_positions = list(range(len(prefix) + start, len(prefix) + start + width))
        schedule.check_inputs(observed_positions, pending_and_drafts)
        step = schedule.commit(width - 1)
        positions.extend(observed_positions[: step["count"]])
        predictions.extend(range(step["start"], step["start"] + step["count"]))
        emitted.extend(step["tokens"])
        assert step["reject"] == 0
    assert emitted == output
    with pytest.raises(DiagnosticError):
        schedule.commit(width - 1)
    return positions, predictions


def test_m1_and_m8_have_identical_consumed_and_pending_token_conventions():
    assert run_schedule(False) == run_schedule(True) == (list(range(3, 19)), list(range(16)))


@pytest.mark.parametrize("fault", ["token", "position", "width"])
def test_wrong_inputs_positions_or_execution_width_fail_before_comparison(fault):
    schedule = ReplaySchedule([1, 2], list(range(10, 27)), speculation=True)
    schedule.check_inputs([0, 1], [1, 2])
    schedule.commit(0)
    positions, inputs = list(range(2, 10)), list(range(10, 18))
    if fault == "token":
        inputs[3] += 1
    elif fault == "position":
        positions = list(range(3, 11))
        inputs = list(range(11, 19))
    else:
        positions, inputs = positions[:1], inputs[:1]
    with pytest.raises(DiagnosticError):
        schedule.check_inputs(positions, inputs)


def test_empty_comparisons_cannot_report_success():
    with pytest.raises(DiagnosticError):
        aggregate([])


def test_prefill_cannot_skip_repeat_or_emit_before_finishing():
    schedule = ReplaySchedule([1, 2, 3], [4, 5], speculation=False)
    with pytest.raises(DiagnosticError):
        schedule.check_inputs([1], [2])
    schedule.check_inputs([0], [1])
    with pytest.raises(DiagnosticError):
        schedule.check_inputs([0], [1])
    with pytest.raises(DiagnosticError):
        schedule.commit(0)
    schedule.check_inputs([1, 2], [2, 3])
    assert schedule.commit(0)["tokens"] == [4]


def measured(label, **changes):
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-summary:v1",
            "status": "MEASURED",
            "revision": label,
            "corpus": "a" * 64,
            "scope": "same",
            "ordering": "same",
            "metrics": {
                "positions": 10000,
                **{str(k): {"set_exact_percent": 98.0} for k in (1, 10, 20)},
            },
            **changes,
        }
    )


def test_before_after_keeps_both_evidence_identities_and_exact_denominator():
    before, after = measured("baseline"), measured("repaired")
    result = compare_measurements(before, after)
    assert result["positions"] == 10000
    assert result["before"] == before["sha256"] and result["after"] == after["sha256"]
    assert result["top_k"]["20"]["set_agreement_percentage_point_change"] == 0


def saved_rows(*, changed=False, width=1, **changes):
    values = np.arange(64, dtype=np.float32)
    modified = values.copy()
    if changed:
        modified[0] = 100
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-private-rows:v1",
            "continuation": "a" * 64,
            "prefill": row(values),
            "rows": [
                {
                    "position": i,
                    "absolute_position": 60000 + i,
                    "target_rows": width,
                    "logits": row(modified if i == 1 else values),
                }
                for i in range(8)
            ],
            **changes,
        }
    )


@pytest.mark.parametrize("width", [1, 8])
def test_reference_drift_is_measured_even_when_prefill_is_unchanged(width):
    rows, prefill = compare_saved_rows(
        saved_rows(width=width), saved_rows(width=width, changed=True), target_rows=width
    )
    assert prefill["full_logits_exact"]
    result = aggregate(rows)
    assert result["1"]["set_exact"] == 7
    assert result["20"]["mean_overlap_tokens"] == 19.875


@pytest.mark.parametrize("fault", ["history", "position", "length", "width", "seal"])
def test_reference_drift_rejects_incomparable_saved_evidence(fault):
    before, after = saved_rows(), saved_rows()
    after.pop("sha256")
    if fault == "history":
        after["continuation"] = "b" * 64
    elif fault == "position":
        after["rows"][2]["absolute_position"] += 1
    elif fault == "length":
        after["rows"].pop()
    elif fault == "width":
        after["rows"][2]["target_rows"] = 8
    after = seal(after)
    if fault == "seal":
        after["continuation"] = "b" * 64
    with pytest.raises(DiagnosticError):
        compare_saved_rows(before, after, target_rows=1)


@pytest.mark.parametrize(
    "change",
    [
        {"corpus": "b" * 64},
        {"status": "RUNNING"},
        {"revision": "baseline"},
        {"metrics": {"positions": 16}},
        {"ordering": "different"},
    ],
)
def test_a_pilot_partial_run_or_changed_workload_cannot_be_claimed_as_after(change):
    with pytest.raises(DiagnosticError):
        compare_measurements(measured("baseline"), measured("repaired", **change))
