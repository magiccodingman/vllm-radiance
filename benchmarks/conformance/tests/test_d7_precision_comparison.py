from copy import deepcopy

import pytest
from test_conformance_execution_modes import reseal, saved_rows, side

from qwen_r9700_lab.conformance_execution_modes import compare_pair
from qwen_r9700_lab.conformance_precision_intervention import FLAG, compare_precision_intervention
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def experiment():
    rows = saved_rows()
    left, right = side("eager", rows), side("compiled", rows)
    for record, value in ((left, False), (right, True)):
        rt = record["runtime"]["runtime"]
        rt["flags"][FLAG] = str(int(value))
        rt["compiler_settings"] = {"emulate_precision_casts": value}
        record["runtime"]["runtime"] = reseal(rt)
        record["runtime"] = reseal(record["runtime"])
    return left, right, rows


def test_explicit_intervention_preserves_receipts_and_separate_prefill():
    left, right, rows = experiment()
    original = deepcopy((left, right))
    result = compare_precision_intervention(left, right, rows, rows)
    assert result["decode"]["full_logits_exact"] == 320
    assert result["prefill"]["full_logits_exact"] == 1
    admission = result["admission"]
    assert admission["original_receipts"][1][2] == right["runtime"]["sha256"]
    assert admission["normalized_receipts"][1][2] != right["runtime"]["sha256"]
    assert original == (left, right)
    with pytest.raises(DiagnosticError):
        compare_pair(left, right, rows, rows)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "ignored_flag",
        "missing_flag",
        "other_setting",
        "other_flag",
        "custom_op",
        "driver",
    ],
)
def test_does_not_admit_unknown_precision_or_other_changes(fault):
    left, right, rows = experiment()
    rt = right["runtime"]["runtime"]
    if fault == "unknown":
        rt["compiler_settings"]["emulate_precision_casts"] = None
    elif fault == "ignored_flag":
        rt["compiler_settings"]["emulate_precision_casts"] = False
    elif fault == "missing_flag":
        rt["flags"].pop(FLAG)
    elif fault == "other_setting":
        rt["compiler_settings"]["arbitrary_future_setting"] = True
    elif fault == "other_flag":
        rt["flags"]["RADIANCE_FP8_STREAM"] = "1"
    elif fault == "custom_op":
        right["config"]["compilation_config"]["custom_ops"] = ["none", "+silu_and_mul"]
        right["config"] = reseal(right["config"])
    elif fault == "driver":
        right["measurement"]["driver_sha256"] = "0" * 64
        right["measurement"] = reseal(right["measurement"])
    right["runtime"]["runtime"] = reseal(rt)
    right["runtime"] = reseal(right["runtime"])
    with pytest.raises(DiagnosticError):
        compare_precision_intervention(left, right, rows, rows)


def test_mode_only_comparer_also_rejects_changed_observed_setting_with_same_env():
    left, right, rows = experiment()
    rt = right["runtime"]["runtime"]
    rt["flags"][FLAG] = "0"
    right["runtime"]["runtime"] = reseal(rt)
    right["runtime"] = reseal(right["runtime"])
    with pytest.raises(DiagnosticError, match="observed compiler settings"):
        compare_pair(left, right, rows, rows)
