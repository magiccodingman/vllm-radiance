from copy import deepcopy

import pytest
from test_conformance_execution_modes import reseal
from test_d7_precision_comparison import experiment as precision_experiment

from qwen_r9700_lab.conformance_rotary_intervention import (
    admit_common_rounding,
    compare_common_rounding,
    compare_rotary_intervention,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def experiment():
    reference, compiled, rows = precision_experiment()
    reference["config"]["worker_cls"] = "execution_mode_d7_worker.ExecutionModeWorker"
    reference["config"] = reseal(reference["config"])
    candidate = deepcopy(reference)
    identity = seal(
        {
            "native_source": "1" * 64,
            "patched_source": "2" * 64,
            "worker_source": "3" * 64,
            "patcher_source": "4" * 64,
            "installed_before_load": True,
        }
    )
    binding = seal(
        {
            **{
                k: identity[k]
                for k in ("native_source", "patched_source", "worker_source", "patcher_source")
            },
            "base_driver": "d" * 64,
            "experiment_driver": "e" * 64,
        }
    )
    candidate["measurement"]["driver_sha256"] = "e" * 64
    candidate["measurement"] = reseal(candidate["measurement"])
    candidate["config"]["worker_cls"] = "rotary_mode_d7_worker.RotaryRneWorker"
    candidate["config"] = reseal(candidate["config"])
    candidate["runtime"]["rotary_intervention"] = {"identity": identity, "calls": 8}
    candidate["runtime"] = reseal(candidate["runtime"])
    candidate["pass"]["observation"]["rotary_intervention"] = {"identity": identity, "calls": 100}
    candidate["pass"] = reseal(candidate["pass"])
    return reference, candidate, compiled, rows, binding


def test_admits_bound_kernel_change_and_keeps_both_interventions_explicit():
    a, b, c, rows, binding = experiment()
    saved = deepcopy((a, b, c))
    change = compare_rotary_intervention(a, b, rows, rows, binding)
    assert change["decode"]["full_logits_exact"] == 320
    combined = compare_common_rounding(a, b, c, rows, rows, binding)
    assert combined["decode"]["full_logits_exact"] == 320
    assert combined["prefill"]["full_logits_exact"] == 1
    assert combined["status"] == "COMPARED_TWO_DECLARED_INTERVENTIONS"
    assert combined["original_receipts"][1][0] == b["measurement"]["sha256"]
    assert saved == (a, b, c)
    admission = admit_common_rounding(a, b, c, binding)
    assert admission["status"] == "ADMITTED_TWO_DECLARED_INTERVENTIONS"
    assert admission["original_receipts"][1][0] == b["measurement"]["sha256"]


def test_rejects_an_unreviewed_reference_worker():
    a, b, c, rows, binding = experiment()
    a["config"]["worker_cls"] = "another.Worker"
    a["config"] = reseal(a["config"])
    with pytest.raises(DiagnosticError, match="reference worker"):
        compare_common_rounding(a, b, c, rows, rows, binding)


@pytest.mark.parametrize(
    "fault", ["no_calls", "wrong_kernel", "late", "driver", "capacity", "worker"]
)
def test_rejects_unobserved_or_unrelated_changes(fault):
    a, b, c, rows, binding = experiment()
    if fault in {"wrong_kernel", "late"}:
        for record in (
            b["runtime"]["rotary_intervention"],
            b["pass"]["observation"]["rotary_intervention"],
        ):
            identity = deepcopy(record["identity"])
            identity["patched_source" if fault == "wrong_kernel" else "installed_before_load"] = (
                "f" * 64 if fault == "wrong_kernel" else False
            )
            record["identity"] = reseal(identity)
    elif fault == "no_calls":
        b["pass"]["observation"]["rotary_intervention"]["calls"] = 8
    elif fault == "driver":
        b["measurement"]["driver_sha256"] = "f" * 64
    elif fault == "capacity":
        b["config"]["max_num_seqs"] = 1
    elif fault == "worker":
        b["config"]["worker_cls"] = "an.unreviewed.Worker"
    for key in ("runtime", "pass", "measurement", "config"):
        b[key] = reseal(b[key])
    with pytest.raises(DiagnosticError):
        compare_common_rounding(a, b, c, rows, rows, binding)
