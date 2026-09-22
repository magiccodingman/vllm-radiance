"""Reject incomplete, mislabelled and unchecked native stage evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

SOURCE = (
    Path(__file__).resolve().parents[1] / "analyze_native_d7_stages.py"
)
SPEC = importlib.util.spec_from_file_location("native_stage_audit", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def group(stage="Full BF16 target head", instances=("head",)):
    prediction = {
        str(k): {"set_exact": True, "ranked_exact": True, "overlap": k} for k in (1, 10, 20)
    }
    prediction["full_logits_exact"] = True
    variants = {
        name: {
            "local_output_exact": True,
            "local_state_exact": True,
            "suffix": "validated reference reuse",
        }
        for pair in audit.PAIRS.values()
        for name in pair
    }
    return seal(
        {
            "status": "REFERENCE_REPLAY_CHECKED",
            "positions": 8,
            "cache_regions": 64,
            "reference_full_logits_exact": True,
            "injected_output_fault_detected": True,
            "injected_prefix_fault_detected": True,
            "authoritative_cache_restored": True,
            "stages": [
                {
                    "stage": stage,
                    "instance": instance,
                    "variants": copy.deepcopy(variants),
                    "comparisons": {
                        column: [copy.deepcopy(prediction) for _ in range(8)]
                        for column in audit.PAIRS
                    },
                }
                for instance in instances
            ],
        }
    )


def test_all_layer_instances_must_pass_the_position():
    sample = group("MLP gate/up projection", tuple(map(str, range(64))))
    sample["stages"][17]["comparisons"]["old"][2]["20"].update(
        set_exact=False, ranked_exact=False, overlap=19
    )
    sample["stages"][17]["comparisons"]["old"][2]["full_logits_exact"] = False
    sample = seal({k: v for k, v in sample.items() if k != "sha256"})
    result = audit.aggregate_groups([sample])
    assert result["stages"]["MLP gate/up projection"]["old"]["top20_set_exact"] == 7
    assert result["stages"]["MLP gate/up projection"]["fixed"]["top20_set_exact"] == 8


@pytest.mark.parametrize(
    "fault",
    [
        "coverage",
        "duplicate",
        "unchecked",
        "state",
        "prefix",
        "column",
        "ordering",
        "missing_variant",
        "group_drift",
    ],
)
def test_broken_evidence_is_rejected_even_if_resealed(fault):
    sample = group()
    groups = [sample]
    if fault == "coverage":
        sample = group("MLP gate/up projection", ("0",))
        groups = [sample]
    elif fault == "duplicate":
        sample["stages"].append(copy.deepcopy(sample["stages"][0]))
    elif fault == "unchecked":
        sample["stages"][0]["variants"]["old_m8"]["local_output_exact"] = False
    elif fault == "state":
        sample["authoritative_cache_restored"] = False
    elif fault == "prefix":
        sample["injected_prefix_fault_detected"] = False
    elif fault == "column":
        sample["stages"][0]["comparisons"]["fixed"] = []
    elif fault == "ordering":
        sample["stages"][0]["comparisons"]["old"][0]["20"]["set_exact"] = False
    elif fault == "missing_variant":
        del sample["stages"][0]["variants"]["old_m8"]
    elif fault == "group_drift":
        groups.append(group("Final normalization/layout", ("final",)))
    groups[0] = seal({k: v for k, v in sample.items() if k != "sha256"})
    with pytest.raises(DiagnosticError):
        audit.aggregate_groups(groups)


def test_unsealed_mutation_is_rejected():
    sample = group()
    sample["stages"][0]["comparisons"]["old"][0]["full_logits_exact"] = False
    with pytest.raises(DiagnosticError):
        audit.aggregate_groups([sample])
