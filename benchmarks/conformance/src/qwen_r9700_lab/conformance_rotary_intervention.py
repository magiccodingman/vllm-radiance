"""Admit only the declared native RoPE rounding change in a saved eager replay."""

from copy import deepcopy

from qwen_r9700_lab.conformance_execution_modes import admit_pair, compare_pair
from qwen_r9700_lab.conformance_precision_intervention import (
    admit_precision_intervention,
    compare_precision_intervention,
)
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def normalized_rotary(reference, candidate, binding):
    authenticate(binding)
    for side in (reference, candidate):
        for key in ("measurement", "config", "runtime", "pass"):
            authenticate(side[key])
        require(side["measurement"]["execution_mode"] == "eager", "rotary arm must be eager")
    require(
        reference["measurement"]["driver_sha256"] == binding["base_driver"], "base driver changed"
    )
    require(
        candidate["measurement"]["driver_sha256"] == binding["experiment_driver"],
        "rotary experiment driver changed",
    )
    capture = candidate["measurement"]["isolated_capture"]
    require(reference["measurement"]["isolated_capture"] == capture, "capture controls differ")
    require(
        reference["config"]["worker_cls"]
        == "execution_mode_d7_worker.ExecutionMode" + ("CaptureWorker" if capture else "Worker"),
        "unexpected reference worker",
    )
    require(
        candidate["config"]["worker_cls"]
        == "rotary_mode_d7_worker.RotaryRne" + ("CaptureWorker" if capture else "Worker"),
        "unexpected rotary worker",
    )
    before = candidate["runtime"]["rotary_intervention"]
    after = candidate["pass"]["observation"]["rotary_intervention"]
    for observation in (before, after):
        identity = observation["identity"]
        authenticate(identity)
        require(identity["installed_before_load"] is True, "late rotary installation")
        for key in ("native_source", "patched_source", "worker_source", "patcher_source"):
            require(identity[key] == binding[key], f"rotary intervention changes {key}")
    require(before["identity"] == after["identity"], "rotary identity changed during replay")
    require(after["calls"] > before["calls"] >= 0, "modified rotary did not execute during replay")
    normalized = deepcopy(candidate)
    normalized["measurement"].pop("sha256")
    normalized["measurement"]["driver_sha256"] = binding["base_driver"]
    normalized["measurement"] = seal(normalized["measurement"])
    # All other config/runtime/source checks still run in ordinary admission.
    admit_pair(reference, normalized)
    return normalized


def compare_rotary_intervention(reference, candidate, reference_rows, candidate_rows, binding):
    normalized = normalized_rotary(reference, candidate, binding)
    result = compare_pair(reference, normalized, reference_rows, candidate_rows)
    return seal(
        {
            "schema": "qwen.rotary-intervention-comparison.v1",
            "status": "COMPARED_DECLARED_INTERVENTION",
            "binding": binding,
            "normalized_comparison": result,
            "original_receipts": [
                [s[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for s in (reference, candidate)
            ],
            "observed_rotary": candidate["pass"]["observation"]["rotary_intervention"],
            "decode": result["decode"],
            "prefill": result["prefill"],
            "scope": (
                "Same eager replay, one native product-rounding intervention; sampled outputs only."
            ),
        }
    )


def compare_common_rounding(reference, candidate, compiled, candidate_rows, compiled_rows, binding):
    normalized = normalized_rotary(reference, candidate, binding)
    result = compare_precision_intervention(normalized, compiled, candidate_rows, compiled_rows)
    return seal(
        {
            "schema": "qwen.rotary-and-casts-comparison.v1",
            "status": "COMPARED_TWO_DECLARED_INTERVENTIONS",
            "binding": binding,
            "normalized_comparison": result,
            "original_receipts": [
                [s[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for s in (reference, candidate, compiled)
            ],
            "observed_rotary": candidate["pass"]["observation"]["rotary_intervention"],
            "decode": result["decode"],
            "prefill": result["prefill"],
            "scope": (
                "Eager native RoPE products changed to nearest-even; compiled precision-cast "
                "emulation enabled. Other metadata admitted unchanged; 320 sampled predictions."
            ),
        }
    )


def admit_common_rounding(reference, candidate, compiled, binding):
    normalized = normalized_rotary(reference, candidate, binding)
    admission = admit_precision_intervention(normalized, compiled)
    return seal(
        {
            "schema": "qwen.rotary-and-casts-admission.v1",
            "status": "ADMITTED_TWO_DECLARED_INTERVENTIONS",
            "binding": binding,
            "normalized_admission": admission,
            "original_receipts": [
                [s[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for s in (reference, candidate, compiled)
            ],
            "captures": admission["captures"],
            "scope": "RNE RoPE and compiler cast preservation; other controls checked.",
        }
    )
