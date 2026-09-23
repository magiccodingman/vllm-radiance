"""A controlled Inductor rounding experiment; original receipts remain immutable."""

from copy import deepcopy

from qwen_r9700_lab.conformance_execution_modes import admit_pair, compare_pair
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, seal

FLAG = "TORCHINDUCTOR_EMULATE_PRECISION_CASTS"
SETTING = "emulate_precision_casts"


def normalized_candidate(left, right):
    for side, expected in ((left, False), (right, True)):
        for key in ("measurement", "config", "runtime", "pass"):
            authenticate(side[key])
        runtime = side["runtime"]["runtime"]
        authenticate(runtime)
        require(runtime["flags"].get(FLAG) == str(int(expected)), "precision flag not explicit")
        observed = runtime.get("compiler_settings", {}).get(SETTING)
        require(observed is expected, "actual compiler precision setting not confirmed")
    require(
        right["measurement"]["execution_mode"] in {"compiled", "compiled-no-graphs"},
        "precision intervention must execute compiled candidate",
    )
    candidate = deepcopy(right)
    runtime = candidate["runtime"]["runtime"]
    runtime.pop("sha256")
    runtime["flags"][FLAG] = "0"
    runtime["compiler_settings"][SETTING] = False
    candidate["runtime"]["runtime"] = seal(runtime)
    candidate["runtime"].pop("sha256")
    candidate["runtime"] = seal(candidate["runtime"])
    return candidate


def admit_precision_intervention(left, right):
    checked = admit_pair(left, normalized_candidate(left, right))
    checked.pop("sha256")
    checked["schema"] = "qwen.precision-casts-intervention-admission.v1"
    checked["status"] = "ADMITTED_DECLARED_INTERVENTION"
    checked["normalized_receipts"] = checked.pop("receipts")
    checked["original_receipts"] = [
        [side[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
        for side in (left, right)
    ]
    checked["declared_change"] = {FLAG: ["0", "1"], SETTING: [False, True]}
    checked["scope"] = (
        "Metadata admission modulo one explicit, observed compiler rounding option. "
        "Generated kernels may change; this is not a mode-only comparison."
    )
    return seal(checked)


def compare_precision_intervention(left, right, left_rows, right_rows):
    checked = compare_pair(left, normalized_candidate(left, right), left_rows, right_rows)
    return seal(
        {
            "schema": "qwen.precision-casts-intervention-comparison.v1",
            "status": "COMPARED_DECLARED_INTERVENTION",
            "admission": admit_precision_intervention(left, right),
            "row_sources": checked["row_sources"],
            "decode": checked["decode"],
            "prefill": checked["prefill"],
            "scope": (
                "320 forced decode predictions and one prefill prediction. "
                "An observed compiler option changes; no universal or state-equivalence proof."
            ),
        }
    )
