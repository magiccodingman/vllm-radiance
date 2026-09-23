"""Check one numerical substitution from a common, immutable reference cut.

Native adapters must supply the captured reference inputs/state and implement
the stage and reference remainder. This checker grants no native qualification
merely because an adapter exists or a whole-model replay passed.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qwen_r9700_lab.conformance_topk import compare_rows, summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal


def validate_capture_bridge(reference, candidate, original_fixture, capture_fixture, capture):
    """Bind an instrumented prefix of a replay to the observed release outputs.

    This checks outputs and coverage, not hidden-state equality or an operator's
    isolated correctness. Those are separate obligations.
    """
    for record in (reference, candidate, original_fixture, capture_fixture, capture):
        authenticate(record)
    count = len(capture_fixture["output"]) - 1
    if (
        count < 1
        or capture_fixture["prefix"] != original_fixture["prefix"]
        or capture_fixture["output"] != original_fixture["output"][: count + 1]
        or reference["continuation"] != original_fixture["sha256"]
        or candidate["continuation"] != capture_fixture["sha256"]
    ):
        raise DiagnosticError("capture and release do not consume the same saved prefix")
    if len(candidate["rows"]) != count or len(reference["rows"]) < count:
        raise DiagnosticError("capture bridge has an incomplete position domain")
    start = len(capture_fixture["prefix"])
    expected = list(range(start, start + count))
    observed = [r["absolute_position"] for r in candidate["rows"]]
    captured = [p for b in capture["batches"] for p in b["positions"]]
    if observed != expected or captured != expected or capture["positions"] != count:
        raise DiagnosticError("capture bridge position coverage changed")
    if not any(k.startswith("inductor/") and v > 0 for k, v in capture["counts"].items()):
        raise DiagnosticError("no actual compiled launch was captured")
    if candidate["prefill"] != reference["prefill"]:
        raise DiagnosticError("instrumented prefill differs from the release")
    if candidate["rows"] != reference["rows"][:count]:
        raise DiagnosticError("instrumented decode differs from the release")
    for left, right in zip(reference["rows"][:count], candidate["rows"], strict=True):
        comparison = compare_rows(left["logits"], right["logits"])
        if not comparison["full_logits_exact"]:
            raise DiagnosticError("captured full-vocabulary digest differs")
    return seal(
        {
            "schema": "qwen.compiled-capture-output-bridge.v1",
            "status": "MATCHED",
            "positions": count,
            "full_logits_exact": count,
            "prefill_exact": True,
            "same_saved_inputs": True,
            "release_rows_sha256": reference["sha256"],
            "captured_rows_sha256": candidate["sha256"],
            "capture_manifest_sha256": capture["sha256"],
            "reference_fixture_sha256": original_fixture["sha256"],
            "capture_fixture_sha256": capture_fixture["sha256"],
            "scope": (
                "Observed compiled outputs and position coverage only; "
                "not isolated-stage or hidden-state qualification"
            ),
        }
    )


@dataclass
class Cut:
    position: int
    layer: int | None
    stage: str
    inputs: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    reference_output: dict[str, np.ndarray]
    reference_next_state: dict[str, np.ndarray]
    reference_logits: np.ndarray


class Adapter(Protocol):
    def stage(self, arm, inputs, state):
        """Return (output arrays, next state arrays), without modifying inputs."""

    def remainder(self, output, next_state):
        """Evaluate the pinned reference remainder from an isolated state copy."""


def clone(arrays):
    return {k: np.array(v, copy=True, order="K") for k, v in arrays.items()}


def exact(a, b):
    if a.keys() != b.keys():
        return False
    return all(
        a[k].dtype == b[k].dtype
        and a[k].shape == b[k].shape
        and a[k].tobytes(order="C") == b[k].tobytes(order="C")
        for k in a
    )


def logits_exact(a, b):
    return exact({"logits": a}, {"logits": b})


def evaluate(cut: Cut, adapter: Adapter, *, arms=("old", "fixed")):
    if not cut.inputs or not cut.reference_output:
        raise DiagnosticError("an isolated comparison needs observed inputs and outputs")
    logits = cut.reference_logits
    if (
        logits.ndim != 1
        or logits.size < 21
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise DiagnosticError("the isolated comparison requires full finite vocabulary logits")
    # This is a mandatory check of the remainder, not an assumption that any
    # function called 'reference' implements the reference calculation.
    replayed = adapter.remainder(clone(cut.reference_output), clone(cut.reference_next_state))
    if not logits_exact(logits, replayed):
        raise DiagnosticError(
            "reference remainder does not reproduce the admitted reference logits"
        )
    reference_summary = summarize_logits(logits)
    results = {}
    for arm in arms:
        inputs, initial = clone(cut.inputs), clone(cut.state)
        output, state = adapter.stage(arm, inputs, initial)
        if not exact(inputs, cut.inputs):
            raise DiagnosticError("candidate modified an input outside the declared state")
        # State changes are never hidden behind an output-only comparison.
        output_equal = exact(output, cut.reference_output)
        state_equal = exact(state, cut.reference_next_state)
        if output_equal and state_equal:
            observed = logits
            propagation = "identical output and state imply identical pinned reference remainder"
        else:
            observed = adapter.remainder(clone(output), clone(state))
            propagation = "reference remainder evaluated on this isolated substitution"
        if observed.shape != logits.shape or not np.isfinite(observed).all():
            raise DiagnosticError(
                "candidate remainder did not produce full finite vocabulary logits"
            )
        results[arm] = {
            "position": cut.position,
            "layer": cut.layer,
            "stage": cut.stage,
            "input_exact": True,
            "reference_remainder_verified": True,
            "stage_output_exact": output_equal,
            "stage_state_exact": state_equal,
            "full_logits_exact": logits_exact(logits, observed),
            "topk": compare_rows(reference_summary, summarize_logits(observed)),
            "propagation": propagation,
        }
    return results


def aggregate_stage(records, positions, layers):
    """Count a position only if every declared layer instance meets the metric."""
    positions, layers = tuple(positions), tuple(layers)
    if len(positions) != 320 or len(set(positions)) != 320 or not layers:
        raise DiagnosticError("isolated stage publication requires 320 unique positions")
    if len(set(layers)) != len(layers):
        raise DiagnosticError("duplicate declared layer instance")
    expected = {(p, layer) for p in positions for layer in layers}
    actual = {(r["position"], r["layer"]) for r in records}
    if actual != expected or len(records) != len(expected):
        raise DiagnosticError(
            "isolated stage coverage is missing, duplicated or outside its domain"
        )
    if not all(r["input_exact"] and r["reference_remainder_verified"] for r in records):
        raise DiagnosticError("the stage comparisons do not share verified reference inputs")
    by_position = {p: [r for r in records if r["position"] == p] for p in positions}
    return {
        "positions": 320,
        "layer_instances": list(layers),
        "evaluations": len(records),
        "isolated_inputs_verified": True,
        "reference_remainder_verified": True,
        "criterion": "a position passes only when every declared layer instance passes",
        "stage_output_exact": sum(
            all(r["stage_output_exact"] for r in rs) for rs in by_position.values()
        ),
        "stage_state_exact": sum(
            all(r["stage_state_exact"] for r in rs) for rs in by_position.values()
        ),
        "top20_set_exact": sum(
            all(r["topk"]["20"]["set_exact"] for r in rs) for rs in by_position.values()
        ),
        "top20_order_exact": sum(
            all(r["topk"]["20"]["ranked_exact"] for r in rs) for rs in by_position.values()
        ),
    }
