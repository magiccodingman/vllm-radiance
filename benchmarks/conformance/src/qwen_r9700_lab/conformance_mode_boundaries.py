"""Align observed semantic boundaries without claiming complete state coverage.

Event numbers and physical addresses are not semantic identities. Compare only
unique named operations with the same logical owner and ordered positions.
Tensor equality here is not vocabulary top-k or proof of a complete transition.
"""

from collections import defaultdict

import numpy as np

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal


def silu_cut(metadata, tensors, layer):
    """Identify the MLP activation by both adjacent named projections and wiring."""
    authenticate(metadata)
    stem = f"language_model.model.layers.{layer}.mlp."

    def projection(suffix):
        matches = [
            event
            for event in metadata["events"]
            if event["operation"] == "radiance.mxfp4_linear.default"
            and any(
                owner.startswith(stem + suffix + ".")
                for owner in event.get("logical_identities", ())
            )
        ]
        if len(matches) != 1:
            raise DiagnosticError("MLP cut needs unique same-layer projection owners")
        return matches[0]

    gate, down = projection("gate_up_proj"), projection("down_proj")
    middle = [e for e in metadata["events"] if gate["index"] < e["index"] < down["index"]]
    if len(middle) != 1:
        raise DiagnosticError("unrecognized operation sequence inside the MLP activation cut")
    activation = middle[0]
    name = activation["operation"]
    if name == "_C.silu_and_mul.default":
        input_key = f"{activation['index']}.before.args.1"
        output_key = f"{activation['index']}.after.mutable.result"
    elif name in {
        "inductor/triton_poi_fused_mul_mxfp4_linear_silu_slice_0",
        "inductor/triton_poi_fused_mul_mxfp4_linear_silu_slice_1",
    }:
        input_key = f"{activation['index']}.before.args.0"
        output_key = f"{activation['index']}.after.out_ptr0"
    else:
        raise DiagnosticError("unrecognized MLP activation implementation")
    descriptors = {
        r["key"]: r
        for e in (gate, activation, down)
        for phase in ("before", "after")
        for r in e[phase]
    }
    gate_key, down_key = f"{gate['index']}.after.result", f"{down['index']}.before.args.0"
    keys = (input_key, output_key, gate_key, down_key)
    if any(k not in descriptors or k not in tensors for k in keys):
        raise DiagnosticError("MLP cut lacks captured inputs or outputs")
    if any(descriptors[k]["dtype"] != "torch.bfloat16" for k in keys):
        raise DiagnosticError("MLP cut requires the pinned BF16 activation contract")
    positions = metadata["positions"]
    if descriptors[input_key]["shape"] != [len(positions), 34816] or descriptors[output_key][
        "shape"
    ] != [len(positions), 17408]:
        raise DiagnosticError("MLP activation shape differs from this pinned model")
    for left, right in ((input_key, gate_key), (output_key, down_key)):
        result = compare_arrays([tensors[left]], [tensors[right]], positions)
        if result is None or result["exact_positions"] != len(positions):
            raise DiagnosticError(
                "MLP activation tensors do not connect to their named projections"
            )
    return {"operation": name, "input": input_key, "output": output_key}


def anchors(metadata):
    authenticate(metadata)
    grouped = defaultdict(list)
    for event in metadata["events"]:
        identities = tuple(event.get("logical_identities", ()))
        if identities and event["after"]:
            grouped[(event["operation"], identities)].append(event)
    # Repeated calls with the same owner may have different semantic roles.
    # Leave these unpaired instead of assuming the first occurrence corresponds.
    unique = {key: value[0] for key, value in grouped.items() if len(value) == 1}
    return unique, len(grouped) - len(unique)


def compare_arrays(left, right, positions):
    """Compare ordered captured tensors; return unknown for incomplete layouts."""
    if not left or len(left) != len(right) or not positions:
        return None
    equal = np.ones(len(positions), dtype=np.bool_)
    elements = 0
    differing = 0
    for a, b in zip(left, right, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype or not a.ndim:
            return None
        if a.shape[0] not in (len(positions), len(positions) * 48):
            return None
        # Byte comparison retains signed zero/NaN payload and FP8 storage bits.
        raw_a = np.ascontiguousarray(a).view(np.uint8).reshape(len(positions), -1)
        raw_b = np.ascontiguousarray(b).view(np.uint8).reshape(len(positions), -1)
        equal &= (raw_a == raw_b).all(axis=1)
        elements += a.size
        differing += np.count_nonzero((raw_a != raw_b).reshape(-1, a.dtype.itemsize).any(axis=1))
    return {
        "positions": len(positions),
        "exact_positions": int(equal.sum()),
        "different_positions": [p for p, ok in zip(positions, equal, strict=True) if not ok],
        "compared_elements": int(elements),
        "different_elements": int(differing),
    }


def admit_bridge(bridge, pass_receipt, capture, prefill=None):
    """Bind tensors to a completed pass whose observer reproduced its control."""
    for value in (bridge, pass_receipt, capture):
        authenticate(value)
    admission = bridge["admission"]
    authenticate(admission)
    if sorted(admission["captures"]) != [False, True]:
        raise DiagnosticError("output bridge must compare an observer with an uncaptured control")
    side = admission["captures"].index(True)
    if admission["receipts"][side][-1] != pass_receipt["sha256"]:
        raise DiagnosticError("output bridge belongs to another captured pass")
    if (
        bridge["decode"]["positions"] != 320
        or bridge["decode"]["full_logits_exact"] != 320
        or bridge["prefill"]["positions"] != 1
        or bridge["prefill"]["full_logits_exact"] != 1
    ):
        raise DiagnosticError("observer changed the prefill or decode outputs")
    observed = pass_receipt["observation"]["isolated_capture"]
    if observed["sha256"] != capture["sha256"]:
        raise DiagnosticError("tensor capture does not belong to the bridged pass")
    if prefill is not None:
        authenticate(prefill)
        if (
            prefill["sha256"] != observed["prefill_capture"]
            or prefill["decode_capture"] != capture["sha256"]
        ):
            raise DiagnosticError("sampled prefill capture does not belong to the bridged pass")


def compare_group(left, right, left_tensors, right_tensors):
    a, aa = anchors(left)
    b, ab = anchors(right)
    positions = left["positions"]
    if not positions or positions != right["positions"] or len(set(positions)) != len(positions):
        raise DiagnosticError("boundary captures have different or duplicate positions")
    records = []
    for key in sorted(a.keys() & b.keys(), key=lambda k: a[k]["index"]):
        ea, eb = a[key], b[key]

        def values(event, tensors, which):
            return [tensors[r["key"]] for r in event[which]]

        def layouts(event, which):
            return [(r.get("shape"), r.get("dtype")) for r in event[which]]

        if layouts(ea, "after") != layouts(eb, "after"):
            continue
        after = compare_arrays(
            values(ea, left_tensors, "after"), values(eb, right_tensors, "after"), positions
        )
        if after is None:
            continue
        before = compare_arrays(
            values(ea, left_tensors, "before"), values(eb, right_tensors, "before"), positions
        )
        if layouts(ea, "before") != layouts(eb, "before"):
            before = None
        records.append(
            {
                "operation": key[0],
                "owners": list(key[1]),
                "left_event": ea["index"],
                "right_event": eb["index"],
                "captured_inputs": before,
                "outputs": after,
            }
        )
    if not records:
        raise DiagnosticError("no unambiguous comparable semantic boundaries")
    return {
        "positions": positions,
        "boundaries": records,
        "left_unique_unpaired": len(a.keys() - b.keys()),
        "right_unique_unpaired": len(b.keys() - a.keys()),
        "left_ambiguous": aa,
        "right_ambiguous": ab,
    }


def summarize(groups, *, expected_positions, sources):
    observed = [p for group in groups for p in group["positions"]]
    if observed != expected_positions or len(set(observed)) != len(observed):
        raise DiagnosticError("boundary comparison has incomplete or duplicate position coverage")
    totals = {}
    first = None
    for group in groups:
        for row in group["boundaries"]:
            key = (row["operation"], tuple(row["owners"]))
            total = totals.setdefault(
                key,
                {
                    "operation": key[0],
                    "owners": list(key[1]),
                    "positions": 0,
                    "exact_output_positions": 0,
                    "different_output_elements": 0,
                    "captured_input_positions": 0,
                    "exact_captured_input_positions": 0,
                },
            )
            output, inputs = row["outputs"], row["captured_inputs"]
            total["positions"] += output["positions"]
            total["exact_output_positions"] += output["exact_positions"]
            total["different_output_elements"] += output["different_elements"]
            if inputs is not None:
                total["captured_input_positions"] += inputs["positions"]
                total["exact_captured_input_positions"] += inputs["exact_positions"]
            if output["different_positions"]:
                candidate = {
                    "position": min(output["different_positions"]),
                    "operation": key[0],
                    "owners": list(key[1]),
                    "left_event": row["left_event"],
                    "right_event": row["right_event"],
                    "all_captured_inputs_exact": inputs is not None
                    and inputs["exact_positions"] == inputs["positions"],
                }
                if first is None or (candidate["position"], candidate["left_event"]) < (
                    first["position"],
                    first["left_event"],
                ):
                    first = candidate
    return seal(
        {
            "schema": "qwen.execution-mode-boundary-comparison.v1",
            "status": "COMPARED_OBSERVED_BOUNDARIES",
            "sources": sources,
            "positions": len(observed),
            "boundaries": list(totals.values()),
            "first_observed_different_boundary": first,
            "unpaired_and_ambiguous": [
                {k: v for k, v in g.items() if k not in ("positions", "boundaries")} for g in groups
            ],
            "scope": (
                "Named captured activation boundaries only; intervening operations and full "
                "KV/GDN/conv state are not proved equal. No isolated vocabulary top-k claim."
            ),
        }
    )
