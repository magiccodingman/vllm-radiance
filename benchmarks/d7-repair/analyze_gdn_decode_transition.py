"""CPU-only first-decode comparison after the causal prefill and convolution fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_reference import bf16
from qwen_r9700_lab.conformance_state import read_frame, values
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def call(root, arm, site):
    base = root / arm / "run/capture/calls"
    manifest = private_json(base / "calls.json")
    authenticate(manifest)
    rows = [r for r in manifest["calls"] if r["site"] == site and r["before"]["frame"]]
    if len(rows) != 1:
        raise ValueError(f"expected one captured {arm}/{site} transition, got {len(rows)}")
    return base / f"call-{rows[0]['index']:09d}"


def array(path, name, first_index=None):
    frame = read_frame(path)
    descriptor = frame["components"][name]
    raw = (path / descriptor["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("captured tensor hash mismatch")
    result = values(raw, descriptor["dtype"]).reshape(descriptor["shape"])
    return result if first_index is None else result[first_index]


def stats(left, right):
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("comparison requires matching finite arrays")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "equal": bool(np.array_equal(left, right)),
        "different_elements": int(np.count_nonzero(delta)),
        "elements": delta.size,
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(right), 1e-30)),
    }


def analyze(root):
    model = private_json(root / "probe-result.json")
    authenticate(model)
    if not model["initial_prefill_exact"]:
        raise ValueError("cannot attribute decode difference when initial states already differ")
    stock = call(root, "serial", "stock.gdn.packed_decode")
    native = call(root, "d7", "radiance_gdn.recurrent_update")
    a0, b0, log_a, dt = [
        array(stock / "before", "kwargs." + n) for n in ("a", "b", "A_log", "dt_bias")
    ]
    a1, b1, log_a1, dt1 = [array(native / "before", "args." + str(i)) for i in (3, 4, 5, 6)]
    s0 = array(stock / "before", "kwargs.initial_state.selected_values", 0)
    after0 = array(stock / "after", "kwargs.initial_state.selected_values", 0)
    ids = array(native / "before", "args.7.indices").astype(int)
    mapping = array(native / "before", "args.10").astype(int)
    accepted = array(native / "before", "args.11").astype(int)
    if accepted.tolist() != [1]:
        raise ValueError("unreviewed accepted-prefix mapping")
    input_slot = int(np.flatnonzero(ids == mapping[0, accepted[0] - 1])[0])
    output_slot = int(np.flatnonzero(ids == mapping[0, 0])[0])
    s1 = array(native / "before", "args.7.selected_values", input_slot)
    after1 = array(native / "after", "args.7.selected_values", output_slot)
    packed = array(stock / "before", "kwargs.mixed_qkv")[0]
    q0, k0, v0 = np.split(packed, [16 * 128, 32 * 128])
    q0, k0, v0 = q0.reshape(16, 128), k0.reshape(16, 128), v0.reshape(48, 128)
    q1, k1, v1 = [array(native / "before", "args." + str(i))[0] for i in (0, 1, 2)]
    qn = q0 / np.sqrt(np.sum(q0 * q0, axis=-1, keepdims=True) + 1e-6)
    kn = k0 / np.sqrt(np.sum(k0 * k0, axis=-1, keepdims=True) + 1e-6)
    inputs = {
        "initial_state": stats(s0, s1),
        "a": stats(a0[0], a1[0]),
        "b": stats(b0[0], b1[0]),
        "A_log": stats(log_a, log_a1),
        "dt_bias": stats(dt, dt1),
        "value": stats(v0, v1),
        "query_raw": stats(q0, q1),
        "key_raw": stats(k0, k1),
    }
    gate = -np.exp(log_a) * np.logaddexp(0, a0[0] + dt)
    beta = 1 / (1 + np.exp(-b0[0]))
    out0 = array(stock / "after", "kwargs.out")[0, 0]
    out1 = array(native / "after", "args.8")[0]
    variants = []
    for label, q, k in (
        ("unrounded_normalization", qn, kn),
        ("bf16_normalization", bf16(qn).astype(float), bf16(kn).astype(float)),
    ):
        for beta_label, b in (("unrounded_beta", beta), ("bf16_beta", bf16(beta).astype(float))):
            qs = np.repeat(q, 3, axis=0) * (128**-0.5)
            ks = np.repeat(k, 3, axis=0)
            state = s0 * np.exp(gate)[:, None, None]
            residual = b[:, None] * (v0 - np.sum(state * ks[:, None, :], axis=-1))
            state = state + residual[:, :, None] * ks[:, None, :]
            output = np.sum(state * qs[:, None, :], axis=-1)
            variants.append(
                {
                    "qk": label,
                    "beta": beta_label,
                    "state_to_m1": stats(state, after0),
                    "state_to_d7": stats(state, after1),
                    "output_to_m1": stats(output, out0),
                    "output_to_d7": stats(output, out1),
                }
            )
    return seal(
        {
            "scope": (
                "One captured first decode transition; float64 arithmetic diagnostics, not a proof."
            ),
            "model_report": model["sha256"],
            "inputs": inputs,
            "committed_state_difference": stats(after0, after1),
            "output_difference": stats(out0, out1),
            "variants": variants,
            "selected_storage_metadata": {
                "d7_indices": ids.tolist(),
                "d7_mapping": mapping.tolist(),
                "previous_accepted": accepted.tolist(),
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    write_private(args.root / "gdn-numerical-analysis.json", result)
    print(json.dumps(result))
