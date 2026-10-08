"""Qualify the pinned-stock GDN fold against a retained native transition.

This opt-in operator experiment never installs a hook or changes the matrix.
Run only as an admitted GPU job, after the current owner releases its lease.
The recorded wall time includes diagnostic checks and is not a speed benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def compare_tensors(label, left, right, output):
    if left.dtype != right.dtype or left.shape != right.shape:
        raise DiagnosticError("GDN comparison requires identical representations")
    # Comparing physical bytes detects signed zeros and one-bit state damage.
    import torch

    arrays = []
    for value in (left, right):
        # A singleton can be "contiguous" while retaining a non-unit stride.
        # Materialize canonical logical storage before reinterpreting its bytes.
        packed = torch.empty(value.shape, dtype=value.dtype, device="cpu")
        packed.copy_(value.detach())
        arrays.append(packed.reshape(-1).view(torch.uint8).numpy())
    differences = np.flatnonzero(arrays[0].reshape(-1) != arrays[1].reshape(-1))
    row = {
        "case": label,
        "equal": len(differences) == 0,
        "different_bytes": len(differences),
        "first_differing_byte": int(differences[0]) if len(differences) else None,
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "payload_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
    }
    if len(differences):
        evidence = output / label
        evidence.mkdir(mode=0o700)
        for name, array in zip(("actual.bin", "expected.bin"), arrays, strict=True):
            (evidence / name).write_bytes(array.tobytes())
        write_private(evidence / "comparison.json", seal(row))
    return row


def classify(rows, negative_controls):
    # Check the actual named domain, not merely all([]) or a subset of cases.
    expected = {"captured-stock-state", "captured-stock-output"}
    expected.update(f"accept-{k}-{kind}" for k in range(8) for kind in ("state", "output"))
    expected.update(f"suffix-{k}-{kind}" for k in range(7) for kind in ("state", "output"))
    expected.update(f"chunks-{i}-{kind}" for i in range(3) for kind in ("state", "output"))
    if len(rows) != len(expected) or {r["case"] for r in rows} != expected:
        return "INCOMPLETE"
    if set(negative_controls) != {"wrong-prefix-state", "one-bit-state-corruption"}:
        return "INCOMPLETE"
    if not all(negative_controls.values()):
        return "INVALID_CHECKER"
    return "TESTED" if all(r["equal"] for r in rows) else "FAILED"


def replay(args):
    import torch
    from analyze_gdn_decode_transition import array
    from probe_gdn_beta_precision import validate_capture
    from stock_gdn_sequence import native_sequence

    stock_call, _, _ = validate_capture(args.capture)
    stock = native_sequence(args.contract, args.compiled_root, args.stock_source, args.op_source)
    serial = native_sequence(args.contract, args.compiled_root, args.stock_source, args.op_source)
    if args.candidate == "scan":
        from stock_gdn_scan import StockScan

        stock = StockScan(torch, stock.device)
    device = stock.device

    def tensor(value, dtype):
        return torch.from_numpy(value.copy()).to(device=device, dtype=dtype)

    def raw(name):
        return array(stock_call / "before", "kwargs." + name)

    initial = tensor(raw("initial_state.selected_values")[0], torch.float32)
    a_log = tensor(raw("A_log"), torch.float32)
    dt_bias = tensor(raw("dt_bias"), torch.bfloat16)
    # Row zero reproduces the captured transition exactly. Later rows are
    # deterministic synthetic continuations of these operator inputs.
    rng = np.random.default_rng(1701)
    values = []
    for name in ("mixed_qkv", "a", "b"):
        base = raw(name).astype(np.float32)
        repeated = np.repeat(base, 8, axis=0)
        repeated[1:] += rng.normal(0, 0.03125, repeated[1:].shape).astype(np.float32)
        values.append(tensor(repeated, torch.bfloat16))
    qkv, a, b = values
    expected_state = tensor(
        array(stock_call / "after", "kwargs.initial_state.selected_values")[0], torch.float32
    )
    expected_output = tensor(array(stock_call / "after", "kwargs.out")[0, 0], torch.bfloat16)
    rows = []

    def check(label, actual, expected):
        rows.append(compare_tensors(label, actual, expected, args.output))

    first = stock.run(initial, qkv[:1], a[:1], b[:1], a_log, dt_bias)
    check("captured-stock-state", first.final_state, expected_state)
    check("captured-stock-output", first.outputs[0], expected_output)
    if not all(r["equal"] for r in rows):
        return rows, {}, "captured stock control did not reproduce"

    batch = stock.d7(initial, qkv, a, b, a_log, dt_bias)
    state = initial
    serial_outputs = []
    for accepted in range(8):
        stop = accepted + 1
        part = serial.run(
            state, qkv[accepted:stop], a[accepted:stop], b[accepted:stop], a_log, dt_bias
        )
        state = part.final_state
        serial_outputs.append(part.outputs)
        check(f"accept-{accepted}-state", batch.accepted_state(accepted), state)
        check(f"accept-{accepted}-output", batch.outputs[:stop], torch.cat(serial_outputs))
    for accepted in range(7):
        poisoned = [value.clone() for value in (qkv, a, b)]
        for value in poisoned:
            value[accepted + 1 :] = -3
        other = stock.d7(initial, *poisoned, a_log, dt_bias)
        check(
            f"suffix-{accepted}-state",
            other.accepted_state(accepted),
            batch.accepted_state(accepted),
        )
        check(
            f"suffix-{accepted}-output",
            other.outputs[: accepted + 1],
            batch.outputs[: accepted + 1],
        )
    for index, sizes in enumerate(((1,) * 8, (2, 3, 3), (0, 3, 0, 5, 0))):
        state, start, outputs = initial, 0, []
        for size in sizes:
            stop = start + size
            part = stock.run(state, qkv[start:stop], a[start:stop], b[start:stop], a_log, dt_bias)
            state, start = part.final_state, stop
            outputs.append(part.outputs)
        check(f"chunks-{index}-state", state, batch.final_state)
        check(f"chunks-{index}-output", torch.cat(outputs), batch.outputs)

    wrong = compare_tensors(
        "negative-wrong-prefix-state", batch.accepted_state(1), batch.accepted_state(0), args.output
    )
    damaged = batch.accepted_state(0).cpu()
    damaged.view(torch.uint8).reshape(-1)[0] ^= 1
    corrupt = compare_tensors(
        "negative-one-bit-state-corruption", damaged, batch.accepted_state(0), args.output
    )
    controls = {
        "wrong-prefix-state": not wrong["equal"],
        "one-bit-state-corruption": not corrupt["equal"],
    }
    return rows, controls, None


def main(args):
    if not args.allow_gpu:
        raise DiagnosticError("stock GDN native replay requires explicit --allow-gpu")
    if not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("stock GDN native replay requires the shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    started = time.monotonic()
    rows, controls, error = [], {}, None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            rows, controls, error = replay(args)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    status = "FAILED" if error else classify(rows, controls)
    report = seal(
        {
            "schema": "urn:qwen:stock-gdn-sequence-probe:v1",
            "status": status,
            "scope": "One retained stock transition and synthetic eight-row operator sequences.",
            "checks": rows,
            "negative_controls": controls,
            "error": error,
            "contract": json.loads(args.contract.read_text())["sha256"],
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(
                Path(__file__)
                .with_name(
                    "stock_gdn_scan.py" if args.candidate == "scan" else "stock_gdn_sequence.py"
                )
                .read_bytes()
            ).hexdigest(),
            "candidate": args.candidate,
            "kernel_sha256": (
                hashlib.sha256(
                    Path(__file__).with_name("stock_gdn_scan_kernel.py").read_bytes()
                ).hexdigest()
                if args.candidate == "scan"
                else None
            ),
            "wall_seconds_including_diagnostics": time.monotonic() - started,
            "performance_benchmark": False,
            "full_model_equivalence": "UNPROVED",
            "formal_equivalence": "UNPROVED",
            "installed_backend_changed": False,
            "baseline_matrix_changed": False,
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(
        json.dumps({"status": status, "checks": len(rows), "sha256": report["sha256"]}), flush=True
    )
    return 0 if status == "TESTED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--stock-source", type=Path, required=True)
    parser.add_argument("--op-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true", required=True)
    parser.add_argument("--candidate", choices=("oracle", "scan"), default="oracle")
    raise SystemExit(main(parser.parse_args()))
