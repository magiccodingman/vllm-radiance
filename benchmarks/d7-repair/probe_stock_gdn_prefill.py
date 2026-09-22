"""Native causal-prefix/chunk qualification of the stock-arithmetic scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def qualify(args):
    import numpy as np
    import torch
    from analyze_gdn_decode_transition import array
    from probe_gdn_beta_precision import validate_capture
    from probe_stock_gdn_sequence import compare_tensors
    from stock_gdn_scan import StockScan
    from stock_gdn_sequence import native_sequence

    reference = native_sequence(
        args.contract, args.compiled_root, args.stock_source, args.op_source
    )
    captured, _, _ = validate_capture(args.capture)
    device = reference.device
    candidate = StockScan(torch, device)

    def raw(name, dtype):
        return torch.from_numpy(array(captured / "before", "kwargs." + name).copy()).to(
            device, dtype
        )

    initial = raw("initial_state.selected_values", torch.float32)[0]
    a_log, bias = raw("A_log", torch.float32), raw("dt_bias", torch.bfloat16)
    rng = np.random.default_rng(4242)
    inputs = []
    for name in ("mixed_qkv", "a", "b"):
        x = array(captured / "before", "kwargs." + name).astype(np.float32)
        x = np.repeat(x, 257, axis=0)
        x += rng.normal(0, 0.0625, x.shape).astype(np.float32)
        inputs.append(torch.from_numpy(x).to(device, torch.bfloat16))
    qkv, a, b = inputs
    # These rows cross sigmoid/softplus saturation and the softplus threshold.
    # Future extreme gates must not change any earlier output or state.
    a[128] = -96
    a[256] = 64
    b[64] = -32
    b[127] = 32
    rows = []

    def check(name, actual, expected):
        rows.append(compare_tensors(name, actual, expected, args.output))

    lengths = (1, 8, 63, 64, 65, 127, 128, 129, 257)
    for fresh in (False, True):
        origin = torch.zeros_like(initial) if fresh else initial
        state = origin
        reference_outputs = []
        prefixes = {}
        for position in range(257):
            stop = position + 1
            part = reference.run(
                state, qkv[position:stop], a[position:stop], b[position:stop], a_log, bias
            )
            state = part.final_state
            reference_outputs.append(part.outputs)
            if stop in lengths:
                prefixes[stop] = state.clone()
        expected_outputs = torch.cat(reference_outputs)
        for length in lengths:
            result = candidate.run(origin, qkv[:length], a[:length], b[:length], a_log, bias)
            label = f"fresh-{int(fresh)}-length-{length}"
            check(label + "-state", result.final_state, prefixes[length])
            check(label + "-output", result.outputs, expected_outputs[:length])
        for label, sizes in (
            ("regular", (64, 64, 64, 64, 1)),
            ("irregular", (1, 7, 0, 55, 2, 63, 1, 128)),
        ):
            state, start, parts = origin, 0, []
            for size in sizes:
                stop = start + size
                part = candidate.run(
                    state, qkv[start:stop], a[start:stop], b[start:stop], a_log, bias
                )
                state, start = part.final_state, stop
                parts.append(part.outputs)
            if start != 257:
                raise DiagnosticError("prefill qualification omitted part of the domain")
            check(f"fresh-{int(fresh)}-{label}-state", state, prefixes[257])
            check(f"fresh-{int(fresh)}-{label}-output", torch.cat(parts), expected_outputs)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "contract", "compiled-root", "stock-source", "op-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("stock prefill probe requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    rows, error = [], None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            rows = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = (
        "TESTED"
        if error is None and len(rows) == 44 and all(x["equal"] for x in rows)
        else "FAILED"
    )
    result = seal(
        {
            "status": status,
            "error": error,
            "checks": rows,
            "scope": (
                "Fresh/continued states, nine lengths through 257, future extreme gates "
                "and two chunk partitions."
            ),
            "formal_equivalence": "UNPROVED",
            "sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "probe_stock_gdn_prefill.py",
                    "stock_gdn_scan.py",
                    "stock_gdn_scan_kernel.py",
                )
            },
        }
    )
    write_private(args.output / "probe-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if status == "TESTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
