"""Native state-slot and arithmetic qualification of the indexed stock scan."""

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
    from stock_gdn_indexed_adapter import StockIndexedAdapter
    from stock_gdn_sequence import STATE_STRIDE, native_sequence

    reference = native_sequence(
        args.contract, args.compiled_root, args.stock_source, args.op_source
    )
    captured, _, _ = validate_capture(args.capture)
    device = reference.device

    def value(name, dtype):
        return torch.from_numpy(array(captured / "before", "kwargs." + name).copy()).to(
            device, dtype
        )

    initial = value("initial_state.selected_values", torch.float32)[0]
    a_log = value("A_log", torch.float32)
    bias = value("dt_bias", torch.bfloat16)
    rng = np.random.default_rng(1701)
    rows = []
    inputs = []
    for name in ("mixed_qkv", "a", "b"):
        x = array(captured / "before", "kwargs." + name).astype(np.float32)
        x = np.repeat(x, 8, axis=0)
        x[1:] += rng.normal(0, 0.03125, x[1:].shape).astype(np.float32)
        inputs.append(torch.from_numpy(x).to(device, torch.bfloat16))
    qkv, a, b = inputs
    expected = reference.d7(initial, qkv, a, b, a_log, bias)
    ids = torch.tensor([[9, 3, 11, 5, 2, 8, 6, 1]], device=device, dtype=torch.int32)
    physical = [9, 3, 11, 5, 2, 8, 6, 1]
    adapter = StockIndexedAdapter(torch)
    for count in range(1, 9):
        q, k, v = [
            part.contiguous().view(count, heads, 128)
            for part, heads in zip(
                qkv[:count].split((2048, 2048, 6144), -1), (16, 16, 48), strict=True
            )
        ]
        for previous in range(1, 9):
            # Padding and unused slots are compared too, so writing the wrong
            # prefix or touching another allocation cannot pass this check.
            storage = torch.full((13 * STATE_STRIDE,), 17.0, device=device, dtype=torch.float32)
            pool = storage.as_strided((13, 48, 128, 128), (STATE_STRIDE, 16384, 128, 1))
            pool[physical[previous - 1]].copy_(initial)
            expect_storage = storage.clone()
            expect_pool = expect_storage.as_strided(pool.shape, pool.stride())
            for i in range(count):
                expect_pool[physical[i]].copy_(expected.after_rows[i])
            output = torch.full((count, 48, 128), float("nan"), device=device, dtype=torch.bfloat16)
            accepted = torch.tensor([previous], device=device, dtype=torch.int32)
            cu = torch.tensor([0, count], device=device, dtype=torch.int32)
            adapter(
                q,
                k,
                v,
                a[:count],
                b[:count],
                a_log,
                bias.float(),
                pool,
                output,
                cu,
                ids,
                accepted,
                1,
                48,
                16,
                128**-0.5,
            )
            label = f"rows-{count}-previous-{previous}"
            rows.append(compare_tensors(label + "-state", storage, expect_storage, args.output))
            rows.append(
                compare_tensors(label + "-output", output, expected.outputs[:count], args.output)
            )
    # Padded rows must leave the null slot and every other slot untouched.
    null_ids = torch.zeros_like(ids)
    before = storage.clone()
    output.fill_(float("nan"))
    adapter(
        q,
        k,
        v,
        a,
        b,
        a_log,
        bias.float(),
        pool,
        output,
        cu,
        null_ids,
        torch.ones_like(accepted),
        1,
        48,
        16,
        128**-0.5,
    )
    rows.append(compare_tensors("null-state", storage, before, args.output))
    rows.append(compare_tensors("null-output", output, torch.zeros_like(output), args.output))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "contract", "compiled-root", "stock-source", "op-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("indexed GDN probe requires admission and the shared GPU lease")
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
        if error is None and len(rows) == 130 and all(r["equal"] for r in rows)
        else "FAILED"
    )
    result = seal(
        {
            "status": status,
            "error": error,
            "checks": rows,
            "scope": (
                "One sequence, all 8 previous acceptance widths by all 8 current row counts; "
                "padded null slot."
            ),
            "formal_equivalence": "UNPROVED",
            "multi_sequence": "UNPROVED",
            "graphs": "UNPROVED",
            "sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "probe_stock_gdn_indexed.py",
                    "stock_gdn_indexed_adapter.py",
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
