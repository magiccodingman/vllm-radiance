"""Compare captured indexed-GDN execution with the pinned stock M1 oracle."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def qualify(args):
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

    def raw(name, dtype):
        return torch.from_numpy(array(captured / "before", "kwargs." + name).copy()).to(
            device, dtype
        )

    initial = raw("initial_state.selected_values", torch.float32)[0]
    a_log = raw("A_log", torch.float32)
    bias = raw("dt_bias", torch.bfloat16)
    qkv = raw("mixed_qkv", torch.bfloat16).repeat(8, 1)
    a, b = [raw(n, torch.bfloat16).repeat(8, 1) for n in ("a", "b")]
    q, k, v = [
        part.contiguous().view(8, heads, 128)
        for part, heads in zip(qkv.split((2048, 2048, 6144), -1), (16, 16, 48), strict=True)
    ]
    mapping = [9, 3, 11, 5, 2, 8, 6, 1]
    indices = torch.tensor([mapping], device=device, dtype=torch.int32)
    accepted = torch.ones(1, device=device, dtype=torch.int32)
    cu = torch.tensor([0, 8], device=device, dtype=torch.int32)
    baseline = torch.full((13 * STATE_STRIDE,), 17.0, device=device, dtype=torch.float32)
    baseline_pool = baseline.as_strided((13, 48, 128, 128), (STATE_STRIDE, 16384, 128, 1))
    baseline_pool[mapping[0]].copy_(initial)
    other_initial = initial + 0.001
    baseline_pool[mapping[7]].copy_(other_initial)
    storage = baseline.clone()
    pool = storage.as_strided(baseline_pool.shape, baseline_pool.stride())
    output = torch.empty((8, 48, 128), device=device, dtype=torch.bfloat16)
    widened_bias = bias.float()
    adapter = StockIndexedAdapter(torch)

    def invoke():
        storage.copy_(baseline)
        adapter(
            q,
            k,
            v,
            a,
            b,
            a_log,
            widened_bias,
            pool,
            output,
            cu,
            indices,
            accepted,
            1,
            48,
            16,
            128**-0.5,
        )

    # Compile and populate allocator pools before entering capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    rows = []
    for previous, origin in ((1, initial), (8, other_initial), (1, initial)):
        accepted.fill_(previous)
        graph.replay()
        result = reference.d7(origin, qkv, a, b, a_log, bias)
        expected = baseline.clone()
        expected_pool = expected.as_strided(pool.shape, pool.stride())
        for row, slot in enumerate(mapping):
            expected_pool[slot].copy_(result.after_rows[row])
        label = f"replay-{len(rows) // 2}-previous-{previous}"
        rows.append(compare_tensors(label + "-state", storage, expected, args.output))
        rows.append(compare_tensors(label + "-output", output, result.outputs, args.output))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "contract", "compiled-root", "stock-source", "op-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("stock graph probe requires admission and shared GPU lease")
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
        "TESTED" if error is None and len(rows) == 6 and all(x["equal"] for x in rows) else "FAILED"
    )
    result = seal(
        {
            "status": status,
            "error": error,
            "checks": rows,
            "scope": (
                "One indexed D7 sequence captured once, replayed with changed acceptance "
                "1/8/1 and distinct input states."
            ),
            "formal_equivalence": "UNPROVED",
            "full_model_graphs": "UNPROVED",
            "sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "probe_stock_gdn_graphs.py",
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
