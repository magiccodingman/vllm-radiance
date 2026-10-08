#!/usr/bin/env python3
"""Check native GDN prefill against an independent FP64 sequential recurrence.

Random tensors only: no checkpoint or conversation is needed. The optional
--library selects a separately compiled scan translation unit for qualification.
--expect-failure verifies that an uncorrected library reproduces the defect.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import time
from pathlib import Path


def probe(output_path: Path, library_path: Path | None, heads: int):
    import torch
    import radiance_gdn as native
    import r4d

    torch.set_grad_enabled(False)
    torch.manual_seed(14929)
    device = "cuda"
    query_heads, width = heads // 3, 128
    assert heads in (24, 48)
    scale = width ** -0.5
    observed = hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest()
    assert native.ENABLED and native.CHUNK == 64
    library_sha256 = None
    if library_path is not None:
        library_sha256 = hashlib.sha256(library_path.read_bytes()).hexdigest()
        library = ctypes.CDLL(str(library_path.resolve()))
        scan = library.r4d_gdn_chunk_scan_k128_v128_c64_bf16
        scan.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int] * 6
                         + [ctypes.c_float, ctypes.c_void_p])
        scan.restype = ctypes.c_int
        native._CHUNK_SCAN = scan
    started = time.monotonic()
    rows = []
    cases = [(65, 0.02, 1, [65], False), (128, 1.0, 1, [128], False),
             (128, 2.5, 1, [128], False), (128, 3.2, 1, [128], False),
             (128, 8.0, 1, [128], False), (257, 0.02, 1, [257], False),
             (257, 3.2, 1, [257], False), (1024, 0.02, 1, [1024], False),
             (128, 32.0, 1, [128], False), (128, 2.5, 10000, [128], False),
             (195, 0.02, 1, [65, 130], False), (195, 3.2, 1, [65, 130], True)]
    for tokens, decay, amplitude, lengths, mixed_heads in cases:
        q = torch.randn(tokens, query_heads, width, device=device)
        k = torch.randn_like(q)
        q = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(k, dim=-1).to(torch.bfloat16)
        v = torch.randn(tokens, heads, width, device=device, dtype=torch.bfloat16)
        v *= amplitude
        steps = torch.full((tokens, heads), -decay, device=device, dtype=torch.float32)
        if mixed_heads:
            steps[:, 1:-1] = -0.02
        beta = torch.full_like(steps, 0.5)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        assert boundaries[-1] == tokens
        cumulative = torch.cat([
            steps[i:min(i + native.CHUNK, end)].cumsum(0)
            for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            for i in range(begin, end, native.CHUNK)
        ], dim=0).contiguous()
        cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
        initial = torch.randn(len(lengths), heads, width, width, device=device) * 0.01
        matrix = native.kkt_solve(k, beta, cumulative, cu, len(lengths), tokens, heads, query_heads)
        # Guard the caller-owned output, including non-full final chunks.
        count = v.numel()
        slab = torch.full((count + 512,), 37.0, device=device, dtype=torch.bfloat16)
        output = slab[256:256 + count].view(1, tokens, heads, width)
        actual, final = native.fused_prefill(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), matrix.unsqueeze(0),
            cumulative.unsqueeze(0), beta.unsqueeze(0), scale, initial, True, cu,
            None, out=output,
        )
        torch.cuda.synchronize()
        # No WY factorization, chunk scan, kernel output or native intermediate
        # participates in this reference. It follows the defining recurrence.
        qr = q.double().repeat_interleave(heads // query_heads, dim=1)
        kr = k.double().repeat_interleave(heads // query_heads, dim=1)
        vr, br, gr = v.double(), beta.double(), steps.double().exp()
        oracle = torch.empty_like(vr)
        oracle_final = torch.empty_like(initial, dtype=torch.float64)
        for sequence, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            state = initial[sequence].double().clone()
            for position in range(begin, end):
                state *= gr[position, :, None, None]
                residual = vr[position] - (state * kr[position, :, None, :]).sum(-1)
                state += (br[position, :, None] * residual)[:, :, None] * kr[position, :, None, :]
                oracle[position] = (state * qr[position, :, None, :]).sum(-1) * scale
            oracle_final[sequence] = state
        out_error = ((actual[0].double() - oracle).norm() / oracle.norm()).item()
        state_error = ((final.double() - oracle_final).norm() / oracle_final.norm()).item()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(final).all())
        guards = bool((slab[:256] == 37).all() and (slab[-256:] == 37).all())
        row = {"tokens": tokens, "heads": heads, "query_heads": query_heads,
               "sequence_lengths": lengths, "mixed_head_decay": mixed_heads,
               "value_amplitude": amplitude,
               "negative_log_decay_per_token": decay,
               "maximum_chunk_decay_span": min(tokens - 1, 63) * decay,
               "output_relative_error": out_error if math.isfinite(out_error) else None,
               "state_relative_error": state_error if math.isfinite(state_error) else None,
               "finite": finite, "guards_intact": guards,
               "within_one_percent": finite and guards and max(out_error, state_error) < 0.01}
        rows.append(row)
        output_path.write_text(json.dumps({"complete": False, "cases": rows}, indent=2))
        print(json.dumps(row), flush=True)
    # The last case has two unequal sequence lengths and mixed affected /
    # unaffected heads. Capture both launches and replay with changed q data.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output, graph_final = native.fused_prefill(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), matrix.unsqueeze(0),
            cumulative.unsqueeze(0), beta.unsqueeze(0), scale, initial, True, cu,
            None, out=output,
        )
    q.mul_(-1)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    graph_out_error = ((graph_output[0].double() + oracle).norm() / oracle.norm()).item()
    graph_state_error = ((graph_final.double() - oracle_final).norm() / oracle_final.norm()).item()
    graph_report = {"replays": 3, "changed_query_inputs": True,
                    "output_relative_error": graph_out_error,
                    "state_relative_error": graph_state_error,
                    "within_one_percent": max(graph_out_error, graph_state_error) < 0.01}
    report = {"complete": True, "reference": "independent FP64 sequential gated delta recurrence",
              "r4d_sha256": observed, "elapsed_seconds": time.monotonic() - started,
              "all_within_one_percent": all(row["within_one_percent"] for row in rows),
              "scan_library_sha256": library_sha256,
              "graph_capture": graph_report, "cases": rows}
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--heads", type=int, choices=(24, 48), default=48)
    parser.add_argument("--expect-failure", action="store_true")
    args = parser.parse_args()
    result = probe(args.output, args.library, args.heads)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))
    passed = result["all_within_one_percent"] and result["graph_capture"]["within_one_percent"]
    if args.expect_failure:
        assert not passed, "uncorrected control did not reproduce the defect"
    else:
        assert passed, "native GDN disagrees with the independent recurrence"
