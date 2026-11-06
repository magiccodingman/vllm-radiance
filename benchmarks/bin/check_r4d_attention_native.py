#!/usr/bin/env python3
"""Bounded R4D paged-attention correctness, not a throughput benchmark.

Compare installed kernels with independent FP64 causal attention over the exact
stored BF16/FP8 cache values. Exercise shuffled pages, partial pages/tiles,
sequences of different context lengths, decode split merging, output guards and
changed-input graph replay. Requires exactly one visible authorized GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import r4d


def reference(q, stored, table, lengths, kscale, vscale):
    n, qlen, qheads, width = q.shape
    kvheads = stored.shape[1]
    result = torch.empty_like(q, dtype=torch.float64)
    for seq in range(n):
        context = int(lengths[seq])
        slots = torch.arange(context, device=q.device)
        pages = table[seq, slots // 16].long()
        values = stored.float()[pages, :, slots % 16, :].double()
        key = (values[..., :width] * kscale[seq][None, :, None]).repeat_interleave(qheads // kvheads, dim=1)
        value = (values[..., width:] * vscale[seq][None, :, None]).repeat_interleave(qheads // kvheads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[seq].double(), key) / width**0.5
        query_positions = torch.arange(context - qlen, context, device=q.device)
        mask = slots[None, :] > query_positions[:, None]
        scores.masked_fill_(mask[None], float("-inf"))
        result[seq] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "preserve prior run evidence"
    assert torch.cuda.device_count() == 1, "expose only the authorized GPU"
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    torch.manual_seed(3306)
    records = []
    cases = [(1, [17, 33]), (7, [65, 129]), (10, [1025, 769]),
             (17, [129, 193]), (65, [257, 193]), (129, [513, 385])]
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp8", torch.float8_e4m3fn)):
        for qlen, lengths_list in cases:
            n, kvheads, width = len(lengths_list), 4, 256
            qheads = kvheads * 6
            maximum = max(lengths_list)
            blocks = (maximum + 15) // 16
            table = (torch.randperm(n * blocks, device="cuda") + 1).int().view(n, blocks)
            stored = (torch.randn((1 + n * blocks, kvheads, 16, 2 * width), device="cuda") * 0.5).to(dtype)
            q = torch.randn((n, qlen, qheads, width), device="cuda", dtype=torch.bfloat16)
            lengths = torch.tensor(lengths_list, device="cuda", dtype=torch.int32)
            ks = torch.full((n, kvheads), 0.75, device="cuda")
            vs = torch.full((n, kvheads), 1.25, device="cuda")
            geometry = dict(head_dim=width, gqa=6, block_size=16, causal=1,
                            q_dtype="bf16", kv_dtype=("fp8_e4m3" if dtype_name == "fp8" else "bf16"))
            operation = "attn_decode_paged" if qlen <= 10 else "attn_prefill_paged"
            if qlen <= 10:
                geometry["q_len"] = qlen
            name = r4d.select(operation, **geometry)
            assert name is not None, (operation, geometry)
            launch = getattr(r4d, name)
            size = int(r4d.attn_decode_h256_gqa6_scratch_bytes(n, min(qlen, 10), qheads, kvheads, width, maximum, 0))
            scratch = torch.empty(max(1, size), dtype=torch.uint8, device="cuda")
            slab = torch.full((q.numel() + 512,), 37.0, dtype=q.dtype, device="cuda")
            out = slab[256:-256].view_as(q)

            def call():
                launch(q.data_ptr(), stored.data_ptr(), table.data_ptr(), lengths.data_ptr(),
                       out.data_ptr(), ks.data_ptr(), vs.data_ptr(), scratch.data_ptr(),
                       n, qlen, qheads, kvheads, width, 16, blocks, stored.stride(0),
                       stored.stride(1), width**-0.5, 0, maximum,
                       torch.cuda.current_stream().cuda_stream)

            def check():
                torch.cuda.synchronize()
                expected = reference(q, stored, table, lengths_list, ks, vs)
                delta = out.double() - expected
                relative = float(delta.norm() / expected.norm())
                max_error = float(delta.abs().max())
                assert torch.isfinite(out).all()
                assert relative < 0.01, (name, qlen, relative)
                assert max_error < float(expected.abs().max()) * 0.02 + 1e-4
                assert (slab[:256] == 37).all() and (slab[-256:] == 37).all()
                return relative, max_error

            for _ in range(3):
                call()
            relative, maximum_error = check()
            captured = out.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            q.mul_(1.5)
            graph.replay()
            graph_relative, _ = check()
            assert not torch.equal(captured, out), "graph ignored changed query"
            records.append({"kv_dtype": dtype_name, "q_len": qlen, "contexts": lengths_list,
                            "kernel": name, "relative_error": relative,
                            "maximum_absolute_error": maximum_error,
                            "changed_input_graph_relative_error": graph_relative})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"status": "NATIVE_ATTENTION_PASS_NOT_PERFORMANCE",
        "r4d_version": r4d.__version__, "maintained_commit": getattr(r4d, "__maintained_commit__", None),
        "extension_sha256": hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0), "cases": records}, indent=2) + "\n")
    print(f"R4D paged attention/guards/changed-input graphs: {len(records)} cases PASS")


if __name__ == "__main__":
    main()
