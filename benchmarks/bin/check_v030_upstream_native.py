#!/usr/bin/env python3
"""Bounded native regressions for retired sampler/graph overlays."""
import argparse
import json
from pathlib import Path
import torch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p, apply_top_k_top_p_pytorch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(3)
    # Two request rows, nonzero cache columns, and a wider reserved vocabulary.
    logits = torch.randn((2, 256), device="cuda")
    mapping = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    temp = torch.ones(2, device="cuda")
    seeds = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    pos = torch.tensor([9, 10], device="cuda", dtype=torch.int32)
    cache = torch.full((2, 4, 272), -123.0, device="cuda")
    columns = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    gumbel_sample(logits, mapping, temp, seeds, pos, False, True, cache, columns)
    torch.cuda.synchronize()
    assert torch.equal(cache[0, 1, :256], logits[0])
    assert torch.equal(cache[1, 2, :256], logits[1])
    assert (cache[:, :, 256:] == -123).all()
    assert (cache[:, 0] == -123).all()
    try:
        gumbel_sample(logits, mapping, temp, seeds, pos, False, True,
                      torch.empty((2, 4, 128), device="cuda"), columns)
        raise RuntimeError("narrow cache accepted")
    except AssertionError as error:
        assert "narrower" in str(error)
    for rows in (1, 2, 8):
        x = torch.randn((rows, 4096), device="cuda")
        k = torch.full((rows,), 20, device="cuda", dtype=torch.int32)
        p = torch.full((rows,), 0.95, device="cuda")
        actual = apply_top_k_top_p(x.clone(), k, p)
        expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
        assert torch.equal(actual, expected)
    # Changed-input capture on the current nondefault stream. Full-model V2
    # evidence separately proves its original graph wrapper selects this path.
    stream = torch.cuda.Stream()
    x = torch.ones(128, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = x + 1
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = x + 1
    ptr = y.data_ptr()
    with torch.cuda.stream(stream):
        x.fill_(7)
        graph.replay()
    stream.synchronize()
    assert y.data_ptr() == ptr and (y == 8).all()
    args.out.write_text(json.dumps({"device": args.device, "status": "PASS",
        "gumbel_stride_and_guard": True, "topk_rows": [1,2,8],
        "changed_input_graph": True}, indent=2))


if __name__ == "__main__":
    main()
