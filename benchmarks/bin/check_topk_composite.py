#!/usr/bin/env python3
"""CPU equivalence gate for the RX4 small-k composite sampling mask.

This compares the new multi-block top-k implementation with vLLM's untouched
full-sort PyTorch reference. Random fp32 logits almost surely avoid the exact
boundary-tie limitation documented in radiance_topk.py; explicit top-k-only tie
cases verify that all values tied at the k-th threshold remain admitted.
"""

from __future__ import annotations

import argparse

import torch

import radiance_topk
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def check_case(rows: int, vocab: int, kvals: list[int], pval: float | None,
               seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn((rows, vocab), generator=generator, dtype=torch.float32)
    k = torch.tensor(kvals, dtype=torch.int32)
    p = None if pval is None else torch.full((rows,), pval, dtype=torch.float32)
    expected = apply_top_k_top_p_pytorch(
        logits.clone(), k, p, allow_cpu_sync=True)
    actual = radiance_topk.apply_top_k_top_p_composite(logits.clone(), k, p)
    if not torch.equal(torch.isfinite(actual), torch.isfinite(expected)):
        mismatches = int(
            (torch.isfinite(actual) != torch.isfinite(expected)).sum().item())
        raise AssertionError(
            f"mask mismatch rows={rows} vocab={vocab} p={pval}: {mismatches}")
    finite = torch.isfinite(expected)
    if not torch.equal(actual[finite], expected[finite]):
        raise AssertionError(
            f"kept-logit mismatch rows={rows} vocab={vocab} p={pval}")


def check_topk_boundary_ties() -> None:
    logits = torch.arange(256, dtype=torch.float32).repeat(3, 1)
    logits[:, 230:240] = 1000.0
    k = torch.tensor([20, 24, 32], dtype=torch.int32)
    expected = apply_top_k_top_p_pytorch(
        logits.clone(), k, None, allow_cpu_sync=True)
    actual = radiance_topk.apply_top_k_top_p_composite(
        logits.clone(), k, None)
    if not torch.equal(actual, expected):
        raise AssertionError("top-k boundary ties did not match the full-sort reference")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-vocab", type=int, default=151936)
    args = parser.parse_args()

    cases = [
        (1, 4096, [1]),
        (4, 4096, [1, 5, 20, 64]),
        (9, 4096, [20, 20, 5, 64, 1, 32, 7, 40, 16]),
        (4, args.large_vocab, [1, 20, 32, 64]),
    ]
    seed = 20260831
    for rows, vocab, kvals in cases:
        for pval in (None, 0.5, 0.9, 0.95, 1.0):
            check_case(rows, vocab, kvals, pval, seed)
            seed += 1
    check_topk_boundary_ties()
    print("RX4 composite top-k/top-p matches the full-sort reference on 21 cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
