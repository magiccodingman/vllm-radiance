"""Composite top-k/top-p masking for small row counts on gfx1201.

WHY. The Qrita-style `_topk_topp_kernel` runs ONE Triton program per row, and each program scans
the whole 151,936-entry vocabulary in serial tiles, several passes deep. At the rejection
sampler's operating point -- batch x (SPEC+1) = 8-9 rows, once per decode step -- that is 8
workgroups on a 32-WGP GPU: 907 us of a ~24 ms step measured in the 2026-08-30 census, the
single largest non-GEMM kernel in the build. The same trace prices `torch.topk` on the identical
[rows, vocab] shape at ~51 us (the drafter rerank calls it every step), because gatherTopK is a
multi-block radix select that actually fills the machine.

WHAT. For rows whose top_k is a real cap (0 < k <= KCAP), the exact answer only needs the top-64
candidates per row:
  1. `torch.topk(logits, KCAP)` -- multi-block, fast, gives (vals desc, idx).
  2. top-k as a VALUE threshold: the reference sort path keeps every entry >= the k-th largest
     value (boundary ties are ALL kept), so `logits < vals[k-1]` -> -inf reproduces it exactly,
     ties included, even ties extending past KCAP.
  3. top-p on the candidate window. The reference (ascending sort) masks entries with
     cumsum(probs_asc) <= 1-p; in descending terms that is: mask j iff
     (desc-cumsum through j) - prob_j >= p. Probabilities come from softmax over the
     top-k-masked row, which is softmax over the kept candidates.
  4. One scatter writes the p-masked candidates back; sub-threshold positions are already -inf.

EXACTNESS BOUNDARY, stated rather than hidden: (a) if the k-th value's ties extend past KCAP,
step 3's softmax is missing that tail-tie mass and those tail ties cannot be p-masked (the
top-k mask in step 2 still keeps them); (b) at the top-p boundary, equal-probability candidates
are split in whatever order the sort visited them, and topk's tie order need not match the full
sort's. Both need >=2 exactly-equal fp32 logits at the exact cut position. Gated with the
mask-equivalence harness plus seeded sampled completions, not by assertion.

Routed from `apply_top_k_top_p` (patch_topk_composite.py) only when the batch's largest top_k is
known ON THE CPU to be <= KCAP -- `max_top_k` rides SamplingMetadata from the input batch's
existing numpy mirror, so the route costs no GPU sync. Rows with top_k disabled carry
vocab_size there, which routes the whole batch to the old path; k=None likewise.

RADIANCE_TOPK_COMPOSITE=0 restores the previous routing. KCAP via RADIANCE_TOPK_COMPOSITE_KCAP.
"""
import os

import torch

KCAP = int(os.environ.get("RADIANCE_TOPK_COMPOSITE_KCAP", "64"))
_NEG_INF = float("-inf")


def apply_top_k_top_p_composite(logits: torch.Tensor, k: torch.Tensor,
                                p: torch.Tensor | None) -> torch.Tensor:
    vals, idx = torch.topk(logits, KCAP, dim=-1)          # [B, KCAP] descending
    kk = k.to(torch.long).clamp(1, KCAP)
    thr = vals.gather(1, kk.unsqueeze(1) - 1)             # k-th largest value per row
    logits.masked_fill_(logits < thr, _NEG_INF)           # value threshold: boundary ties kept
    if p is not None:
        kept = vals >= thr
        wvals = torch.where(kept, vals, torch.full_like(vals, _NEG_INF))
        probs = wvals.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        maskp = (cum - probs) >= p.unsqueeze(1)           # ascending-form mask, derived above
        src = torch.where(maskp | ~kept, torch.full_like(vals, _NEG_INF), vals)
        logits.scatter_(1, idx, src)
    return logits
