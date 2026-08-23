#!/usr/bin/env python3
"""GPU kernels for the RADIANCE draft controller (Triton, gfx1201).

Two device kernels feed the per-slot decision that gates the MTP draft loop, keeping its inputs off the
full-vocab softmax path:

  capture_gpu : per draft slot, one split-V online logsumexp over the vocab logits -> the top-1
                confidence, with NO full-vocab softmax and NO topk. Result stays on device. The
                confidence is the softmax maximum, exp(max - M) / sum(exp(l - M)) = 1 / S; bit-exact vs
                the torch reduction.

  match_gpu   : batched longest-suffix n-gram match over a GPU context mirror -> the verbatim
                continuation tokens and their count, all on device, full context (no window). Bit
                identical to a CPU longest-suffix matcher.

The tiny per-slot threshold decision runs on the host (top-1 confidence + the drafted token id, one
coalesced copy per slot), which is what lets the controller short-circuit the host-launched forward
loop. Both kernels compute exactly the quantities the policy uses, so drafting output is unchanged."""
import torch
import triton
import triton.language as tl

_MAXL, _MIN, _SH = 24, 3, 20      # n-gram match bounds; key packs (matchlen << _SH | end_pos)


@triton.jit
def _match_scan(ctx, nA, ML, key, MAXL: tl.constexpr, MIN: tl.constexpr, SH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)                                   # request row in the context mirror
    n = tl.load(nA + i); row = ctx + i * ML
    q = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    alive = q < (n - 1); L = tl.zeros((BLOCK,), tl.int32)  # longest backward suffix match at end-pos q
    for k in range(MAXL):
        sk = n - 1 - k                                     # suffix position; guard >=0 for short contexts (n<MAXL)
        sfx = tl.load(row + sk, mask=sk >= 0, other=-2)    # else row[negative] page-faults the GPU (short prompts)
        qi = q - k; m = alive & (qi >= 0) & (sk >= 0)
        c = tl.load(row + qi, mask=m, other=-1); alive = m & (c == sfx); L = L + alive.to(tl.int32)
    tl.atomic_max(key + i, tl.max(tl.where(L >= MIN, (L << SH) | q, 0), 0))


@triton.jit
def _match_gather(ctx, nA, ML, key, cont, clen, SH: tl.constexpr, NSPEC: tl.constexpr):
    i = tl.program_id(0); n = tl.load(nA + i); row = ctx + i * ML; k = tl.load(key + i)
    start = (k & ((1 << SH) - 1)) + 1; has = k != 0
    t = tl.arange(0, NSPEC); valid = has & (start + t < n)
    tl.store(cont + i * NSPEC + t, tl.where(valid, tl.load(row + start + t, mask=valid, other=-1), -1))
    tl.store(clen + i, tl.sum(valid.to(tl.int32), 0))


def match_gpu(ctx, n_arr, nspec, nmax):
    """ctx [B, maxlen] int32 GPU context mirror; n_arr [B] int32 lengths; nmax = max length (from the
    host-side num_tokens_no_spec, so no D2H). Returns cont [B,nspec] i32 and cont_len [B], both on
    device. Full-context, no window."""
    B, ML = ctx.shape
    key = torch.zeros(B, dtype=torch.int32, device=ctx.device)
    cont = torch.empty(B * nspec, dtype=torch.int32, device=ctx.device)
    clen = torch.empty(B, dtype=torch.int32, device=ctx.device)
    _match_scan[(B, triton.cdiv(nmax, 512))](ctx, n_arr, ML, key, _MAXL, _MIN, _SH, BLOCK=512)
    _match_gather[(B,)](ctx, n_arr, ML, key, cont, clen, _SH, nspec)
    return cont.view(B, nspec), clen


# capture split-V reduction: NSPLIT chunks of CHUNK, stepped BLOCK at a time (CHUNK/BLOCK static iters)
_NSPLIT, _CBLOCK, _CHUNK = 64, 512, 4096


@triton.jit
def _cap_s1(logits, V, pm, ps, NSPLIT, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); sp = tl.program_id(1)
    row = logits + b * V; base = sp * CHUNK
    m = -float("inf"); s = 0.0
    for k in range(CHUNK // BLOCK):
        idx = base + k * BLOCK + tl.arange(0, BLOCK)
        mask = idx < V
        l = tl.load(row + idx, mask=mask, other=-float("inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(l, 0))
        e = tl.where(mask, tl.exp(l - nm), 0.0)
        sc = tl.where(m == float("-inf"), 0.0, tl.exp(m - nm))
        s = s * sc + tl.sum(e, 0); m = nm
    o = b * NSPLIT + sp
    tl.store(pm + o, m); tl.store(ps + o, s)


@triton.jit
def _cap_s2(logits, V, pm, ps, conf, NSPLIT, BN: tl.constexpr):
    b = tl.program_id(0)
    r = tl.arange(0, BN); mask = r < NSPLIT; o = b * NSPLIT + r
    m = tl.load(pm + o, mask=mask, other=-float("inf"))
    s = tl.load(ps + o, mask=mask, other=0.0)
    M = tl.max(m, 0); sc = tl.exp(m - M)
    S = tl.sum(s * sc, 0)
    tl.store(conf + b, 1.0 / S)                             # top-1 softmax prob = exp(max-M)/S = 1/S


def capture_gpu(logits, out_conf, scratch):
    """logits [B,V] (draft-head). Writes the per-row top-1 confidence into out_conf [B] in-place.
    scratch holds the [B*NSPLIT] partials (reused across slots)."""
    B, V = logits.shape
    pm, ps = scratch
    _cap_s1[(B, _NSPLIT)](logits, V, pm, ps, _NSPLIT, _CHUNK, _CBLOCK)
    _cap_s2[(B,)](logits, V, pm, ps, out_conf, _NSPLIT, 64)


@triton.jit
def _cap_s2_local(pm, ps, om, os_, NSPLIT, BN: tl.constexpr):
    """Same combine as _cap_s2, but emits the shard's (max, sum-exp) instead of a finished
    confidence -- the cross-rank logsumexp finishes it."""
    b = tl.program_id(0)
    r = tl.arange(0, BN); mask = r < NSPLIT; o = b * NSPLIT + r
    m = tl.load(pm + o, mask=mask, other=-float("inf"))
    s = tl.load(ps + o, mask=mask, other=0.0)
    M = tl.max(m, 0); sc = tl.exp(m - M)
    tl.store(om + b, M)
    tl.store(os_ + b, tl.sum(s * sc, 0))


def capture_local(logits, out_max, out_sum, scratch):
    """logits [B,Vlocal] (this rank's vocabulary shard only). Writes the shard's running max into
    out_max [B] and its sum of exp(l - max) into out_sum [B]. Combining these across ranks with a
    logsumexp gives exactly the number capture_gpu returns for the gathered row, while moving three
    floats per row instead of the whole 248320-wide logit row."""
    B, V = logits.shape
    pm, ps = scratch
    _cap_s1[(B, _NSPLIT)](logits, V, pm, ps, _NSPLIT, _CHUNK, _CBLOCK)
    _cap_s2_local[(B,)](pm, ps, out_max, out_sum, _NSPLIT, 64)


def make_scratch(B, device):
    z = lambda: torch.empty(B * _NSPLIT, device=device)
    return z(), z()
