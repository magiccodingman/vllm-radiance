"""2-bit MTP draft head with an exact rerank, behind RADIANCE_FAST_DRAFT.

Off by default, in which case the drafter uses the stock bf16 head that vLLM shares with the target
model (_maybe_share_lm_head does this unconditionally for an MTP drafter, so it costs no extra
memory). That head reads 1.18 GiB/rank on every draft slot and measures ~2002 us per call.
RADIANCE_FAST_DRAFT=1 replaces it with a 2-bit head at ~473 us including the rerank, for 0.167
GiB/rank.

The head is the largest bandwidth consumer in a decode step: per rank it is x[M,5120] @
W[5120,124160], it runs once per draft slot (up to 8 times per engine step) and it is flat in M from
1 to 72, so the only lever is fewer bytes.

Weights are int2 with an asymmetric per-(row, group-of-128) scale. Four properties make that pay,
and narrower weights alone is not one of them: at group 64 the same kernel measures 781 us, slower
than a 4-bit head.

  * Group 128, not 64. The per-group scale and zero-point arithmetic on the [BLOCK_M, BLOCK_N]
    accumulator is the dominant non-memory term, so halving the group count halves it: 781 -> 417 us
    on identical bytes.
  * Quarter-split packing. Byte j carries k = j, K/4+j, K/2+j and 3K/4+j, so all four 2-bit planes
    feed contiguous k ranges and one byte load serves four contiguous dots. A contiguous-4 layout
    (byte j holding k = 4j..4j+3, one dot per group) measures 676 us instead, because four tile rows
    then share a byte and gather rather than stream.
  * Bit-pattern dequant, hoisted. 0x3F80 | ((b << (5-2q)) & 0x60) reads as the bf16 value 1 + v/4,
    one shift and one mask, with no int-to-float convert and no extract-then-reposition; the uint16
    conversion is hoisted out of the quarter loop, one per tile rather than four. 417 -> 350 us. The
    1.0 bias is exact rather than an approximation: the dot returns sum_k x_k + dot(x,v)/4, and the
    kernel already holds sum_k x_k per group for the zero point, so the contribution collapses to
    (4 s)*p - (4 s + z s)*sum_k x_k, the same two accumulator ops against premultiplied scales.
  * The scale is applied to the accumulator, never to the weight tile. Dequantising [G, BLOCK_N]
    elementwise costs ~400 us instead, 8x more elements to touch.

Accuracy comes from reranking rather than from bits. Each program already holds the maximum of its
64 columns, so it emits the top KCAND of them for free; the top RERANK of those are scored exactly
against the bf16 weight (a few hundred KB against the coarse pass's 0.167 GiB) and written back over
the coarse values. On 8192 draft-head inputs captured from a live serve this matches the exact bf16
argmax on every row, against 22 misses for a 4-bit head at KCAND=1.

KCAND is the lever, not RERANK. Selection emits the top K of each block, so at K=1 a winner that
shares a block with a stronger token is never a candidate at any R, and recall saturates. 2 bits
needs K=8; 4 bits is adequate at K=1. Selecting by block max rather than a token-level topk over the
full row is also the faster choice, 65 us against 105.

Model output cannot move as a result: the draft head decides which tokens are proposed, the target
model verifies every proposal with its own untouched bf16 head on a separate LogitsProcessor
instance, and speculative decoding is distribution-preserving, so a worse draft costs acceptance
rather than a different token. mtp.fc is deliberately left alone: the checkpoint lists it in
modules_to_not_convert alongside the norms, gates, lm_head and embed_tokens, and it is worth only
~0.5% of a decode step.
"""
import os
import sys
import types

import torch

try:
    import triton
    import triton.language as tl
except Exception:                       # pragma: no cover - triton always present in the image
    triton = None

# RADIANCE_FAST_DRAFT gates draft-head quantisation entirely.
#   0 (default): nothing here installs. The drafter uses the stock bf16 head -- which vLLM shares
#                with the target model, so it costs no extra memory but reads 1.18 GiB/rank per draft
#                slot and measures ~2002 us per call.
#   1:           2-bit head, 0.167 GiB/rank, ~473 us per call including the rerank. The rerank makes
#                it exact: on 8192 real draft-head inputs it matches the bf16 argmax on every row.
FAST = os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"

GROUP = 128        # weight-quantisation group along K; also the kernel's BLOCK_K
BLOCK_N = 64       # do_bench optimum, and the width of one block-max entry
BITS = 2
RERANK = 32        # candidates scored exactly per row; swept, and KCAND is the recall lever
KCAND = 8          # candidates emitted per block; R caps the final count, K feeds it
# Launch geometry, per M band. The head changes regime across the batch sizes one serve produces:
# at M=16 it is memory-bound (427 GB/s, 68% of the DRAM roofline on its 152 MiB of int2) and at
# M=64 it is compute-bound (67 TF/s against a 207 TF/s bf16 WMMA ceiling, only 130 GB/s). The warp
# count that suits one end is wrong at the other, and the penalty is not symmetric: warps=2 is the
# optimum at M=16 (1.10x) and 0.38x at M=64, while warps=8 is the optimum at M=64 (1.07x) and 0.78x
# at M=32. num_stages>1 regresses everywhere. BLOCK_N=64 wins or ties at every M measured.
# Isolated, 124160 x 5120 (one rank's vocab shard), CUDA-graph window, us:
#     M=16   warps 2 / 4 / 8 = 372.4 / 408.8 / 425.6
#     M=32   warps 2 / 4 / 8 = 660.4 / 540.0 / 690.3
#     M=64   warps 2 / 4 / 8 = 3428.4 / 1309.4 / 1218.7
# M is padded to a power of two >= 16 before it gets here, so the bands are 16, 32, 64.
_CFG_BY_M = ((16, {"num_warps": 2, "num_stages": 1}),
             (48, {"num_warps": 4, "num_stages": 1}),
             (64, {"num_warps": 8, "num_stages": 1}))
_CFG = {"num_warps": 4, "num_stages": 1}    # fallback for an M past the table


def _cfg_for(m):
    for lim, cfg in _CFG_BY_M:
        if m <= lim:
            return cfg
    return _CFG


if triton is not None:

    @triton.jit
    def _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC: tl.constexpr, BLOCK_N: tl.constexpr):
        """Top-KC of this block, by successive max-and-mask.

        The exact winner is always the maximum of its own block, so block maxima carry the rerank
        candidates at 1/BLOCK_N the selection width of a token-level top-R; a token-level topk over
        the full row measures 105 us against 65 for this, so the cheap selection is also the fast
        one. KC matters where R does not: R caps how many candidates are finally rescored, but at
        KC=1 a winner sharing a block with a stronger token is never a candidate at any R.
        """
        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        for c in tl.static_range(KC):
            mx = tl.max(masked, axis=1)
            am = tl.argmax(masked, axis=1)
            tl.store(BM + offs_m * (NBLK * KC) + (pid * KC + c), mx)
            tl.store(BI + offs_m * (NBLK * KC) + (pid * KC + c),
                     (pid * BLOCK_N + am).to(tl.int32))
            masked = tl.where(tl.arange(0, BLOCK_N)[None, :] == am[:, None], float("-inf"), masked)

    @triton.jit
    def _draft_head_int2(X, XS, Wq, S, ZS, Y, BM, BI, K: tl.constexpr, N, stride_wq, stride_s,
                         stride_xs, NBLK, KC: tl.constexpr, G: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """2 bits/weight. Packing splits K into quarters, byte j carrying k = j, K/4+j, K/2+j and
        3K/4+j, so all four planes feed contiguous k ranges and one byte load serves four contiguous
        dots. Group 128 rather than 64 is what makes it pay: the per-group accumulator work is the
        dominant non-memory term, and halving the group count moves this kernel 781 -> 417 us."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            # hoisted out of the quarter loop: it is the same tile each time, so one convert not four
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                # 0x3F80 | ((b << (5-2q)) & 0x60) is exactly 0x3F80 | (((b >> 2q) & 3) << 5): one
                # shift and one mask instead of extract-then-reposition. Reads as bf16 1 + v/4, and
                # the 1.0 bias divides out through the group's sum of x, as in the 4-bit path.
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :]).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.load(XS + offs_m * stride_xs + gi).to(tl.float32)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mask_n[None, :])
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)

    @triton.jit
    def _rerank_exact(X, W, IDX, OUT, K: tl.constexpr, stride_w, R: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        """Exact logit for R candidate rows per draft row, straight off the bf16 weight."""
        m = tl.program_id(0)
        j = tl.program_id(1)
        n = tl.load(IDX + m * R + j)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            acc += (tl.load(X + m * K + offs).to(tl.float32)
                    * tl.load(W + n * stride_w + offs).to(tl.float32))
        tl.store(OUT + m * R + j, tl.sum(acc, axis=0))


def _pow2_at_least(m):
    """tl.arange needs a power-of-two extent, and tl.dot needs at least 16 rows."""
    n = 16
    while n < m:
        n *= 2
    return n


def _apply_head_int2(self, lm_head, hidden_states, embedding_bias):
    """Drop-in for LogitsProcessor._apply_head against the quantised draft head."""
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    m, k = x.shape
    M = _pow2_at_least(m)
    if M != m:
        x = torch.cat([x, x.new_zeros(M - m, k)])
    x = x.contiguous()

    n = self._radiance_n
    nblk = self._radiance_nblk
    ng = k // GROUP
    xs = x.reshape(M, ng, GROUP).float().sum(-1).contiguous()
    y = torch.empty(M, n, dtype=torch.bfloat16, device=x.device)
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    _draft_head_int2[(nblk,)](
        x, xs, self._radiance_wq, self._radiance_scale, self._radiance_zs, y, bm, bi,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))

    w = getattr(lm_head, "weight", None)
    if w is not None and w.dtype in (torch.bfloat16, torch.float16) and w.shape == (n, k):
        idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
        ex = torch.empty(M, RERANK, dtype=torch.float32, device=x.device)
        _rerank_exact[(M, RERANK)](x, w, idx, ex, k, w.stride(0), R=RERANK, BLOCK_K=512,
                                   num_warps=4)
        if getattr(self, "_radiance_topk_only", False):
            # A top-k caller ranks the whole row against itself, so the ~124k entries the rerank
            # did NOT touch are still coarse 2-bit values competing with exact ones -- and a
            # spuriously high coarse entry becomes a candidate. An argmax caller never noticed
            # (the true winner is a reranked block maximum); a top-16 caller sees garbage in
            # every slot the coarse pass over-scored. Make the reranked set the only eligible
            # one, so recall is bounded by RERANK rather than by 2-bit noise.
            y.fill_(float("-inf"))
        y.scatter_(1, idx.long(), ex.to(torch.bfloat16))
    elif not self._radiance_warned:
        # Without the exact weight the coarse pass stands on its own; that is a ~5% top-1 change
        # against bf16, so say so rather than let it pass as the reranked path.
        self._radiance_warned = True
        sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: no bf16 lm_head to rerank against "
                         f"({None if w is None else (tuple(w.shape), w.dtype)}); "
                         "running the coarse 2-bit head unreranked\n")
        sys.stderr.flush()

    if M != m:
        y = y[:m]
    if embedding_bias is not None:
        y = y + embedding_bias
    if self.head_dtype is not None and self.head_dtype != y.dtype:
        y = y.to(self.head_dtype)
    return y.reshape(*hidden_states.shape[:-1], -1)


def _apply_head_lazy(self, lm_head, hidden_states, embedding_bias):
    """First real call quantises, then hands over to the quantised path for good.

    Needed because the weight a drafter scores against may not exist yet when load_weights
    returns; here it is the argument, so it is guaranteed populated.
    """
    if float(lm_head.weight.data.abs().max()) == 0.0:
        # Still empty on a real call: something is wrong, but a coarse head would be silently
        # catastrophic, so fall back to the stock GEMM rather than guess.
        return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
    status = _quantize_head_now(self, lm_head)
    sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD (lazy): {status}\n")
    sys.stderr.flush()
    return self._apply_head(lm_head, hidden_states, embedding_bias)


def _quantize_draft_head(mtp, lp_attr="logits_processor"):
    """Quantise the head a drafter scores with, and rebind that LogitsProcessor's _apply_head.

    lp_attr names which LogitsProcessor to hook. MTP has exactly one; DFlash2 keeps a separate
    `candidate_logits_processor` for candidate generation, and hooking THAT is what confines the
    approximation to the draft path -- the target samples through its own instance and is
    untouched. Both share the same lm_head weight, which is why the bf16 copy has to stay: it is
    the target's, and it is also what the rerank scores against.
    """
    lm_head = getattr(mtp, "lm_head", None)
    lp = getattr(mtp, lp_attr, None)
    if lm_head is None or lp is None or not hasattr(lm_head, "weight"):
        return f"no lm_head/{lp_attr}"
    w = lm_head.weight
    if w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"unsupported draft-head weight {tuple(w.shape)} {w.dtype}"
    lp._radiance_topk_only = lp_attr == "candidate_logits_processor"
    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target's tensor shared in
    # AFTER load_weights returns, so at this point the parameter is still allocated-but-empty.
    # Quantising that yields an all-zero head, and the failure is silent and total: the serve comes
    # up, text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --
    # which reads as a plausible accuracy verdict on the quantisation. Defer instead.
    if float(w.data.abs().max()) == 0.0:
        lp._apply_head = types.MethodType(_apply_head_lazy, lp)
        return "lm_head empty at load_weights (shared in later); quantising on first use"
    return _quantize_head_now(lp, lm_head)


def _quantize_head_now(lp, lm_head):
    w = lm_head.weight
    n, k = w.shape
    if k % (2 * GROUP):
        return f"hidden size {k} not a multiple of {2 * GROUP}"

    # Asymmetric min/max at BITS, quarter-split packing (see the module docstring).
    # Quantise in row chunks: a whole-tensor fp32 intermediate is 2.5 GiB here, and the caching
    # allocator keeps that reservation for the rest of the process, which comes straight out of
    # the VRAM the KV cache could have used.
    per_byte = 8 // BITS
    lv = (1 << BITS) - 1
    packed = torch.empty(n, k // per_byte, dtype=torch.uint8, device=w.device)
    scale = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    zs = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    CH = 8192
    for i in range(0, n, CH):
        j = min(i + CH, n)
        wg = w.data[i:j].float().reshape(j - i, k // GROUP, GROUP)
        lo, hi = wg.amin(dim=2), wg.amax(dim=2)
        sc = ((hi - lo) / lv).clamp(min=1e-8)
        zp = torch.round(-lo / sc).clamp(0, lv)
        q = torch.round(wg / sc[:, :, None] + zp[:, :, None]).clamp(0, lv).to(torch.uint8)
        q = q.reshape(j - i, k)
        if BITS == 2:
            # quarter-split: byte j carries k = j, K/4+j, K/2+j, 3K/4+j
            Q = k // 4
            packed[i:j] = (q[:, :Q] | (q[:, Q:2 * Q] << 2)
                           | (q[:, 2 * Q:3 * Q] << 4) | (q[:, 3 * Q:] << 6))
        else:
            packed[i:j] = q[:, : k // 2] | (q[:, k // 2:] << 4)
        # The kernel's dot returns sum_k x_k + dot(x,v)/16 because every weight carries the bf16
        # 1.0 bias, so s*dot(x,q) - zp*s*sum_k x_k becomes (16 s)*p - (16 s + zp s)*sum_k x_k.
        # Both premultiplied here, which also saves a load in the inner loop.
        # the bf16 bias is 1 + v/(2^m) with m the mantissa slot used, so the premultiplier is
        # 2^m: 16 for the 4-bit path (v << 3), 4 for the 2-bit one (v << 5)
        bias = 4.0 if BITS == 2 else 16.0
        scale[i:j] = (bias * sc).to(torch.bfloat16)
        zs[i:j] = (bias * sc + zp * sc).to(torch.bfloat16)
        del wg, lo, hi, sc, zp, q
    lp._radiance_wq = packed
    lp._radiance_scale = scale
    lp._radiance_zs = zs
    lp._radiance_n = n
    lp._radiance_nblk = (n + BLOCK_N - 1) // BLOCK_N
    lp._radiance_warned = False
    lp._apply_head = types.MethodType(_apply_head_int2, lp)
    torch.cuda.empty_cache()

    stored = (lp._radiance_wq.numel()
              + (lp._radiance_scale.numel() + lp._radiance_zs.numel()) * 2)
    # The bf16 weight is deliberately left in place: the rerank scores against it, and leaving it
    # lets vLLM's _maybe_share_lm_head point the MTP at the target's copy instead of keeping a
    # second one. Blanking it here (as the fp8 head did) defeats that share.
    return (f"draft head ({n}, {k}) bf16 -> int{BITS} g{GROUP} asym "
            f"({stored / 2**30:.2f} GiB/rank), {KCAND} cand/block, rerank top-{RERANK} exact")


def install():
    if not FAST:
        # stock bf16 head; nothing patched, nothing quantised
        return
    if triton is None:
        sys.stderr.write("[radiance] quantised draft head off: no triton\n")
        return

    # (module, class, which LogitsProcessor that drafter scores candidates with)
    targets = []
    for mod_name, cls_name, lp_attr in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP", "logits_processor"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP", "logits_processor"),
        # DFlash2 reaches the head through get_top_k_tokens, which calls _apply_head like
        # everything else. Its lm_head is not in the drafter checkpoint -- it is shared in from
        # the target after load_weights -- so this one always takes the lazy path.
        ("vllm.model_executor.models.qwen3_dflash2", "DFlash2Qwen3ForCausalLM",
         "candidate_logits_processor"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            targets.append((getattr(mod, cls_name), lp_attr))
        except Exception:
            continue
    if not targets:
        sys.stderr.write("[radiance] quantised draft head off: no drafter class found\n")
        return

    for cls, lp_attr in targets:
        if getattr(cls, "_radiance_quant_head_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig, _lp=lp_attr):
            loaded = _orig(self, weights)
            try:
                status = _quantize_draft_head(self, _lp)
            except Exception as e:
                status = f"FAILED, bf16 head kept: {e!r}"
            sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: {status}\n")
            sys.stderr.flush()
            return loaded

        cls.load_weights = wrapped
        cls._radiance_quant_head_wrapped = True
    sys.stderr.write(f"[radiance] int{BITS} draft head armed (RADIANCE_FAST_DRAFT)\n")
    sys.stderr.flush()
