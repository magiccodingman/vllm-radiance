"""int2 TARGET verify head with an exact rerank, gated on the batch's sampling params.

The decode profile of the shipped DFlash2 build shows exactly one 2.02 ms bf16 GEMM per step --
`Cijk_..._MT16x32x256`, 5.9% of wall, one call per engine step. That is the target's `lm_head`:
124160 x 5120 per rank, 1.18 GiB, and it runs at 613 GB/s, i.e. at the DRAM roofline. As with the
draft head, the only lever is bytes.

radiance_drafthead already has the machinery -- an int2 coarse pass that emits the top KCAND of each
64-wide block, and an exact rerank of the best RERANK of those against the untouched bf16 weight.
This module points it at the target's LogitsProcessor as well. Because DFlash2 shares one lm_head
between the drafter and the target, the packing is literally the same buffers (see _HEAD_CACHE in
radiance_drafthead): arming this costs no additional VRAM.

WHY THIS IS A DIFFERENT RISK CLASS FROM THE DRAFT HEAD, AND WHAT MAKES IT SAFE.

A drafter only chooses what is *proposed*; the target verifies every token, so a bad draft costs
acceptance and cannot change what the model emits. This head IS the target. A token that the coarse
pass fails to surface is a token the model can no longer produce, so the approximation has to be
exact over everything the sampler can actually consume.

`_radiance_topk_only` makes the reranked set the ONLY eligible one (everything else is -inf), so a
row carries exactly RERANK finite, exactly-scored entries. That is sufficient when, and only when,
the sampler's support is a subset of those RERANK tokens:

  * A GREEDY request (temperature 0) is always safe: only the argmax matters, which is this head's
    original guarantee. Note that vLLM DISABLES top_k for greedy requests -- it is meaningless once
    the sample is a max -- so top_k arrives as vocab_size, and a gate written only around top_k
    rejects precisely the traffic that is safest. That mistake sent both arms of a GSM8K 500q A/B
    down the bf16 fallback and produced a clean-looking result that gated nothing.
  * A SAMPLED request needs `top_k <= RERANK // 4`. The same 4x margin the drafter needed:
    selector_top_k=16 was correct at RERANK=64 and lossy at 32. top_k bounds the support, and its
    truncation happens before top_p, so top_p and temperature ride along safely -- both are
    monotonic and operate inside the kept set.
  * `min_p == 0` on the sampled rows. min_p keeps every token within a ratio of the max
    probability, which is a threshold on the FULL row, not a rank cut, so it can admit tokens past
    RERANK. Greedy rows ignore min_p.
  * no logprobs. A logprobs response needs the row's logsumexp over the whole vocabulary; ours is
    the logsumexp of RERANK entries.
  * no grammar bitmask. Structured output masks the full vocabulary to the grammar's allowed set,
    which can be disjoint from our RERANK -- and a row of all -inf is not a distribution.

Anything else falls back to the exact bf16 head for that step. The gate is evaluated per step over
the requests actually in the batch, so one logprobs request slows that step down and changes
nothing about it.

**The rejection sampler is safe under the same condition, and it is worth saying why explicitly.**
It needs the target's probability at each drafted token id, and on rejection it resamples from the
target distribution. Both are taken AFTER top_k truncation, so their support is the top_k set. A
drafted token outside the top_k set has target probability zero in the exact distribution too, so
rejecting it is the correct outcome rather than an artefact -- provided the true top_k tokens are
all present, which is the recall property the 4x margin buys and GSM8K gates.

Off by default (`RADIANCE_VERIFY_HEAD=1`). Requires `RADIANCE_FAST_DRAFT=1`, which is what builds
the int2 packing this reuses.
"""
import os
import sys
import types

import torch

try:
    import radiance_drafthead as _dh
except Exception as e:                  # pragma: no cover
    _dh = None
    sys.stderr.write(f"[radiance.verifyhead] radiance_drafthead unavailable: {e!r}\n")

ENABLED = os.environ.get("RADIANCE_VERIFY_HEAD", "0") == "1"
# Rows above which the gate declines. NON-BINDING BY DEFAULT, deliberately.
#
# This knob was added at 32 because a BetterBench single pass showed conc 8 at -6.2%, and the theory
# fit: the coarse pass is memory-bound at M=16 (372 us on 0.167 GiB) but compute-bound at M=64
# (1218 us), while the bf16 head is flat in M, so the advantage shrinks with the batch. DFlash2
# reaches M=64 in normal serving because its row count is num_reqs x (1 + num_speculative_tokens).
#
# THE REGRESSION WAS NOISE. conc 8 at 16 requests read 573.7 / 538.3 / 613.0 across three runs -- a
# 14% spread. Re-measured at 48 requests, 3 reps per arm (aggregate t/s, mean):
#     bf16 499.6 (494.1/497.9/506.7) | uncapped 505.3 (506.5/502.3/507.2) | capped 509.0
# All three overlap. There is no measured M at which this head loses -- even the isolated figures
# have int2 at 1218 us against bf16's 2002 at M=64. So the cap does not bind, and it is kept only
# because batches wider than 64 rows are untested here (max_num_seqs=8 x SPEC 7 + 1 is the ceiling).
MAX_ROWS = int(os.environ.get("RADIANCE_VERIFY_HEAD_MAX_M", "4096"))

# NO_LOGPROBS sentinel in vllm/v1/worker/gpu/sample/states.py.
_NO_LOGPROBS = -1

_state = {"lp": None, "armed": False, "failed": False, "fast": 0, "slow": 0, "reported": False}


def _find_target_lp(model):
    """The LogitsProcessor that owns the target's lm_head.

    Qwen3.8 loads as Qwen3_5ForConditionalGeneration, which delegates compute_logits to an inner
    Qwen3_5ForCausalLM, so the attribute is one or two levels down and its depth is a property of
    the wrapper rather than of anything we control. Look for the module that holds BOTH an lm_head
    and a logits_processor, which is the pairing compute_logits actually uses.
    """
    for m in [model] + [c for _, c in model.named_children()]:
        lp = getattr(m, "logits_processor", None)
        lm = getattr(m, "lm_head", None)
        if lp is not None and lm is not None:
            return lp, lm
    for _, m in model.named_modules():
        lp = getattr(m, "logits_processor", None)
        lm = getattr(m, "lm_head", None)
        if lp is not None and lm is not None:
            return lp, lm
    return None, None


def _arm(model):
    """Quantise the target head once, and rebind its _apply_head to the gated dispatcher."""
    lp, lm_head = _find_target_lp(model)
    if lp is None:
        _state["failed"] = True
        sys.stderr.write("[radiance.verifyhead] no (lm_head, logits_processor) pair found; off\n")
        return
    w = getattr(lm_head, "weight", None)
    if w is None or w.dim() != 2 or float(w.data.abs().max()) == 0.0:
        _state["failed"] = True
        sys.stderr.write("[radiance.verifyhead] target lm_head not a live 2-D weight; off\n")
        return

    # The exact path we fall back to. Bind it BEFORE _quantize_head_now replaces _apply_head, and
    # take it off the class rather than the instance so it is the untouched implementation.
    exact = types.MethodType(type(lp)._apply_head, lp)

    lp._radiance_topk_only = True
    status = _dh._quantize_head_now(lp, lm_head)
    if not hasattr(lp, "_radiance_wq"):
        _state["failed"] = True
        sys.stderr.write(f"[radiance.verifyhead] quantisation declined: {status}; off\n")
        return

    fast = lp._apply_head                    # the int2 path _quantize_head_now just bound
    lp._radiance_exact_head = exact
    lp._radiance_fast_head = fast
    lp._radiance_fast_ok = False
    lp._apply_head = types.MethodType(_apply_head_gated, lp)
    _state["lp"] = lp
    _state["armed"] = True
    sys.stderr.write(f"[radiance.verifyhead] VERIFY_HEAD: {status} "
                     f"(max top_k {_dh.RERANK // 4}, exact fallback otherwise)\n")
    sys.stderr.flush()


def _apply_head_gated(self, lm_head, hidden_states, embedding_bias=None):
    if getattr(self, "_radiance_fast_ok", False):
        return self._radiance_fast_head(lm_head, hidden_states, embedding_bias)
    return self._radiance_exact_head(lm_head, hidden_states, embedding_bias)


def _batch_is_safe(runner, input_batch, grammar_output) -> bool:
    if grammar_output is not None:
        return False
    sampler = getattr(runner, "sampler", None) or getattr(runner, "rejection_sampler", None)
    ss = getattr(sampler, "sampling_states", None)
    if ss is None:
        return False
    # sampling_states is indexed by req_state_idx, NOT by batch position -- reading [:num_reqs]
    # would test whichever requests happen to occupy the first slots, which is how a logprobs
    # request in slot 7 would silently keep the fast path armed.
    idx = input_batch.idx_mapping_np[: input_batch.num_reqs]
    if idx.size == 0:
        return False
    # Decline a batch too wide for the int2 head to win on. logits_indices is one row per sampled
    # position, which is exactly the M the head is about to be called with.
    li = getattr(input_batch, "logits_indices", None)
    if li is not None and li.shape[0] > MAX_ROWS:
        return False
    try:
        if int(ss.num_logprobs[idx].max()) != _NO_LOGPROBS:
            return False
        # A GREEDY request needs only the argmax, which is this head's original and
        # best-validated guarantee (it matched the bf16 argmax on all 8192 captured inputs, and on
        # 8/8 real completions here). vLLM DISABLES top_k for greedy -- it is meaningless once the
        # sample is a max -- so top_k arrives as vocab_size and a top_k-only gate rejects exactly
        # the traffic that is safest. That is not hypothetical: it silently sent both arms of a
        # GSM8K 500q A/B down the bf16 fallback, so the run gated nothing.
        greedy = ss.temperature.np[idx] == 0.0
        if greedy.all():
            return True
        # Mixed or sampled batches: every sampled request must keep its support inside the
        # reranked set. min_p is a threshold on the full row rather than a rank cut, so it can
        # admit tokens past RERANK; it is irrelevant for the greedy rows, hence the mask.
        sampled = ~greedy
        if int(ss.top_k.np[idx][sampled].max()) > _dh.RERANK // 4:
            return False
        if float(ss.min_p.np[idx][sampled].max()) != 0.0:
            return False
    except Exception:
        return False
    return True


def before_compute_logits(runner, input_batch, grammar_output) -> None:
    """Called from model_runner.sample() immediately before compute_logits."""
    if not ENABLED or _dh is None or _state["failed"]:
        return
    if not _state["armed"]:
        _arm(runner.model)
        if not _state["armed"]:
            return
    ok = _batch_is_safe(runner, input_batch, grammar_output)
    _state["lp"]._radiance_fast_ok = ok
    _state["fast" if ok else "slow"] += 1
    if not _state["reported"] and _state["fast"] + _state["slow"] == 200:
        _state["reported"] = True
        sys.stderr.write(f"[radiance.verifyhead] first 200 steps: {_state['fast']} on the int2 "
                         f"head, {_state['slow']} fell back to bf16\n")
        sys.stderr.flush()
