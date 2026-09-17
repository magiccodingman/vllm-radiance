"""Experimental INT2 target head with BF16-weight reranking and full-head fallback.

The default target method selects 256 candidates globally from the complete
INT2 score row. RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=128 selects a smaller global
shortlist; setting 0 restores block selection with the sampled capacity gate
top_k <= min(RERANK // 4, KCAND). That gate is necessary, not a recall proof.

Global candidate selection removes the per-tile
cap. Every selected token is rescored using the original BF16 weights; the
other logits become -inf. The drafter keeps its existing block-8 shortlist.
The global path is limited to TP1, BF16, and at most 32 target rows per call.

Both paths are approximate. Greedy winner recall is not guaranteed, and
reranking against BF16 weights does not imply bitwise equality to the full
BF16 GEMM. A real 61K-token TP1 replay retained the full top-20 in 649/650
rows with global-256 and also observed retained-logit differences. This is
not a completeness certificate or a distribution-preserving target head.

Grammar, logprobs, unsupported sampled top-k/min-p and wide batches use the
full head. Global selection additionally declines biases, penalties, bad-word
masks, thinking-budget interventions and unsupported layouts. No certificate
failure can be detected by this implementation: approximate recall misses can
still pass the global path. Use RADIANCE_VERIFY_HEAD=0 for the full target head
on every request.

RADIANCE_VERIFY_HEAD=1 requires RADIANCE_FAST_DRAFT=1 to build the shared INT2
packing. Setting GLOBAL_TOPK alone does not enable target-head optimization.
See docs/VERIFY_HEAD_GLOBAL_TOPK.md for measurements, scope and configuration.
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
GLOBAL_TOPK = int(os.environ.get("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", "256"))
if GLOBAL_TOPK not in (0, 128, 256):
    raise ValueError("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK must be 0, 128 or 256")
_GLOBAL_MAX_ROWS = 32
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

    fast = (types.MethodType(_apply_head_global, lp) if GLOBAL_TOPK
            else lp._apply_head)             # target-only; drafter binding is untouched
    lp._radiance_exact_head = exact
    lp._radiance_fast_head = fast
    lp._radiance_fast_ok = False
    lp._apply_head = types.MethodType(_apply_head_gated, lp)
    _state["lp"] = lp
    _state["armed"] = True
    selection = f"global-{GLOBAL_TOPK} approximate TP1" if GLOBAL_TOPK else "block shortlist"
    sys.stderr.write(f"[radiance.verifyhead] VERIFY_HEAD: {status}; target {selection} "
                     f"(max top_k {_sampled_top_k_limit()}, full-head fallback otherwise)\n")
    sys.stderr.flush()


def _apply_head_gated(self, lm_head, hidden_states, embedding_bias=None):
    if getattr(self, "_radiance_fast_ok", False):
        return self._radiance_fast_head(lm_head, hidden_states, embedding_bias)
    return self._radiance_exact_head(lm_head, hidden_states, embedding_bias)


def _sampled_top_k_limit():
    # Retain the existing empirical 4x margin. It is not a recall certificate.
    return GLOBAL_TOPK // 4 if GLOBAL_TOPK else min(_dh.RERANK // 4, _dh.KCAND)


def _apply_head_global(self, lm_head, hidden_states, embedding_bias=None):
    """Full-row INT2 top-N and BF16 rerank, with no per-vocabulary-tile quota."""
    weight = lm_head.weight
    if (embedding_bias is not None or getattr(lm_head, "tp_size", None) != 1
            or hidden_states.dim() != 2 or weight.dim() != 2
            or hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not hidden_states.is_cuda or weight.device != hidden_states.device
            or weight.stride(1) != 1 or hidden_states.shape[-1] != weight.shape[-1]
            or hidden_states.shape[-1] % 512 != 0
            or not 0 < hidden_states.shape[0] <= min(MAX_ROWS, _GLOBAL_MAX_ROWS)
            or weight.shape[0] <= GLOBAL_TOPK
            or getattr(self, "head_dtype", None) not in (None, torch.bfloat16)):
        return self._radiance_exact_head(lm_head, hidden_states, embedding_bias)

    rows, width = hidden_states.shape
    padded = _dh._pow2_at_least(rows)
    x = hidden_states
    if padded != rows:
        x = torch.cat((x, x.new_zeros(padded - rows, width)))
    x = x.contiguous()
    vocab, blocks = self._radiance_n, self._radiance_nblk
    sums = x.reshape(padded, width // _dh.GROUP, _dh.GROUP).float().sum(-1).contiguous()
    logits = torch.empty(padded, vocab, dtype=torch.bfloat16, device=x.device)
    unused_scores = torch.empty(1, dtype=torch.float32, device=x.device)
    unused_ids = torch.empty(1, dtype=torch.int32, device=x.device)
    # KC=0 removes the tile's max/mask emission loop. The complete BF16 coarse
    # row remains available for global selection. No draft setting is changed.
    _dh._draft_head_int2[(blocks,)](
        x, sums, self._radiance_wq, self._radiance_scale, self._radiance_zs,
        logits, unused_scores, unused_ids, width, vocab,
        self._radiance_wq.stride(0), self._radiance_scale.stride(0), sums.stride(0),
        blocks, 0, G=_dh.GROUP, BLOCK_M=padded, BLOCK_N=_dh.BLOCK_N,
        **_dh._cfg_for(padded),
    )
    ids = logits.topk(GLOBAL_TOPK, dim=-1).indices.to(torch.int32).contiguous()
    rescored = torch.empty(padded, GLOBAL_TOPK, dtype=torch.float32, device=x.device)
    _dh._rerank_exact[(padded, GLOBAL_TOPK)](
        x, weight, ids, rescored, width, weight.stride(0),
        R=GLOBAL_TOPK, BLOCK_K=512, num_warps=4,
    )
    logits.fill_(-float("inf"))
    logits.scatter_(1, ids.long(), rescored.to(torch.bfloat16))
    return logits[:rows]


def _global_processors_supported(sampler, idx):
    """Reject transformations that can promote tokens outside the shortlist."""
    try:
        if (sampler.penalties_state.use_penalty[idx].any()
                or sampler.logit_bias_state.use_logit_bias[idx].any()
                or (sampler.bad_words_state.num_bad_words.np[idx] != 0).any()
                or (sampler.logprob_token_ids_state.num_token_ids.np[idx] != 0).any()):
            return False
        thinking = sampler.thinking_budget_state
        if thinking.enabled and thinking.use_thinking_budget[idx].any():
            return False
        if getattr(sampler, "trace_replay_state", None) is not None:
            return False
    except (AttributeError, IndexError, TypeError):
        return False  # Unknown sampler layouts must not silently use this path.
    return True


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
    limit = min(MAX_ROWS, _GLOBAL_MAX_ROWS) if GLOBAL_TOPK else MAX_ROWS
    if li is not None and li.shape[0] > limit:
        return False
    try:
        if int(ss.num_logprobs[idx].max()) != _NO_LOGPROBS:
            return False
        if GLOBAL_TOPK and not _global_processors_supported(sampler, idx):
            return False
        # vLLM disables top_k for greedy requests. Preserve greedy dispatch,
        # without treating observed argmax recall as a mathematical guarantee.
        greedy = ss.temperature.np[idx] == 0.0
        if greedy.all():
            return True
        # Mixed or sampled batches: every sampled request must keep its support inside the
        # reranked set. min_p is a threshold on the full row rather than a rank cut, so it can
        # admit tokens past RERANK; it is irrelevant for the greedy rows, hence the mask.
        sampled = ~greedy
        # Reranking cannot recover tokens already discarded by a per-block shortlist.
        # This capacity check is necessary; it does not prove approximate-head recall.
        if int(ss.top_k.np[idx][sampled].max()) > _sampled_top_k_limit():
            return False
        if GLOBAL_TOPK and int(ss.top_k.np[idx][sampled].min()) <= 0:
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
