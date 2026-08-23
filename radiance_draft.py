#!/usr/bin/env python3
"""RADIANCE dynamic MTP draft-depth controller.

A per-request, per-slot controller for MTP self-speculation. At each draft slot it decides one of:
keep drafting (run another MTP forward), take a verbatim n-gram continuation, or stop and verify -- so
the server drafts deep on repetitive/high-acceptance content (code, JSON) and stays shallow on
prose/chat, and never runs deep serial forwards at concurrency.

Lossless: it only changes how many tokens are drafted (and MTP vs a verbatim copy of earlier text).
Every drafted token verifies identically through the unchanged rejection sampler, so outputs cannot
change -- this is a throughput optimization only.

Rule (evaluated per slot): draft while the running product of the drafter's top-1 confidences stays
>= RADIANCE_DRAFT_TAU; take an n-gram match only when its next token equals the drafter's own top guess
(a free win); otherwise stop and verify. A batch-size schedule caps how many serial MTP forwards run at
a given concurrency; the free verbatim n-gram tail can still fill the remaining draft width.

All hot-path work is on-device (Triton confidence capture + n-gram matcher, radiance_draft_gpu.py). One
tiny per-slot device->host copy (top-1 confidence + the drafted token id) lets the host short-circuit
the draft-forward loop (patch_mtp_loopbreak.py). If the GPU kernels are unavailable the controller stays
out of the way and the server runs stock MTP.

Env (the only knobs):
  RADIANCE_DYNAMIC_DRAFT   1=on (default), 0=off -> byte-identical stock MTP
  RADIANCE_DRAFT_SCHEDULE  "bs:max_depth,..." batch-size MTP-forward ceiling (default 1:8,2:7,4:6,8:5,16:4)
  RADIANCE_DRAFT_TAU       confidence-product stop threshold (default 0.35)
"""
import os
import sys
import numpy as np

# ----- config (the only three knobs) ------------------------------------------
DYNAMIC = os.environ.get("RADIANCE_DYNAMIC_DRAFT", "1") == "1"
# Draft policy, tuned on the BetterBench corpus for the 2-bit head (RADIANCE_FAST_DRAFT=1):
# tau 0.28 with 1:8,2:7,4:6,8:5 measured +5.3% over the policy tuned for a 4-bit head. A cheaper
# draft step lowers the marginal acceptance a draft position has to clear, so the optimum sits
# deeper than it does when drafting is expensive.
TAU = float(os.environ.get("RADIANCE_DRAFT_TAU") or "0.28")
# batch-size MTP-forward ceiling "bs:max_depth,..." (carry-forward). Caps how deep the drafter forwards
# by running batch size, so deep serial drafts do not run at concurrency. The per-slot rule still stops
# earlier within it, and the free n-gram tail is unaffected. Empty string disables the cap.
SCHEDULE = (os.environ.get("RADIANCE_DRAFT_SCHEDULE") or "1:8,2:7,4:6,8:5,16:4").strip()
# Runtime state, not a knob: the shard-local draft path (see _local_draft) is always used, and this
# latches off for the rest of the process if it ever raises, so a single failure degrades to the
# gathered path instead of taking the serve down.
_LOCAL_OK = [True]

_GPU = [None]          # the radiance_draft_gpu kernels module (set at install if it imports)


def _log(msg):
    sys.stderr.write(f"[radiance.draft] {msg}\n")
    sys.stderr.flush()


def _parse_schedule(spec):
    pairs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            bs, d = tok.split(":")
            pairs.append((int(bs), int(d)))
    return sorted(pairs)


_SCHED = _parse_schedule(SCHEDULE)


def _batch_ceil(num_reqs):
    """Max MTP forward depth at the current batch size (carry-forward on the schedule); large when unset."""
    if not _SCHED:
        return 1 << 30
    d = _SCHED[0][1]
    for bs, v in _SCHED:
        if num_reqs >= bs:
            d = v
        else:
            break
    return max(1, d)


# ----- local-vocabulary draft sampling (no full-vocab all-gather) -------------
# The gate needs exactly two numbers per row: the drafted token id and the top-1 softmax
# probability. Getting them the obvious way -- compute_logits() -> argmax -- makes vLLM all-gather
# the whole 248320-wide logit row across the TP group on every draft slot, which the decode profile
# prices at ~374 us per slot. Both numbers are recoverable from per-rank partial reductions instead:
# each rank reduces its own vocabulary shard to (max, sum-exp, argmax), the ranks exchange three
# floats per row, and a cross-rank logsumexp finishes the softmax. That is the same arithmetic on
# the same values, up to floating-point associativity -- not an approximation.
#
# vLLM has its own version of half of this (`use_local_argmax_reduction`), but it only produces the
# token id; the confidence gate would still need the gathered logits, which is why this controller
# switches itself off whenever that flag is set.
def _local_draft(proposer, hidden_states):
    """Returns (draft_token_ids [B] int64, conf [B] float32) without gathering the vocabulary."""
    import torch
    from vllm.distributed import (get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)
    gpu = _GPU[0]
    model = proposer.model
    lp = model.logits_processor
    lm_head = model.lm_head
    local = lp._apply_head(lm_head, hidden_states, None)
    if local.dim() > 2:
        local = local.reshape(-1, local.shape[-1])
    npad = getattr(getattr(lm_head, "shard_indices", None), "num_org_vocab_padding", 0) or 0
    if npad > 0:
        local[..., -npad:] = float("-inf")      # padding entries must not win the argmax

    B = local.shape[0]
    sc = getattr(proposer, "_radiance_lscratch", None)
    if sc is None or sc[0].shape[0] != B * gpu._NSPLIT:
        sc = proposer._radiance_lscratch = gpu.make_scratch(B, local.device)
    lmax = torch.empty(B, device=local.device)
    lsum = torch.empty(B, device=local.device)
    gpu.capture_local(local, lmax, lsum, sc)
    lidx = local.argmax(dim=-1)

    start = getattr(getattr(lm_head, "shard_indices", None), "org_vocab_start_index", 0) or 0
    tp = get_tensor_model_parallel_world_size()
    if tp == 1:
        return (lidx + start).to(torch.int64), 1.0 / lsum
    stats = torch.stack([lmax, lsum, (lidx + start).to(torch.float32)], dim=-1)   # [B,3]
    g = tensor_model_parallel_all_gather(stats, dim=-1).view(B, tp, 3)
    gm = g[:, :, 0]
    M, which = gm.max(dim=1)
    S = (g[:, :, 1] * torch.exp(gm - M[:, None])).sum(dim=1)
    ids = g[:, :, 2].gather(1, which[:, None]).squeeze(1).to(torch.int64)
    return ids, 1.0 / S


# ----- drafter hooks: per-slot confidence capture + A/B/C decision ------------
def _install_drafter_hooks():
    import torch
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    if getattr(SpecDecodeBaseProposer, "_radiance_wrapped", False):
        return
    orig_greedy = SpecDecodeBaseProposer._greedy_sample
    orig_propose = SpecDecodeBaseProposer.propose

    def greedy_sample(self, hidden_states):
        if not getattr(self, "_radiance_active", False):
            return orig_greedy(self, hidden_states)
        gpu = _GPU[0]
        logits = None
        if _LOCAL_OK[0]:
            try:
                draft_token_ids, conf_dev = _local_draft(self, hidden_states)
            except Exception as e:
                _log(f"local-vocab draft failed, using the gathered path: {e!r}")
                _LOCAL_OK[0] = False
        if not _LOCAL_OK[0]:
            logits = self.model.compute_logits(hidden_states)
            draft_token_ids = logits.argmax(dim=-1)
        # Per-slot decision (confidence-gated depth). After this slot's MTP forward, capture the top-1
        # confidence on-device, decide on the host: 0 = another MTP pass (continue), 1 = take the n-gram
        # (stop), 2 = verify now (stop). Once every request has chosen 1 or 2, set _radiance_stop so the
        # patched loop skips the remaining forwards. One tiny per-slot D2H (conf + drafted token).
        j = self._radiance_slot; self._radiance_slot = j + 1
        B = draft_token_ids.shape[0]
        cont = getattr(self, "_radiance_cont", None)
        if cont is None or cont.shape[0] != B:
            return draft_token_ids                       # matcher didn't run -> full native draft (safe)
        sc = getattr(self, "_radiance_scratch", None)
        if sc is None or sc[0].shape[0] != B * gpu._NSPLIT:
            sc = self._radiance_scratch = gpu.make_scratch(B, draft_token_ids.device)
        # pack conf + drafted token id into one device tensor so the whole slot is a SINGLE D2H
        # (token ids < 2^24 are exact in fp32).
        packed = torch.empty(2, B, device=draft_token_ids.device)
        if logits is None:
            packed[0] = conf_dev
        else:
            gpu.capture_gpu(logits, packed[0], sc)
        packed[1] = draft_token_ids.to(torch.float32)
        arr = packed.cpu().numpy()
        cfn = arr[0]; mtpn = arr[1].astype(np.int64)
        st = self._radiance_gate
        if j == 0:
            st.update(cum=np.ones(B, np.float32), stopped=np.zeros(B, bool), allagree=np.ones(B, bool),
                      sslot=np.full(B, self._radiance_nspec, np.int64), saction=np.full(B, 2, np.int64))
        clen = self._radiance_clen_cpu; contc = self._radiance_cont_cpu
        agree = (((j < clen) & (contc[:, j] == mtpn)) if j < contc.shape[1] else np.zeros(B, bool))
        # free-win n-gram -> take it (1); else draft on while cumulative confidence holds, else verify (2)
        action = np.where(agree, 1, np.where(st["cum"] >= TAU, 0, 2)).astype(np.int64)
        newly = (~st["stopped"]) & (action != 0)
        # action at slot j is the decision FOR slot j: keep MTP[:j], then the n-gram tail from j on when
        # chosen. slot j's own forward ran only to produce this decision's confidence.
        st["sslot"] = np.where(newly, j, st["sslot"])
        st["saction"] = np.where(newly, action, st["saction"])
        st["stopped"] = st["stopped"] | (action != 0)
        st["cum"] = st["cum"] * cfn
        # running MTP<->n-gram agreement over the drafted prefix (frozen once a request stops)
        st["allagree"] = st["allagree"] & (agree | st["stopped"])
        fc = self._radiance_fwd_cap
        if fc and (j + 1) >= fc:
            # forward cap hit: stop the remaining MTP forwards, but a request whose MTP has matched the
            # n-gram the whole way keeps drafting for FREE via the verbatim tail (up to full width).
            # Others just verify their MTP prefix. Only the forwards are capped.
            cap_new = ~st["stopped"]
            st["sslot"] = np.where(cap_new, j + 1, st["sslot"])
            st["saction"] = np.where(cap_new & st["allagree"], 1, st["saction"])
            st["stopped"] = st["stopped"] | cap_new
        if st["stopped"].all():
            self._radiance_stop = True
        return draft_token_ids

    def propose(self, num_speculative_tokens, *args, **kwargs):
        bc = getattr(self, "_radiance_batch_ceil", 0)
        # gate only on the standard full-vocab argmax path the capture kernel assumes
        skip = self.use_local_argmax_reduction or self.use_heterogeneous_vocab
        if skip:
            self._radiance_active = False
            if 0 < bc < num_speculative_tokens:          # still honor the concurrency cap by clamping
                num_speculative_tokens = bc
            return orig_propose(self, num_speculative_tokens, *args, **kwargs)
        # Keep num_speculative_tokens at full width so vLLM allocates the full draft; cap the MTP FORWARD
        # count via the loop-break instead (fwd_cap = batch_ceil), so the free n-gram tail can still fill
        # the remaining draft positions at concurrency instead of being truncated.
        self._radiance_fwd_cap = bc if (0 < bc < num_speculative_tokens) else 0
        self._radiance_active = True
        self._radiance_slot = 0
        self._radiance_stop = False
        self._radiance_nspec = num_speculative_tokens
        self._radiance_gate = {}                          # per-slot gating state (filled at j==0)
        return orig_propose(self, num_speculative_tokens, *args, **kwargs)

    SpecDecodeBaseProposer._greedy_sample = greedy_sample
    SpecDecodeBaseProposer.propose = propose
    SpecDecodeBaseProposer._radiance_wrapped = True


# ----- runner wrap: run the matcher, then assemble the gated draft ------------
def _install_runner_wrap():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_radiance_draft_wrapped", False):
        return
    orig = GPUModelRunner.propose_draft_token_ids

    def wrapped(self, *args, **kwargs):
        try:
            self.drafter._radiance_batch_ceil = _batch_ceil(len(self.input_batch.req_ids))
        except Exception:
            pass
        try:
            _prepare_match_gpu(self)          # ctx mirror + matcher -> drafter._radiance_cont/clen
        except Exception as e:
            _log(f"gpu match prep failed (native draft used): {e!r}")
            self.drafter._radiance_cont = None
        draft = orig(self, *args, **kwargs)
        try:
            return _postprocess_gpu(self, draft)
        except Exception as e:
            _log(f"postprocess error (native draft used): {e!r}")
            return draft

    GPUModelRunner.propose_draft_token_ids = wrapped
    GPUModelRunner._radiance_draft_wrapped = True


def _prepare_match_gpu(runner):
    """Update the GPU context mirror with newly generated tokens, then run the on-device matcher.
    Stores cont (device) + cont_cpu/clen_cpu (host) on the drafter for the per-slot capture and the
    draft assembly. The mirror append is O(new tokens) per request, the only host->device movement."""
    import torch
    gpu = _GPU[0]
    ib = runner.input_batch
    req_ids = ib.req_ids
    d = runner.drafter
    d._radiance_cont = None
    B = len(req_ids)
    if B == 0:
        return
    dev = runner.device
    src = ib.token_ids_cpu_tensor            # cpu int tensor [max_reqs, max_len]
    nts = ib.num_tokens_no_spec              # numpy [max_reqs]
    ctxg = getattr(runner, "_radiance_ctx_gpu", None)
    if ctxg is None:
        ctxg = runner._radiance_ctx_gpu = torch.zeros(src.shape[0], src.shape[1], dtype=torch.int32, device=dev)
        runner._radiance_seen = {}
    seen = runner._radiance_seen
    n_list = []
    for i, rid in enumerate(req_ids):
        n = int(nts[i]); n_list.append(n)
        # `seen` records (row, tokens_mirrored). The ROW matters: vLLM reuses and reshuffles
        # input-batch slots as requests come and go, so a request can land on a row that still
        # holds the previous occupant's tokens. Keyed by request id alone, the stale tail stayed
        # and the matcher searched another request's text.
        prev = seen.get(rid)
        s = prev[1] if (prev is not None and prev[0] == i) else 0
        if n > s:
            # NOT non_blocking: the source is the temporary from .to(int32), freed as soon as this
            # statement ends. An async copy from pageable memory can still be reading it, which put
            # nondeterministic garbage in the mirror.
            ctxg[i, s:n].copy_(src[i, s:n].to(torch.int32))
        seen[rid] = (i, n)
    live = set(req_ids)
    for rid in [r for r in seen if r not in live]:
        seen.pop(rid, None)
    nmax = max(n_list)
    if nmax < 3:
        return
    n_arr = torch.tensor(n_list, dtype=torch.int32, device=dev)
    cont, clen = gpu.match_gpu(ctxg[:B], n_arr, runner.num_spec_tokens, nmax)
    d._radiance_cont = cont                       # GPU, for the per-slot capture path's agree check
    d._radiance_cont_cpu = cont.cpu().numpy()     # host copies for the per-slot decision (one sync)
    d._radiance_clen_cpu = clen.cpu().numpy()


def _postprocess_gpu(runner, draft):
    """Assemble the final draft from the per-slot decisions. The patched loop already ran only the
    forwards up to the batch's stop point; here each request is trimmed to its own stop slot and given the
    verbatim n-gram tail where it chose one. Returns a ragged list[list[int]] (native draft format), so
    vLLM verifies exactly the tokens the controller kept."""
    import torch
    d = runner.drafter
    st = getattr(d, "_radiance_gate", None)
    cont_cpu = getattr(d, "_radiance_cont_cpu", None)
    if not st or "sslot" not in st or cont_cpu is None:
        return draft
    if torch.is_tensor(draft) and draft.dim() == 2:
        mtp = draft.int().cpu().numpy(); B = mtp.shape[0]
        row = lambda i: mtp[i]
    elif isinstance(draft, list):
        B = len(draft)
        row = lambda i: draft[i]
    else:
        return draft
    if st["sslot"].shape[0] != B:
        return draft
    clen = d._radiance_clen_cpu; nspec = d._radiance_nspec
    sslot = st["sslot"]; saction = st["saction"]
    out = []
    for i in range(B):
        r = row(i)
        k = int(min(sslot[i], len(r)))
        di = [int(x) for x in r[:k]]
        if saction[i] == 1 and k < int(clen[i]):       # append the verbatim n-gram continuation
            di += [int(x) for x in cont_cpu[i, k:int(clen[i])]]
        if not di:
            di = [int(r[0])] if len(r) else []
        out.append(di[:nspec])
    return out


# ----- entry ------------------------------------------------------------------
def install():
    """Entry from radiance_kernels.install_all(). Env-gated on RADIANCE_DYNAMIC_DRAFT."""
    if not DYNAMIC:
        _log("RADIANCE_DYNAMIC_DRAFT=OFF (stock MTP)")
        return
    try:
        import radiance_draft_gpu as gpu
        _GPU[0] = gpu
    except Exception as e:
        _log(f"GPU kernels unavailable ({e!r}) -> stock MTP")
        return
    try:
        _install_drafter_hooks()
        _install_runner_wrap()
        _log(f"RADIANCE_DYNAMIC_DRAFT=ON  controller=policy  tau={TAU}  schedule={SCHEDULE or 'off'}  "
             f"(GPU-resident capture + n-gram matcher; per-slot confidence gate short-circuits the forward loop)")
    except Exception as e:
        _log(f"install failed (dynamic draft disabled): {e!r}")
