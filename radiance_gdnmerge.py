#!/usr/bin/env python3
"""Run the two gated-delta-net input projections as ONE GEMM.

WHY. Ported from llama.cpp rdna-boosts block 08 ("dual-output mmvq fusion"): that patch detects two
matmuls over the SAME activation and has one kernel compute both, writing the second result to a
separate destination. vLLM's GDN layer has exactly that shape -- `in_proj_qkvz` and `in_proj_ba` are
two ColumnParallel linears over the same `hidden_states`
(qwen_gdn_linear_attn.py:408/:419, consumed back to back at :806-807) -- and this model has 48 GDN
layers, so every target forward pays 48 redundant GEMM launches and 48 redundant activation quants.
That is 96 of ~1904 launches per forward, against a 24.9% launch gap at ~4.7 us per launch.

vLLM's own idiom for this is a merged linear (what `qkv_proj` and `gate_up_proj` already are), but
merging at CONSTRUCTION time would mean re-routing checkpoint weight names through a six-shard
weight_loader. This does it after loading instead, which is possible because the MXFP4 layout
concatenates cleanly along N:

    weight        [N, K/2] uint8    -> cat(dim=0)     (two e2m1 codes per byte, N-major)
    weight_scale  [K/32, N] e8m0    -> cat(dim=1)     (already transposed by the kernel's
                                                       process_weights_after_loading)
    radiance_wref [N] folded ref    -> cat(dim=0)     (per-ROW max of the scale, so row-local)

All three are row-local in N, so the merged tensors are bit-identical to what a natively merged
layer would have produced -- this changes which kernel launches happen, not what they compute. The
per-row masking of a partial N tile is already exercised (in_proj_ba is the N=48 shape
layer_is_supported's docstring calls out), and N1+N2 stays a multiple of 16.

Both the default row-major layout and optional WPERM fragment layout are valid: WPERM is tile-local
along N, and both source widths are multiples of a complete 16-row output tile.

Gated by RADIANCE_GDN_MERGE_INPROJ (default 1 in the image, disable for an A/B control).
Called once from GPUModelRunner.load_model, right after the main model is loaded and its
process_weights_after_loading has run (so weight_scale is already [K/32, N] and
radiance_wref is folded) and before the drafter loads or any graph is captured.
patch_gdn_merge_inproj.py installs that call site.
"""
import os
import sys

ENABLED = os.environ.get("RADIANCE_GDN_MERGE_INPROJ", "0") == "1"


def _log(msg):
    sys.stderr.write(f"[radiance.gdnmerge] {msg}\n")
    sys.stderr.flush()


def _merged_forward_hip(self, hidden_states):
    """forward_hip with one projection GEMM instead of two.

    Mirrors the stock body (qwen_gdn_linear_attn.py:798) exactly apart from the projection: the
    merged GEMM produces [T, N1+N2] and the two views are slices of it, so `qwen_gdn_attention_core`
    receives the same two tensors it always did.
    """
    import torch

    num_tokens = hidden_states.size(0)
    merged = torch.ops.radiance.mxfp4_linear(
        hidden_states, self._rad_w, self._rad_ws, self._rad_wref)
    projected_states_qkvz = merged[:, : self._rad_n1].view(num_tokens, -1)
    projected_states_ba = merged[:, self._rad_n1 :].view(num_tokens, -1)

    core_attn_out = torch.empty(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)
    z = torch.empty(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=projected_states_qkvz.dtype, device=projected_states_qkvz.device)

    torch.ops.vllm.qwen_gdn_attention_core(
        projected_states_qkvz, projected_states_ba, z, core_attn_out,
        layer_name=self._rad_layer_name, use_aiter=True)

    return self._output_projection(core_attn_out, z)


def _merge_one(mod) -> bool:
    """Build the merged weights on one GDN module and swap its forward. True if merged."""
    import torch

    qkvz, ba = getattr(mod, "in_proj_qkvz", None), getattr(mod, "in_proj_ba", None)
    if qkvz is None or ba is None:
        return False
    # Only merge when BOTH sides are actually served by the radiance kernel. A layer that fell back
    # to aiter reads a different weight layout, and merging would hand aiter a tensor it cannot
    # interpret -- the same trap layer_is_supported's docstring describes.
    if not (getattr(qkvz, "radiance_w4a8_ok", False) and getattr(ba, "radiance_w4a8_ok", False)):
        return False
    if qkvz.weight.shape[1] != ba.weight.shape[1]:          # same K
        return False
    n1, n2 = int(qkvz.weight.shape[0]), int(ba.weight.shape[0])
    if n1 % 16 or n2 % 16:
        return False

    mod._rad_w = torch.nn.Parameter(
        torch.cat([qkvz.weight.data, ba.weight.data], dim=0), requires_grad=False)
    mod._rad_ws = torch.nn.Parameter(
        torch.cat([qkvz.weight_scale.data, ba.weight_scale.data], dim=1), requires_grad=False)
    mod._rad_wref = torch.nn.Parameter(
        torch.cat([qkvz.radiance_wref.data, ba.radiance_wref.data], dim=0), requires_grad=False)
    mod._rad_n1 = n1
    # The stock forward re-encodes the layer name on every step; it is constant, so resolve once.
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _m
    mod._rad_layer_name = _m._encode_layer_name(mod.prefix)

    # Release the originals: the merged copy duplicates ~1.07 GB across 48 layers at TP=2, which
    # would come straight out of the 17.29 GiB KV budget.
    empty = torch.empty(0, dtype=qkvz.weight.dtype, device=qkvz.weight.device)
    for lin in (qkvz, ba):
        lin.weight = torch.nn.Parameter(empty, requires_grad=False)
        lin.weight_scale = torch.nn.Parameter(empty.clone(), requires_grad=False)

    import types
    mod.forward_hip = types.MethodType(_merged_forward_hip, mod)
    mod._forward_method = mod.forward_hip
    return True


def merge_model(model) -> None:
    """Merge every GDN layer's two input projections. Best-effort: never blocks the serve.

    The same post-load point initializes the fused GDN barrier before graph capture."""
    try:
        import radiance_gdn
        radiance_gdn.init_fused_counter()           # before any CUDA-graph capture
    except Exception as e:                          # noqa: BLE001
        _log(f"gdn fused counter init failed: {e!r}")
    if not ENABLED:
        return
    # Fragment order (RADIANCE_MXFP4_WPERM=1) is fine to merge: permute_w is tile-local along N
    # (16-row blocks, outermost axis is the n-tile), so cat(permute(A), permute(B)) is exactly
    # permute(cat(A, B)) whenever both N are multiples of 16 -- which _merge_one already gates.
    # This used to refuse as unmeasured; measured equivalent 2026-08-29 (tier7-era serving A/B).
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except Exception as e:                              # noqa: BLE001
        _log(f"not a GDN model, skipping: {e!r}")
        return

    n = skipped = 0
    for m in model.modules():
        if not isinstance(m, QwenGatedDeltaNetAttention):
            continue
        try:
            if _merge_one(m):
                n += 1
            else:
                skipped += 1
        except Exception as e:                          # noqa: BLE001
            skipped += 1
            _log(f"merge failed on {getattr(m, 'prefix', '?')}, left unmerged: {e!r}")
    _log(f"merged {n} GDN layers ({2 * n} launches/forward removed), {skipped} left unmerged")
