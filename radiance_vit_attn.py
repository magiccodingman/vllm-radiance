"""RDNA4 (gfx1201) vision-encoder attention: a native head_dim-72 Triton flash-attention kernel.

On RDNA4 the CK / aiter flash-attention kernels are unavailable (no gfx12x device code), and torch
SDPA falls back to a non-tiled path that runs at a small fraction of peak. This module provides a
dense, non-causal, bf16/fp16 flash-attention kernel that handles the vision tower's head_dim of 72
without padding waste (a 64 + 16 split, the 16-block masked to the real 8 dims so every tl.dot has
dims >= 16), running ~1.5-2x faster than SDPA at the ViT's sequence lengths.

It is installed as a drop-in for the ViT SDPA per-segment apply. The caller splits q/k/v by
cu_seqlens and applies attention per segment, so per-image / windowed (block-diagonal) attention is
preserved unchanged; this only swaps the per-segment dense attention. Only bf16/fp16 head_dim-72 MHA
is handled; anything else (GQA, other head sizes, batched >1) falls back to the original SDPA path.

Gated by RADIANCE_VIT_FLASH (default on) and only active on gfx12x.
"""
import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("RADIANCE_VIT_FLASH", "1") == "1"
_HEAD_DIM = 72


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _vit_flash_kernel(Q, K, V, O, scale, N, H,
                      sqm, sqh, sqd, skm, skh, skd, svm, svh, svd, som, soh, sod,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    d0 = tl.arange(0, 64)                 # head dims 0..63
    d1 = tl.arange(0, 16)                 # head dims 64..79 (only 64..71 real)
    d1v = (64 + d1) < 72
    m_mask = offs_m < N

    qb = Q + pid_h * sqh
    q0 = tl.load(qb + offs_m[:, None] * sqm + d0[None, :] * sqd, mask=m_mask[:, None], other=0.0)
    q1 = tl.load(qb + offs_m[:, None] * sqm + (64 + d1)[None, :] * sqd,
                 mask=m_mask[:, None] & d1v[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tl.zeros([BLOCK_M, 64], tl.float32)
    acc1 = tl.zeros([BLOCK_M, 16], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n_idx = start_n + offs_n
        n_mask = n_idx < N
        kb = K + pid_h * skh
        k0 = tl.load(kb + n_idx[:, None] * skm + d0[None, :] * skd, mask=n_mask[:, None], other=0.0)
        k1 = tl.load(kb + n_idx[:, None] * skm + (64 + d1)[None, :] * skd,
                     mask=n_mask[:, None] & d1v[None, :], other=0.0)
        qk = (tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))) * scale
        qk = tl.where(n_mask[None, :], qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        vb = V + pid_h * svh
        v0 = tl.load(vb + n_idx[:, None] * svm + d0[None, :] * svd, mask=n_mask[:, None], other=0.0)
        v1 = tl.load(vb + n_idx[:, None] * svm + (64 + d1)[None, :] * svd,
                     mask=n_mask[:, None] & d1v[None, :], other=0.0)
        pb = p.to(v0.dtype)
        acc0 = acc0 * alpha[:, None] + tl.dot(pb, v0)
        acc1 = acc1 * alpha[:, None] + tl.dot(pb, v1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc0 = acc0 / l_i[:, None]
    acc1 = acc1 / l_i[:, None]
    ob = O + pid_h * soh
    tl.store(ob + offs_m[:, None] * som + d0[None, :] * sod, acc0.to(O.dtype.element_ty), mask=m_mask[:, None])
    tl.store(ob + offs_m[:, None] * som + (64 + d1)[None, :] * sod, acc1.to(O.dtype.element_ty),
             mask=m_mask[:, None] & d1v[None, :])


def _flash_72(q, k, v, scale):
    """q, k, v: [N, H, 72] (contiguous), returns [N, H, 72]."""
    N, H, _ = q.shape
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), H)
    _vit_flash_kernel[grid](q, k, v, o, scale, N, H,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            o.stride(0), o.stride(1), o.stride(2))
    return o


# set by install() to the original apply_sdpa
_orig_apply_sdpa = None


def apply_flash(q, k, v, scale=None, enable_gqa=False):
    """Drop-in for vit_attn_wrappers.apply_sdpa. q/k/v: [batch, seq, heads, head_dim].
    Uses the head_dim-72 flash kernel for the ViT MHA case; falls back to SDPA otherwise."""
    if (
        not enable_gqa
        and q.dim() == 4
        and q.shape[-1] == _HEAD_DIM
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.is_cuda
    ):
        s = scale if scale is not None else _HEAD_DIM ** -0.5
        b = q.shape[0]
        if b == 1:
            out = _flash_72(q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), s)
            return out.unsqueeze(0)
        outs = [_flash_72(q[i].contiguous(), k[i].contiguous(), v[i].contiguous(), s) for i in range(b)]
        return torch.stack(outs, 0)
    return _orig_apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)


def install():
    """Swap the ViT SDPA per-segment apply for the head_dim-72 flash kernel on gfx12x."""
    global _orig_apply_sdpa
    if not _ENABLED:
        return
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            return
    except Exception:
        return
    try:
        import vllm.v1.attention.ops.vit_attn_wrappers as W
    except Exception:
        return
    if getattr(W, "_radiance_vit_installed", False):
        return
    _orig_apply_sdpa = W.apply_sdpa
    W.apply_sdpa = apply_flash
    W._radiance_vit_installed = True
