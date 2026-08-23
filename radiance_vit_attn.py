"""RDNA4 (gfx1201) vision-encoder attention: a native head_dim-72 kernel.

On RDNA4 the CK / aiter flash-attention kernels are unavailable (no gfx12x device code) and torch
SDPA falls back to a non-tiled path that runs at a small fraction of peak. attn_vit_h72_bf16 is a
hand-written HIP kernel that handles the vision tower's head_dim of 72 without padding waste, dense
and non-causal, taking a whole batch of images in one varlen launch.

It is installed as a drop-in for the ViT SDPA per-segment apply. The caller splits q/k/v by
cu_seqlens and applies attention per segment, so per-image and windowed (block-diagonal) attention
is preserved unchanged; only the per-segment dense attention is swapped. Only bf16 head_dim-72 MHA
is handled. Anything else (GQA, other head sizes, fp16, batched >1) falls through to the original
SDPA path, as does any build where the kernel library is absent or switched off.

Active on gfx12x with RADIANCE_USE_R4D on.
"""
import os
import sys

import torch

# RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py): with
# it off this module behaves exactly as it would in an image built without the library.
_USE_R4D = os.environ.get("RADIANCE_USE_R4D", "1") == "1"
_HEAD_DIM = 72

# RTLD_DEEPBIND for the same reason as radiance_r4d_attn: tilelang (through aiter) puts a
# libhip_stub.so into the global symbol scope whose hipLaunchKernel throws, and every HIP symbol
# resolved globally afterwards hits the stub instead of the real runtime.
_dlflags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
try:
    import r4d
except ImportError:
    r4d = None
finally:
    sys.setdlopenflags(_dlflags)

# Which kernel, if any, this build of R4D has for the tower's geometry. Asking is the point: the
# constraints live in the library's registry, so this file cannot hold a stale copy of them, and
# None -- a build without the kernel, or a tower of another shape -- simply leaves the Triton
# kernel below in place. The dtypes in the question are a correctness gate rather than a
# preference: the kernel reads its operands as raw 16-bit words, so fp16 handed to it would be
# read as bf16.
_R4D_NAME = (
    r4d.select("attn_vit", head_dim=_HEAD_DIM, gqa=1, causal=0, q_dtype="bf16", kv_dtype="bf16")
    if (r4d is not None and _USE_R4D)
    else None
)
_R4D_ATTN = getattr(r4d, _R4D_NAME) if _R4D_NAME else None
_HAS_R4D = _R4D_ATTN is not None
_cu_cache: dict = {}


def _cu_single(n: int, device) -> torch.Tensor:
    """[0, n] on device, cached: the ViT calls this once per layer per image."""
    key = (int(n), str(device))
    t = _cu_cache.get(key)
    if t is None:
        t = torch.tensor([0, int(n)], dtype=torch.int32, device=device)
        _cu_cache[key] = t
    return t


def _r4d_72(q, k, v, scale, cu=None, max_seqlen=None):
    """q, k, v: [T, heads, 72] contiguous bf16, one or more segments named by cu. Returns [T, heads, 72]."""
    T, H, _ = q.shape
    o = torch.empty_like(q)
    if cu is None:
        cu = _cu_single(T, q.device)
        max_seqlen = T
    _R4D_ATTN(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), cu.data_ptr(),
              int(cu.numel()) - 1, int(max_seqlen), int(H), _HEAD_DIM, float(scale),
              torch.cuda.current_stream().cuda_stream)
    return o


# set by install() to the original apply_sdpa
_orig_apply_sdpa = None


def apply_flash(q, k, v, scale=None, enable_gqa=False):
    """Drop-in for vit_attn_wrappers.apply_sdpa. q/k/v: [batch, seq, heads, head_dim].
    Takes the head_dim-72 MHA case; everything else falls back to SDPA."""
    if (
        not enable_gqa
        and q.dim() == 4
        and q.shape[-1] == _HEAD_DIM
        and q.dtype is torch.bfloat16
        and q.is_cuda
    ):
        s = scale if scale is not None else _HEAD_DIM ** -0.5
        b = q.shape[0]
        if b == 1:
            out = _r4d_72(q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), s)
            return out.unsqueeze(0)
        outs = [_r4d_72(q[i].contiguous(), k[i].contiguous(), v[i].contiguous(), s)
                for i in range(b)]
        return torch.stack(outs, 0)
    return _orig_apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)


# --- transformers-native vision towers (e.g. Gemma4) -------------------------------------------
# Some multimodal models (Gemma4) build their vision tower as a plain transformers module via
# AutoModel.from_config rather than a vLLM ViT, so it never routes through vit_attn_wrappers above.
# Those towers dispatch attention through transformers' ALL_ATTENTION_FUNCTIONS registry, keyed by
# config._attn_implementation, with already-projected q/k/v in [batch, heads, seq, head_dim] layout.
# A flash implementation is registered there and the head_dim-72 tower is pointed at it. Projections, norms
# and RoPE are left entirely to stock transformers; only the attention math is swapped.


def _r4d_bhsd(q, k, v, scale):
    """q, k, v: [B, H, S, D] (transformers layout). Returns [B, S, H, D] (what the attn interface
    must hand back, i.e. eager's post-transpose layout)."""
    B = q.shape[0]
    outs = [
        _r4d_72(q[b].transpose(0, 1).contiguous(),   # [H,S,D] -> [S,H,D]
                k[b].transpose(0, 1).contiguous(),
                v[b].transpose(0, 1).contiguous(), scale)   # -> [S,H,D]
        for b in range(B)
    ]
    return torch.stack(outs, 0)   # [B, S, H, D]


def radiance_vit_flash_attn(module, query, key, value, attention_mask,
                            dropout=0.0, scaling=None, softcap=None, **kwargs):
    """A transformers ALL_ATTENTION_FUNCTIONS entry. Handles the dense, non-causal, head_dim-72 MHA
    case with the flash kernel; everything else (a real attention_mask for packed/windowed images,
    GQA, softcap, other head sizes/dtypes) falls back to the stock eager reference so behaviour is
    unchanged there."""
    if (
        query.shape[-1] == _HEAD_DIM
        and attention_mask is None
        and softcap is None
        and getattr(module, "num_key_value_groups", 1) == 1
        and query.dtype is torch.bfloat16
        and query.is_cuda
    ):
        s = scaling if scaling is not None else _HEAD_DIM ** -0.5
        return _r4d_bhsd(query, key, value, s), None
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, softcap=softcap, **kwargs)


def install_gemma_vision():
    """Point the Gemma4 vision tower's attention at the flash kernel via the transformers registry.
    The tower is a transformers module (AutoModel.from_config), so it uses ALL_ATTENTION_FUNCTIONS
    keyed by config._attn_implementation. Registers the impl and flips the head_dim-72 tower to it
    on first forward. Projections/RoPE stay in stock transformers. gfx12x only."""
    if not _HAS_R4D:
        return
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            return
    except Exception:
        return
    try:
        from transformers.models.gemma4 import modeling_gemma4 as G
    except Exception:
        return
    if getattr(G, "_radiance_vit_installed", False):
        return
    try:
        G.ALL_ATTENTION_FUNCTIONS.register("radiance_vit_flash", radiance_vit_flash_attn)
    except Exception:
        G.ALL_ATTENTION_FUNCTIONS["radiance_vit_flash"] = radiance_vit_flash_attn
    _orig_forward = G.Gemma4VisionAttention.forward

    def _forward(self, *a, **k):
        # flip only the head-72 tower; leave any other head size on its stock implementation
        if getattr(self, "head_dim", None) == _HEAD_DIM \
                and self.config._attn_implementation != "radiance_vit_flash":
            self.config._attn_implementation = "radiance_vit_flash"
        return _orig_forward(self, *a, **k)

    G.Gemma4VisionAttention.forward = _forward
    G._radiance_vit_installed = True
    sys.stderr.write("[radiance.vit] Gemma4 vision attention installed (head_dim 72)\n")
    sys.stderr.flush()


def install():
    """Swap the ViT SDPA per-segment apply for the head_dim-72 kernel on gfx12x. Covers both
    vLLM-native ViT wrappers and the transformers-native Gemma4 vision tower. Without the kernel
    library nothing installs and the stock SDPA path stays in place."""
    global _orig_apply_sdpa
    if not _HAS_R4D:
        return
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            return
    except Exception:
        return
    try:
        import vllm.v1.attention.ops.vit_attn_wrappers as W
        if not getattr(W, "_radiance_vit_installed", False):
            _orig_apply_sdpa = W.apply_sdpa
            W.apply_sdpa = apply_flash
            W._radiance_vit_installed = True
            sys.stderr.write("[radiance.vit] head_dim-72 attention installed\n")
            sys.stderr.flush()
    except Exception:
        pass
    try:
        install_gemma_vision()
    except Exception as e:
        sys.stderr.write(f"[radiance.vit] Gemma vision install skipped: {e!r}\n")
