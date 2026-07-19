"""radiance (gfx1201 / RDNA4) custom kernel dispatcher: the single place that picks which kernels vLLM
launches for block-FP8 GEMM, plus the preshuffle-GEMM and tuned unified-attention config hooks. vLLM's
source hooks (installed by patch_radiance_dispatch.py) only delegate here, so routing changes live in
this file instead of a vLLM patch.

The dispatch fns run inside vLLM's torch.compiled region, so they must be dynamo-traceable: plain shape
branches dispatching to registered torch.ops.*, no lru_cache / hasattr / side effects in the hot path.
"""
import os
import sys

import torch

try:
    from vllm.platforms.rocm import on_gfx12x
    GFX12 = bool(on_gfx12x())
except Exception:
    GFX12 = False


# ---- preshuffle GEMM ----
# Shuffle each block-fp8 weight once at load (shuffle_weight + reshape to [N//16, K*16]), transpose the
# act-scale per forward, run AITER's preshuffle kernel. Gated by RADIANCE_PRESHUFFLE=1; when off, weights
# aren't shuffled so the shape-detected route in block_scaled_mm stays a no-op.
PRESHUFFLE = os.environ.get("RADIANCE_PRESHUFFLE", "0") == "1"
try:
    import aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale as _PS
    from aiter.ops.shuffle import shuffle_weight as _shuffle_weight
except Exception:
    _PS = None

_PS_BASE = {"BLOCK_SIZE_K": 128, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1}


def _ps_cfg(M, N, K):
    # BM16 for small-M (decode), BM64 for large-M (prefill). The win is the preshuffle layout itself;
    # split-K and larger BN helped in isolation but were neutral-to-worse end to end.
    return {**_PS_BASE, "BLOCK_SIZE_M": (16 if M <= 32 else 64), "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}


if _PS is not None:
    @torch.library.custom_op("radiance::preshuffle_gemm", mutates_args=())
    def preshuffle_gemm(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                        Bs: torch.Tensor, N: int, K: int) -> torch.Tensor:
        xs_shuf = As.transpose(0, 1).contiguous().view(*As.shape)   # physical transpose, same logical shape
        return _PS.gemm_a8w8_blockscale_preshuffle(
            A, B, xs_shuf, Bs, torch.bfloat16, config=_ps_cfg(A.shape[0], N, K))

    @preshuffle_gemm.register_fake
    def _(A, B, As, Bs, N, K):
        return torch.empty((A.shape[0], N), dtype=torch.bfloat16, device=A.device)


def install_load_hook():
    """Wrap Fp8LinearMethod.process_weights_after_loading to shuffle every block-fp8 weight once at load.
    Idempotent; must run before the model loads."""
    if not PRESHUFFLE or _PS is None:
        return
    import vllm.model_executor.layers.quantization.fp8 as _f8
    if getattr(_f8.Fp8LinearMethod, "_radiance_preshuffle_wrapped", False):
        return
    _orig = _f8.Fp8LinearMethod.process_weights_after_loading

    def _wrapped(self, layer):
        _orig(self, layer)
        try:
            if not getattr(self, "block_quant", False):
                return
            w = getattr(layer, "weight", None)
            ws = getattr(layer, "weight_scale_inv", None)   # block-fp8 scale (deepseek-style name)
            if ws is None:
                ws = getattr(layer, "weight_scale", None)
            if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
                return
            N, Kk = int(w.shape[0]), int(w.shape[1])
            # only genuine 128-block-scaled weights: N,K multiples of 128, weight_scale [N/128, K/128]
            ok = (N % 128 == 0 and Kk % 128 == 0 and ws is not None and ws.dim() == 2
                  and int(ws.shape[0]) == N // 128 and int(ws.shape[1]) == Kk // 128)
            if not ok:
                return
            w_sh = _shuffle_weight(w.data, (16, 16))
            if w_sh.numel() != N * Kk:          # shuffle must preserve element count
                sys.stderr.write(f"[radiance] preshuffle skip N={N} K={Kk}: numel {w_sh.numel()} != {N*Kk}\n"); return
            layer.weight = torch.nn.Parameter(w_sh.reshape(N // 16, Kk * 16).contiguous(), requires_grad=False)
            layer._radiance_preshuffled = (N, Kk)
            layer._radiance_N = N           # true output dim (weight.shape[0] is now N//16)
        except Exception as e:
            sys.stderr.write(f"[radiance] preshuffle skipped a layer: {e!r}\n")

    _f8.Fp8LinearMethod.process_weights_after_loading = _wrapped
    _f8.Fp8LinearMethod._radiance_preshuffle_wrapped = True
    sys.stderr.write("[radiance] preshuffle weight-shuffle-at-load hook installed\n")
    sys.stderr.flush()


def install_attn_config_hook():
    """Override AITER's unified-attention config with the gfx1201-tuned one, per dtype and phase:
       decode fp8 3D   TILE=16 warps4 stages1 waves6  (reduce warps8/stages1/waves2)
       prefill 2D fp8  TILE=16 waves1  (large prefill only)
       prefill 2D bf16 TILE=16 warps4 stages1 waves1
    The bf16 3D-decode config is applied at build time in patch_unified_attention_bf16.py, not here: this
    runtime wrapper is bypassed for that path, and the config is LDS-critical (the aiter default overflows
    the R9700's 64 KiB LDS at head_size 256, OOMing 2-byte KV at cudagraph capture). Gated RADIANCE_ATTN_TUNE=1."""
    if os.environ.get("RADIANCE_ATTN_TUNE", "0") != "1":
        return
    try:
        import aiter.ops.triton.attention.unified_attention as UA
    except Exception:
        return
    if getattr(UA, "_radiance_attn_wrapped", False):
        return
    FP8 = torch.float8_e4m3fn
    _o3, _o2 = UA.select_3d_config, UA.select_2d_config
    TWO_BYTE = (torch.bfloat16, torch.float16)   # includes --kv-cache-dtype auto

    def _s3(*a, **k):   # decode (3D split-KV)
        ac, rc = _o3(*a, **k)
        q_dtype = a[5] if len(a) > 5 else k.get("q_dtype")
        kv_dtype = a[6] if len(a) > 6 else k.get("kv_cache_dtype")
        if q_dtype == FP8 and kv_dtype == FP8:   # fp8 decode; bf16 3D is handled by the build-time patch
            ac["TILE_SIZE"] = 16; ac["num_warps"] = 4; ac["num_stages"] = 1; ac["waves_per_eu"] = 6
            rc["TILE_SIZE"] = 16; rc["num_warps"] = 8; rc["num_stages"] = 1; rc["waves_per_eu"] = 2
        return ac, rc

    def _s2(*a, **k):   # prefill / MTP decode-via-2D
        c = _o2(*a, **k)
        max_seqlen_q = a[4] if len(a) > 4 else k.get("max_seqlen_q", 0)
        kv_dtype = a[9] if len(a) > 9 else k.get("kv_cache_dtype")
        if kv_dtype in TWO_BYTE:                 # bf16/auto KV: warps=4 is decisive on large prefill
            c["TILE_SIZE"] = 16; c["num_warps"] = 4; c["num_stages"] = 1; c["waves_per_eu"] = 1
        elif max_seqlen_q >= 256:                # fp8 large prefill
            c["TILE_SIZE"] = 16; c["waves_per_eu"] = 1
        return c

    UA.select_3d_config = _s3
    UA.select_2d_config = _s2
    UA._radiance_attn_wrapped = True
    sys.stderr.write("[radiance] attn tuned-config override installed\n")
    sys.stderr.flush()


def install_all():
    """Install every gated radiance runtime hook. Called once per process by the vLLM plugin loader,
    after torch/vllm/aiter are imported but before the model loads. Idempotent; each hook is env-gated."""
    try:
        install_load_hook()
    except Exception as e:
        sys.stderr.write(f"[radiance] install_load_hook failed: {e!r}\n")
    try:
        install_attn_config_hook()
    except Exception as e:
        sys.stderr.write(f"[radiance] install_attn_config_hook failed: {e!r}\n")
    try:
        import radiance_allreduce
        radiance_allreduce.install_custom_ar()   # RADIANCE_FAST_REDUCE (default on)
    except Exception as e:
        sys.stderr.write(f"[radiance] install_custom_ar failed: {e!r}\n")
    try:
        import radiance_draft
        radiance_draft.install()                 # RADIANCE_DYNAMIC_DRAFT (default off)
    except Exception as e:
        sys.stderr.write(f"[radiance] radiance_draft install failed: {e!r}\n")


def block_scaled_mm(kernel, A, B, As, Bs):
    """W8A8 block-FP8 GEMM (per-128-block scales) on gfx1201:
      M<=8   AITER split-K Triton GEMM (1.5-1.9x vs generic at small M)
      M>=16  vLLM generic kernel (AITER split-K regresses there)
    Preshuffled weights ([N//16, K*16], shape-detected) take the AITER preshuffle route."""
    M = A.shape[0]
    if GFX12 and _PS is not None and B.shape[1] == A.shape[1] * 16:   # preshuffled weight
        N = B.shape[0] * 16
        return torch.ops.radiance.preshuffle_gemm(A, B, As, Bs, N, A.shape[1])
    if GFX12 and M <= 8:
        return torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale(
            A, B, As, Bs, kernel.config.out_dtype)
    return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        A, B, As, Bs, list(kernel.weight_group_shape), kernel.config.out_dtype)
