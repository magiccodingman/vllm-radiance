"""radiance (gfx1201 / RDNA4) custom kernel dispatcher: the single place that picks which kernels vLLM
launches for block-FP8 GEMM, plus the preshuffle-GEMM and tuned unified-attention config hooks. vLLM's
source hooks (installed by patch_radiance_dispatch.py) only delegate here, so routing changes live in
this file instead of a vLLM patch.

The dispatch fns run inside vLLM's torch.compiled region, so they must be dynamo-traceable: plain shape
branches dispatching to registered torch.ops.*, no lru_cache / hasattr / side effects in the hot path.
"""
import math
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


# Tuned large-prefill 2D attention config, keyed by attention head size. Like the block-FP8 GEMM
# configs, each entry only ever applies to the shape it was measured on, so adding a head size can
# never move a head size that is already tuned.
#
#   head_size 256 (Qwen3.6): AITER's BLOCK_M of 64 is already right. Forcing 32 here was measured at
#     +8% TTFT @32K and +14% @64K, i.e. a clear regression; do not "unify" these two entries.
#   head_size 512 (Gemma4 global-attention layers): BLOCK_M 32 instead of 64 is worth -30% TTFT @32K,
#     -38% @64K and -46% @120K. The kernel is occupancy-limited at this head size, not matmul-shape
#     limited: the optimum sits at 8 query rows per warp (BLOCK_M/num_warps), and the same 8 is
#     reached either by halving BLOCK_M or doubling num_warps (64/8 measured -27%). Raising TILE_SIZE
#     does not help at any context length: 32 and 64 both measured slower than 16.
_PREFILL_2D_BY_HEAD = {
    512: {"TILE_SIZE": 16, "BLOCK_M": 32, "num_warps": 4, "waves_per_eu": 1},
}


# Tuned decode (3D split-KV) geometry for head_size 256 with fp8 Q + fp8 KV, gfx1201.
#
# Two things upstream leaves on the table, both measured in isolation as CUDA-graph replays against
# the shipped config (~/ab/tune3d.py), at 4K-128K context x 1-8 sequences x verify width 1-9:
#
# 1. The split-KV count.  select_3d_config sizes NUM_SEGMENTS_PER_SEQ for the TILE_SIZE it would
#    have used (64), which the override below then replaces with 16 -- so prod inherits a split
#    count computed for a tile 4x larger than the one it runs, off by up to 8x either way. Setting
#    it from the tile count the kernel will actually walk is worth 1.35-1.80x on its own, at every
#    depth and batch size measured, with no shape regressing.
#
# 2. BLOCK_M.  unified_attention derives it from the GQA ratio alone (16 whenever that ratio is
#    <= 16), hence BLOCK_Q = 16 // 6 = 2 here: an MTP verify batch of 9 query tokens is chopped into
#    5 q_blocks per sequence and *each one streams that sequence's whole KV cache*. One 64-row block
#    covers all 9 (BLOCK_Q 10) and reads KV once: 2.03-2.11x at 32K-128K. The catch is row
#    utilisation -- at verify width 6 the wide block is 20% SLOWER than the stock one, because 36 of
#    64 rows are padding. Since the shipped draft schedule runs width 6 at batch 8 and width 9 only
#    at batch 1, both regimes are live, so the widening is gated on utilisation (>= ~80%).
#
# Neither BLOCK_M nor BLOCK_Q is part of the config select_3d_config returns -- the wrapper computes
# them -- so #2 needs the launch shim below rather than a config entry.
_DECODE_3D_HEAD = 256          # only head size these numbers were measured at
_DECODE_3D_MAX_SPLITS = 128    # segm_output is [tokens, heads, splits, 256] fp32; 128 costs ~75 MB
_DECODE_3D_WIDE_ROWS = 0.8     # widen BLOCK_M only if the wide block is at least this full


def _decode_3d_geometry():
    """(num_tokens, num_seqs, num_queries_per_kv, num_kv_heads, all_decode), or None if unreadable.

    select_3d_config is handed the shapes it needs to size a *tile*, but not the batch shape, and
    it is called directly from unified_attention -- whose frame does have it. Reading it there keeps
    the split count and the BLOCK_M decision consistent with the launch that follows in the same
    call, with no cross-call state. Anything unexpected upstream returns None, and the caller then
    keeps the baseline constants.
    """
    try:
        L = sys._getframe(2).f_locals
        return (int(L["q"].shape[0]), int(L["num_seqs"]), int(L["num_queries_per_kv"]),
                int(L["num_kv_heads"]), bool(L["ALL_DECODE"]))
    except Exception:
        return None


def _decode_3d_block_m(geo, head_size):
    """The BLOCK_M to launch with, or None to keep the wrapper's. See note 2 above."""
    if geo is None or head_size != _DECODE_3D_HEAD:
        return None
    ntok, nseq, nqpkv, _, all_decode = geo
    if all_decode or nseq <= 0 or nqpkv <= 0:
        return None
    rows = -(-ntok // nseq) * nqpkv          # rows one 64-row block would have to hold
    if rows > 64 or rows < _DECODE_3D_WIDE_ROWS * 64:
        return None
    return 64


def _decode_3d_splits(max_seqlen_k, tile, prgms):
    """Split count ~ 16 * cbrt(tiles / programs), rounded down to a power of two.

    Fitted to the measured optimum over 16 (depth, batch, verify width) points; worst case 2.2% off
    the best split at any of them, exact at 11 of 16. The two terms are the trade it balances: more
    splits fill the machine, fewer keep each workgroup's Q load and partial-output write amortised
    over enough KV tiles.
    """
    tiles = max(1, -(-int(max_seqlen_k) // int(tile)))
    want = 16.0 * (tiles / max(1, int(prgms))) ** (1.0 / 3.0)
    seg = 1 << max(0, int(math.floor(math.log2(max(1.0, want)))))
    return max(4, min(_DECODE_3D_MAX_SPLITS, seg, 1 << (tiles - 1).bit_length()))


def install_attn_config_hook():
    """Override AITER's unified-attention config with the gfx1201-tuned one, per dtype, phase and
    head size:
       decode fp8 3D   TILE=16 warps2 stages1 waves6 (reduce warps8/stages1/waves2), plus a
                       shape-derived split-KV count and, at head 256 when the verify batch fills
                       it, a widened BLOCK_M -- see the notes above _decode_3d_geometry
       prefill 2D fp8  TILE=16 waves1  (large prefill only) + the per-head-size table above
       prefill 2D bf16 TILE=16 warps4 stages1 waves1
    Purely a tune: every LDS-fit (correctness) clamp lives in patch_unified_attention_lds.py
    instead, so this cannot make a model fail to start."""
    try:
        import aiter.ops.triton.attention.unified_attention as UA
    except Exception:
        return
    if getattr(UA, "_radiance_attn_wrapped", False):
        return
    FP8 = torch.float8_e4m3fn
    _o3, _o2 = UA.select_3d_config, UA.select_2d_config
    TWO_BYTE = (torch.bfloat16, torch.float16)   # includes --kv-cache-dtype auto

    _k3 = UA.kernel_unified_attention_3d
    pending_block_m = [None]      # set by _s3, consumed by the launch shim in the same call
    logged_decode = [False]       # one line, first verify batch, so a serve shows what it picked

    def _s3(*a, **k):   # decode (3D split-KV)
        ac, rc = _o3(*a, **k)
        head_size = a[0] if len(a) > 0 else k.get("head_size", 0)
        max_seqlen_k = a[2] if len(a) > 2 else k.get("max_seqlen_k", 0)
        num_2d_prgms = a[4] if len(a) > 4 else k.get("num_2d_prgms", 1)
        q_dtype = a[5] if len(a) > 5 else k.get("q_dtype")
        kv_dtype = a[6] if len(a) > 6 else k.get("kv_cache_dtype")
        pending_block_m[0] = None
        if q_dtype == FP8 and kv_dtype == FP8:   # fp8 decode; bf16 3D is handled by the build-time patch
            ac["TILE_SIZE"] = 16; ac["num_warps"] = 4; ac["num_stages"] = 1; ac["waves_per_eu"] = 6
            rc["TILE_SIZE"] = 16; rc["num_warps"] = 8; rc["num_stages"] = 1; rc["waves_per_eu"] = 2
            geo = _decode_3d_geometry()
            # ALL_DECODE is the MTP drafter's own single-token pass. It already runs at ~97% of the
            # DRAM roofline (576 GB/s measured), i.e. there is nothing for a wider block or a
            # different split count to recover, and the rules below -- fitted on verify batches --
            # only cost it ~1%. It keeps the constants it has always had.
            if geo is not None and not geo[4] and head_size == _DECODE_3D_HEAD:
                ac["num_warps"] = 2
                block_m = _decode_3d_block_m(geo, head_size)
                prgms = num_2d_prgms
                if block_m is not None:
                    # the wide block also wants a wider KV tile and twice the warps (16 query
                    # rows per warp either way), and it moves the program count the split rule
                    # sees, since the launch it feeds is the widened one.
                    ntok, nseq, nqpkv, kvh, _ = geo
                    ac["TILE_SIZE"] = rc["TILE_SIZE"] = 32
                    ac["num_warps"] = 4
                    prgms = (ntok // (block_m // nqpkv) + nseq) * kvh
                seg = _decode_3d_splits(max_seqlen_k, ac["TILE_SIZE"], prgms)
                ac["NUM_SEGMENTS_PER_SEQ"] = rc["NUM_SEGMENTS_PER_SEQ"] = seg
                pending_block_m[0] = block_m
                if not logged_decode[0]:
                    logged_decode[0] = True
                    sys.stderr.write(
                        f"[radiance] decode 3D geometry: BLOCK_M={block_m or 'stock'} "
                        f"TILE={ac['TILE_SIZE']} splits={seg} warps={ac['num_warps']} "
                        f"(seqs={geo[1]} tokens={geo[0]} kv={max_seqlen_k})\n")
                    sys.stderr.flush()
        return ac, rc

    class _Wide3D:
        """Launch shim: applies the BLOCK_M _s3 chose, and the grid that follows from it.

        The wrapper's q_block count is an upper bound (sum of floor(len_i / BLOCK_Q) + 1), which
        stays correct for any BLOCK_Q -- blocks past a sequence's query length return immediately --
        so widening only ever shrinks the launch. reduce_segments takes BLOCK_Q too but never uses
        it (its find_seq_idx runs in token mode), so it needs no shim.
        """
        __slots__ = ()

        def __getitem__(self, grid):
            def launch(**kw):
                block_m = pending_block_m[0]
                pending_block_m[0] = None
                if not block_m:
                    return _k3[grid](**kw)
                bq = max(1, block_m // kw["num_queries_per_kv"])
                kw["BLOCK_M"] = block_m
                kw["BLOCK_Q"] = bq
                g = (kw["query_ptr"].shape[0] // bq + int(kw["num_seqs"]), grid[1], grid[2])
                return _k3[g](**kw)
            return launch

    def _s2(*a, **k):   # prefill / MTP decode-via-2D
        c = _o2(*a, **k)
        head_size = a[1] if len(a) > 1 else k.get("head_size", 0)
        max_seqlen_q = a[4] if len(a) > 4 else k.get("max_seqlen_q", 0)
        num_queries_per_kv = a[6] if len(a) > 6 else k.get("num_queries_per_kv", 1)
        kv_dtype = a[9] if len(a) > 9 else k.get("kv_cache_dtype")
        if kv_dtype in TWO_BYTE:                 # bf16/auto KV: warps=4 is decisive on large prefill
            c["TILE_SIZE"] = 16; c["num_warps"] = 4; c["num_stages"] = 1; c["waves_per_eu"] = 1
        elif max_seqlen_q >= 256:                # fp8 large prefill
            c["TILE_SIZE"] = 16; c["waves_per_eu"] = 1
            tuned = _PREFILL_2D_BY_HEAD.get(head_size)
            if tuned is not None:
                c.update(tuned)
                # BLOCK_Q is derived from BLOCK_M upstream, so it has to follow it here too.
                c["BLOCK_Q"] = max(1, c["BLOCK_M"] // max(1, num_queries_per_kv))
        return c

    # aiter exposes this file under two module names (`aiter.ops.triton.unified_attention` and
    # `aiter.ops.triton.attention.unified_attention`) and executes it once per name, so there are two
    # module objects with two independent copies of these globals. Whichever name a caller imported
    # from decides which copy its `unified_attention` resolves `select_3d_config` in -- and vLLM's
    # main model and its MTP drafter import at different times. Patching only the module imported here
    # left roughly one attention call per engine step on aiter's stock config, at ~5.6 ms a call.
    # Patch every loaded alias, and import the other name so a later import cannot pick up a fresh
    # unpatched copy.
    targets = []
    for name in ("aiter.ops.triton.unified_attention",
                 "aiter.ops.triton.attention.unified_attention"):
        try:
            targets.append(__import__(name, fromlist=["select_3d_config"]))
        except Exception:
            continue
    for mod in sys.modules.values():
        if mod is not None and getattr(mod, "__name__", "").endswith("unified_attention") \
                and hasattr(mod, "select_3d_config") and mod not in targets:
            targets.append(mod)
    if UA not in targets:
        targets.append(UA)

    patched = 0
    for mod in targets:
        if getattr(mod, "_radiance_attn_wrapped", False):
            continue
        mod.select_3d_config = _s3
        mod.select_2d_config = _s2
        mod.kernel_unified_attention_3d = _Wide3D()
        mod._radiance_attn_wrapped = True
        patched += 1
    sys.stderr.write(f"[radiance] attn tuned-config override installed on {patched} module alias"
                     f"{'es' if patched != 1 else ''}\n")
    sys.stderr.flush()


# ---- R4D kernel selection report ----
# Which R4D kernel serves which part of the model is decided by r4d.select(), and every one of
# those calls happens at import time, scattered across five modules, before there is any log a
# reader is following. The library records the questions (see selections() in r4d_module.hip);
# this prints them back once the worker is up, which is the moment a user is actually looking.
#
# It is deliberately the whole log rather than the kernels that were bound: a query that resolved
# to nothing is the more useful line of the two, because it names the constraint that sent this
# model down a fallback path.

def _r4d_report_lines():
    """The selection report as a list of lines, or None when there is nothing to report."""
    import r4d

    rows = r4d.selections()
    if not rows:
        return None
    version = getattr(r4d, "__version__", "?")
    built = len(r4d.kernels())
    hit = sum(1 for r in rows if r["kernel"])
    out = [
        f"R4D kernel selection: libr4d {version}, {built} kernels built, "
        f"{hit} of {len(rows)} queries resolved"
    ]
    # Pad the op names into a column so the queries line up; the query itself is free-form, since
    # each op asks for a different geometry.
    width = max(len(r["op"]) for r in rows)
    for r in rows:
        query = " ".join(f"{k}={v}" for k, v in r["query"].items())
        times = f"  (asked {r['count']}x)" if r["count"] > 1 else ""
        out.append(f"  {r['op']:<{width}}  {query}")
        if r["kernel"]:
            out.append(f"  {'':<{width}}  -> {r['kernel']}{times}")
        else:
            out.append(f"  {'':<{width}}  -> no kernel, fallback runs: {r['reason']}{times}")
    return out


def install_r4d_report():
    """Print the R4D selection report once the model is loaded and the graphs are captured.

    compile_or_warm_up_model is the worker's last init step, so by the time it returns every hook
    has imported and every select() has been made -- including the ones on paths that only resolve
    when a real forward runs. Rank 0 only: the ranks are symmetric and two copies is just noise.

    Gated by RADIANCE_R4D_REPORT (default on). Never fatal: this is a log line, and a serve must
    not fail to start over one."""
    if os.environ.get("RADIANCE_R4D_REPORT", "1") != "1":
        return
    try:
        import r4d  # noqa: F401
    except Exception:
        return      # no library, nothing to report
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker.compile_or_warm_up_model, "_radiance_r4d_report", False):
        return
    _orig = Worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *a, **kw):
        result = _orig(self, *a, **kw)
        try:
            if self.rank == 0:
                lines = _r4d_report_lines()
                if lines:
                    sys.stderr.write("".join(f"[radiance] {ln}\n" for ln in lines))
                    sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[radiance] R4D selection report failed: {e!r}\n")
        return result

    compile_or_warm_up_model._radiance_r4d_report = True
    Worker.compile_or_warm_up_model = compile_or_warm_up_model


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
        radiance_allreduce.install_custom_ar()   # RADIANCE_USE_R4D_AR (default on)
    except Exception as e:
        sys.stderr.write(f"[radiance] install_custom_ar failed: {e!r}\n")
    try:
        import radiance_draft
        radiance_draft.install()                 # RADIANCE_DYNAMIC_DRAFT (default off)
    except Exception as e:
        sys.stderr.write(f"[radiance] radiance_draft install failed: {e!r}\n")
    try:
        import radiance_drafthead
        radiance_drafthead.install()             # RADIANCE_FAST_DRAFT: 2-bit draft head + exact rerank
    except Exception as e:
        sys.stderr.write(f"[radiance] radiance_drafthead install failed: {e!r}\n")
    try:
        import radiance_vit_attn
        radiance_vit_attn.install()              # gfx12x, needs the R4D library
    except Exception as e:
        sys.stderr.write(f"[radiance] radiance_vit_attn install failed: {e!r}\n")
    try:
        install_r4d_report()                     # RADIANCE_R4D_REPORT (default on)
    except Exception as e:
        sys.stderr.write(f"[radiance] install_r4d_report failed: {e!r}\n")


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
