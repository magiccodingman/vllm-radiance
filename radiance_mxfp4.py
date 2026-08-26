"""RADIANCE gfx1201 MXFP4 dispatch: routes Quark/OCP MXFP4 linears to the hand-written
W4A8 fp8-WMMA kernel, using distinct prefill and decode-shaped tilings.

Why: Triton lowers tl.dot_scaled by upconverting e2m1 to bf16 and using the 16-bit WMMA. Measured
here, register-resident: fp8 WMMA 325.2 TFLOP/s vs f16 160.2 (2.03x), while Triton's own fp8 tl.dot
manages only 43.3 -- it will not emit the fp8 matrix instruction. The hand-written kernel measures
1.6-1.9x the tuned aiter path at prefill shapes.

This is a NUMERICS change, not just a speed one: the checkpoint declares W4A4 and this runs W4A8.
fp8 activations are strictly more precise than the fp4 the model was calibrated against, but output
is no longer bit-identical to emulation, so it is opt-in via RADIANCE_MXFP4_W4A8=1. The qualified
route uses RADIANCE_MXFP4_W4A8_MIN_M=0 so every M stays on W4A8: AITER's generic W4A4 fallback is
numerically wrong for the Qwen GDN N=5120,K=3072 projection. M<=48 uses the decode-shaped kernel;
larger M uses the prefill kernel.

Integration: vLLM 0.27 replaced QuarkOCP_MX's inline dispatch with a kernel plugin ABC --
MxFp4LinearKernel, selected in priority order from _POSSIBLE_MXFP4_KERNELS[platform] by
init_mxfp4_linear_kernel(). RadianceMxfp4W4A8LinearKernel below is that plugin;
patch_quark_mxfp4.py does nothing but put it at the head of the ROCm list. Before 0.27 this took
seven string hunks against a single 389-line file, all of which the rewrite invalidated.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_MXFP4_W4A8", "0") == "1"
MIN_M = int(os.environ.get("RADIANCE_MXFP4_W4A8_MIN_M", "0"))
# Decode band for the small-M kernel. 0 = dark. Also gates the scratch preallocation.
DECODE_MAX_M = int(os.environ.get("RADIANCE_MXFP4_DECODE_MAX_M", "64"))
_decode_scratch_ready = [False]
_decode_scratch = [None, None]   # [partials, block counter] — kept alive for the process

try:
    import radiance_mxfp4_fp8 as _ext
except Exception as e:                      # ext missing: stay on the aiter Triton path
    _ext = None
    ENABLED = False
    sys.stderr.write(f"[radiance.mxfp4] w4a8 ext import failed, disabled: {e!r}\n")

if ENABLED and _ext is not None:
    # Print WHICH .so was loaded. A stale copy sitting in the bind-mounted repo shadows the one
    # compiled into site-packages (the working directory precedes it on sys.path), and a
    # mismatched kernel fails silently -- fluent-looking garbage, no error anywhere.
    sys.stderr.write(
        f"[radiance.mxfp4] W4A8 fp8-WMMA GEMM ENABLED for M>{MIN_M} "
        f"(NOTE: W4A8, not the checkpoint's W4A4 -- more precise activations, not bit-identical)\n"
        f"[radiance.mxfp4] kernel: {getattr(_ext, '__file__', '?')}\n")


# The WHOLE W4A8 path -- activation quant included -- is one registered custom op. An earlier
# version called a plain Python helper from apply_weights instead, which sits inside the
# torch.compile region: dynamo could not trace through the module-object call, broke the graph at
# every one of ~450 linears, and cost 33% of DECODE throughput (61 -> 41 tok/s) even though decode
# never reaches the kernel. A registered op is a single opaque graph node, and the gate in front of
# it must be nothing but plain attribute/shape comparisons.
DEBUG = os.environ.get("RADIANCE_MXFP4_DEBUG", "0") == "1"
PURE_QUANT = os.environ.get("RADIANCE_MXFP4_PUREQUANT", "0") == "1"
# Diagnostic: synchronize after the raw kernel launch. Every layer verifies correct against an
# exact fp32 reference -- but that check calls .item(), which synchronizes, and it only runs for
# the first few calls. If forcing a sync makes the model coherent, the fault is ordering: the
# kernel is being launched on a stream the consumer does not wait on.
_fs = os.environ.get("RADIANCE_MXFP4_SYNC", "0")
FORCE_SYNC = _fs == "1"          # synchronize the stream we launched on
FORCE_DEVSYNC = _fs == "2"       # synchronize the whole device -- if this differs from the above,
                                 # the kernel is not running on the stream we handed it
# Diagnostic: return a copy of the kernel's output buffer instead of the buffer itself. `out` is
# allocated by torch.empty INSIDE the custom op; if returning that buffer is what breaks (lifetime,
# aliasing, or the allocator reusing it), a clone will be coherent where the original is not.
CLONE_OUT = os.environ.get("RADIANCE_MXFP4_CLONE", "0") == "1"
# Allocate `out` inside a padded slab so nothing live sits immediately before or after it. The one
# thing that makes the N=5120 K=3072 layers coherent is perturbing the allocator, which points at
# memory adjacency rather than at the arithmetic (the kernel verifies correct against exact fp32,
# writes every element, and writes nothing out of bounds). If padding fixes it, that is both the
# diagnosis and the fix. Value is in bf16 elements per side.
PAD_OUT = int(os.environ.get("RADIANCE_MXFP4_PADOUT", "0"))
# Shadow check: for one N:K that is being served by AITER (so the model stays healthy and the
# activations are CLEAN), also compute the layer with our kernel and compare both against exact
# fp32. Every previous in-situ comparison was made on an activation that already held NaN, so it
# could not distinguish "our kernel is wrong" from "our kernel faithfully computed on garbage".
_sh = os.environ.get("RADIANCE_MXFP4_SHADOW", "").strip()
SHADOW_NK = tuple(int(v) for v in _sh.split(":")) if _sh else None
# Sanitize non-finite activations before quantizing. NOT optional -- this is the fix for the
# N=5120 K=3072 layers (gdn out_proj / attention o_proj).
#
# Those layers legitimately receive NaN in their input: measured on a HEALTHY, coherent serve,
# one rank's out_proj input carries 2176 non-finite values at M=17, which is exactly 17 rows x 128
# columns -- one whole gated-delta-net head. aiter tolerates it because it quantizes activations to
# mxfp4, and NaN squashes to a finite code. Our path quantizes to per-token fp8, where a single NaN
# makes the row's amax NaN, hence the scale NaN, hence the entire row NaN -- which then propagates
# through the residual stream and destroys the model.
#
# That is why the kernel always verified correct and the model was still garbage: the GEMM was
# faithfully computing on a poisoned row. Zeroing non-finite inputs matches what the mxfp4 path
# effectively does, and the model is coherent under aiter with the same NaN present.
# Default OFF since the libr4d GDN overflows are fixed at source (see README). Kept as a safety
# net for a stock r4d.so, where it is the difference between PPL 8.4004 and 653586 -- but with the
# patched library it is a pure elementwise cost, and measured quality is BETTER without it
# (8.3706 fixed/no-sanitize vs 8.4004 stock/sanitize).
SANITIZE_X = os.environ.get("RADIANCE_MXFP4_SANITIZE", "0") == "1"
# Diagnostic: report the INPUT activation's health per layer. The exact-reference check derives its
# reference from x itself, so it cannot tell a correct kernel on corrupt input from a correct one.
CHECK_X = os.environ.get("RADIANCE_MXFP4_CHECKX", "0") == "1"
# Verify EVERY call at one N:K against exact fp32, with no dedup. The earlier check kept only the
# first call per (N,K,M), so a shape that is right once and wrong later reads as "ok".
_ca = os.environ.get("RADIANCE_MXFP4_CHECKALL", "").strip()
# Comma-separated list of N:K shapes to verify against exact fp32 on every call, e.g.
# "17408:5120,5120:8704". A list rather than a single pair so one serve can gate every
# production shape -- checking them one per serve costs a container restart each.
CHECK_ALL = ({tuple(int(v) for v in pair.split(":")) for pair in _ca.split(",") if pair}
             if _ca else None)
# Bisect which layer class our kernel breaks. Comma-separated N values our kernel is allowed to
# serve; every other layer is handed to aiter (which is known-coherent for the whole model).
# Empty = no restriction. Gating projections (N=48, in_proj_ba) feed exponentials in the GDN core,
# so a small numeric difference there can become Inf/NaN downstream where an MLP shape cannot.
_only = os.environ.get("RADIANCE_MXFP4_KERNEL_N", "").strip()
KERNEL_N = {int(v) for v in _only.split(",") if v} if _only else None
# Finer bisect: N alone is ambiguous. N=5120 is BOTH the MLP down_proj (K=8704) and the
# gated-delta-net out_proj (K=3072), which are very different layers. "N:K,N:K" pairs separate them.
_onlynk = os.environ.get("RADIANCE_MXFP4_KERNEL_NK", "").strip()
KERNEL_NK = {tuple(int(x) for x in pair.split(":")) for pair in _onlynk.split(",") if pair} \
    if _onlynk else None
# Shapes forced onto the kernel's per-block-rescale path (wref=0) instead of the folded path.
# The folded path pre-shifts the block exponent into the weight via kLUT2; the per-block path
# applies the scale in the inner loop instead. Same kernel, different numerics path.
_pb = os.environ.get("RADIANCE_MXFP4_PERBLOCK_NK", "").strip()
PERBLOCK_NK = {tuple(int(x) for x in pair.split(":")) for pair in _pb.split(",") if pair} \
    if _pb else None
# Diagnostic: compute the layer exactly (dequantize the weight, bf16 F.linear) and return THAT,
# while still doing all the surrounding work. Slow. It separates "our kernel's values are wrong in
# situ" from "something outside this op is wrong": if the model is coherent with this on, the op's
# wiring is fine and only the kernel output differs; if it is still garbage, the fault is elsewhere.
REF_LINEAR = os.environ.get("RADIANCE_MXFP4_REFLINEAR", "0") == "1"
_E2M1 = None
_dbg_seen = set()


def _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K, chunk=2048):
    """fp32 reference for this layer, chunked over N so it fits alongside a loaded model.

    Compared against AITER the earlier check was circular -- aiter is itself wrong on some shapes,
    so "rel 0.05-0.16 looks like the W4A8-vs-W4A4 spread" proved nothing. This is ground truth.
    """
    global _E2M1
    if _E2M1 is None:
        _E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                              -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=weight.device)
    xr = x_fp8.float() * x_scale.view(-1, 1).float()
    outs = []
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        wc = weight[a:b]
        codes = torch.stack([wc & 0x0F, (wc >> 4) & 0x0F], -1).reshape(b - a, K)
        sc = torch.pow(2.0, weight_scale[:, a:b].float() - 127.0).T.repeat_interleave(32, dim=1)
        outs.append(xr @ (_E2M1[codes.long()] * sc).T.float())
    return torch.cat(outs, dim=1)


def _debug_compare(x, weight, weight_scale, weight_ref, out, x_fp8, x_scale):
    """One-shot per (N,K): how far is our kernel from the aiter path on REAL activations?

    aiter quantizes x to mxfp4 where we use fp8, so a relative difference around 0.1 is expected
    and healthy (measured 0.0265 vs 0.1119 against exact arithmetic). Order-1 means we are wrong.
    """
    key = (int(weight.shape[0]), int(x.shape[1]), int(x.shape[0]))
    if key in _dbg_seen or len(_dbg_seen) > 40:
        return
    _dbg_seen.add(key)
    try:
        M, N, K = int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        if M > 128:
            return                      # reference is only affordable at small M
        ref = _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
        sys.stderr.write(f"[radiance.mxfp4.exact] N={N} K={K} M={M} rel_vs_fp32={rel:.5f} "
                         f"|ours|={out.float().abs().mean():.5f} |ref|={ref.abs().mean():.5f} "
                         f"{'**WRONG**' if rel > 0.02 else 'ok'}\n")
        sys.stderr.flush()
        return
        sys.stderr.write(
            f"[radiance.mxfp4.debug] N={key[0]} K={key[1]} M={x.shape[0]} "
            f"x={tuple(x.shape)}/{x.dtype}/contig={x.is_contiguous()} "
            f"w={tuple(weight.shape)}/{weight.dtype} "
            f"ws={tuple(weight_scale.shape)}/{weight_scale.dtype}/contig={weight_scale.is_contiguous()} "
            f"wref={tuple(weight_ref.shape)}/{weight_ref.dtype} "
            f"xq={x_fp8.dtype} xs={tuple(x_scale.shape)}/{x_scale.dtype} "
            f"|ours|={out.float().abs().mean().item():.4f} |aiter|={ref.float().abs().mean().item():.4f} "
            f"rel={rel:.4f} nonfinite={(~torch.isfinite(out.float())).sum().item()}\n")
        sys.stderr.flush()
        if rel > 0.5:
            # Deterministic-wrong or a race? Re-run the kernel into a fresh buffer from the SAME
            # inputs, then dump everything so it can be replayed offline against the same .so.
            out2 = torch.empty_like(out)
            _ext.launch(x_fp8.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                        weight_ref.data_ptr(), x_scale.data_ptr(), out2.data_ptr(),
                        int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]),
                        torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            rerun = ((out2.float() - out.float()).norm()
                     / out.float().norm().clamp_min(1e-9)).item()
            rel2 = ((out2.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
            import os as _os
            path = f"/cache/badcase_{_os.getpid()}_{key[0]}_{key[1]}_{key[2]}.pt"
            torch.save({"x_fp8": x_fp8.cpu(), "x_scale": x_scale.cpu(),
                        "weight": weight.cpu(), "weight_scale": weight_scale.cpu(),
                        "weight_ref": weight_ref.cpu(), "out": out.cpu(),
                        "out_rerun": out2.cpu(), "aiter": ref.cpu(), "x": x.cpu(),
                        "M": int(x.shape[0]), "N": int(weight.shape[0]), "K": int(x.shape[1])}, path)
            sys.stderr.write(f"[radiance.mxfp4.debug] BAD CASE rel={rel:.4f} rerun_delta={rerun:.6f} "
                             f"rel_of_rerun={rel2:.4f} -> {path}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[radiance.mxfp4.debug] compare failed: {e!r}\n")


@torch.library.custom_op("radiance::mxfp4_linear", mutates_args=())
def mxfp4_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 weight_ref: torch.Tensor) -> torch.Tensor:
    """Owns the ENTIRE dispatch, because the branch must not be visible to dynamo.

    vLLM compiles the model with a dynamic token dimension, so a plain `x.shape[0] > 256` in
    apply_weights is a data-dependent branch: it splits the graph at every linear and cost ~30% of
    decode throughput even though decode never takes the W4A8 side. Inside a registered custom op
    the body runs eagerly and the Python `if` is free."""
    if not _stats_reported[0]:
        report_stats()          # safe here, and only here: the op body is opaque to dynamo
    # Eligibility is derived from the operands, not passed in. It used to arrive as a Python bool
    # read off the layer in apply_weights -- which dynamo TRACES and bakes into the graph as a
    # per-layer constant. Keeping scalars out of the traced region is the same rule the M
    # comparison already follows, and it means a stale compiled graph cannot carry a stale gate.
    # weight_ref encodes the route, because a Python scalar read in apply_weights would be traced
    # and baked into the compiled graph:
    #   numel == N -> our kernel, folded path (block exponent pre-shifted into the weight)
    #   numel == 2 -> our kernel, per-block path (scale applied in the inner loop); wref ptr = 0
    #   numel == 1 -> the kernel cannot serve this layer at all; hand it to aiter
    nref = weight_ref.numel()
    folded = nref == weight.shape[0]
    w4a8_ok = folded or nref == 2
    if not (w4a8_ok and x.shape[0] > MIN_M):
        STATS["aiter_calls"] = STATS.get("aiter_calls", 0) + 1
        if STATS["aiter_calls"] in (1, 100, 10000):
            sys.stderr.write(f"[radiance.mxfp4] AITER BRANCH TAKEN "
                             f"(call #{STATS['aiter_calls']}, M={x.shape[0]}, "
                             f"N={weight.shape[0]}, w4a8_ok={w4a8_ok})\n")
        # MIN_M <= 0 makes this unreachable: aiter's W4A4 path is measured WRONG on some shapes
        # (N=5120 K=3072 returns ~1/35th of the correct magnitude), so it is not a safe fallback.
        y_aiter = torch.ops.vllm.gemm_with_dynamic_quant(x, weight, weight_scale, False,
                                                         torch.bfloat16)
        if SHADOW_NK is not None and (int(weight.shape[0]), int(x.shape[1])) == SHADOW_NK \
                and x.shape[0] <= 128:
            k = (int(weight.shape[0]), int(x.shape[1]), int(x.shape[0]))
            # Skip CUDA-graph capture entirely: those runs use zero activations (where every
            # kernel trivially agrees, wasting the dedup budget) and, more importantly, ANY .item()
            # here is a device sync, which is illegal during capture and kills the engine.
            _capturing = torch.cuda.is_current_stream_capturing()
            if not _capturing and k not in _dbg_seen and len(_dbg_seen) < 24:
                if float(x.abs().max().item()) == 0.0:
                    return y_aiter          # warmup on zeros: nothing to learn
                _dbg_seen.add(k)
                try:
                    from vllm import _custom_ops as _ops
                    _M, _K = x.shape
                    _N = weight.shape[0]
                    _xq, _xs = _ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
                    _xs = _xs.view(-1).float().contiguous()
                    _wref = make_row_ref(weight_scale)           # folded path needs it live
                    _o = torch.empty((_M, _N), device=x.device, dtype=torch.bfloat16)
                    _ext.launch(_xq.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                                _wref.data_ptr(), _xs.data_ptr(), _o.data_ptr(),
                                _M, _N, _K, torch.cuda.current_stream().cuda_stream)
                    _ref = _exact_ref(_xq, _xs, weight, weight_scale, _N, _K)
                    _ro = ((_o.float() - _ref).norm() / _ref.norm().clamp_min(1e-9)).item()
                    _ra = ((y_aiter.float() - _ref).norm() / _ref.norm().clamp_min(1e-9)).item()
                    _xf = x.float()
                    sys.stderr.write(
                        f"[radiance.mxfp4.shadow] N={_N} K={_K} M={_M} "
                        f"ours_vs_fp32={_ro:.5f} aiter_vs_fp32={_ra:.5f} "
                        f"x_nonfin={(~torch.isfinite(_xf)).sum().item()} "
                        f"|x|={_xf.abs().mean().item():.5f} "
                        f"{'**OURS WRONG**' if _ro > 0.02 else 'ours ok'}\n")
                    sys.stderr.flush()
                except Exception as _e:
                    sys.stderr.write(f"[radiance.mxfp4.shadow] failed: {_e!r}\n")
        return y_aiter
    M, K = x.shape
    N = weight.shape[0]
    if SANITIZE_X:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if PURE_QUANT:
        # Pure-torch per-token e4m3 quantization, no vLLM custom op. Diagnostic only: this exists
        # to answer whether calling torch.ops._C.dynamic_scaled_fp8_quant from INSIDE another
        # custom op is what disturbs the compiled graph.
        amax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12)
        sc = amax / 448.0
        x_fp8 = (x.float() / sc).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        x_scale = sc.view(-1).contiguous()
    else:
        from vllm import _custom_ops as ops
        x_fp8, x_scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
        x_scale = x_scale.view(-1).float().contiguous()
    if PAD_OUT:
        _slab = torch.empty(M * N + 2 * PAD_OUT, device=x.device, dtype=torch.bfloat16)
        out = _slab[PAD_OUT:PAD_OUT + M * N].view(M, N)   # view keeps the slab alive
    else:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch(x_fp8.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                weight_ref.data_ptr() if folded else 0,
                x_scale.data_ptr(), out.data_ptr(), M, N, K,
                torch.cuda.current_stream().cuda_stream)
    if CHECK_ALL is not None and (N, K) in CHECK_ALL and x.shape[0] <= 128:
        _ref = _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K)
        _rel = ((out.float() - _ref).norm() / _ref.norm().clamp_min(1e-9)).item()
        STATS["checked"] = STATS.get("checked", 0) + 1
        if _rel > 0.02:
            STATS["wrong"] = STATS.get("wrong", 0) + 1
            if STATS["wrong"] <= 8:
                xf2 = x.float()
                sys.stderr.write(
                    f"[radiance.mxfp4.all] **WRONG** call#{STATS['checked']} N={N} K={K} "
                    f"M={x.shape[0]} rel={_rel:.4f} |ours|={out.float().abs().mean():.5f} "
                    f"|ref|={_ref.abs().mean():.5f} x_nonfin={(~torch.isfinite(xf2)).sum().item()} "
                    f"x_contig={x.is_contiguous()} x_stride={tuple(x.stride())}\n")
        else:
            _k = ("seen", N, K)
            STATS[_k] = STATS.get(_k, 0) + 1
            if STATS[_k] in (1, 64):
                sys.stderr.write(f"[radiance.mxfp4.all] ok N={N} K={K} M={x.shape[0]} "
                                 f"call#{STATS[_k]} rel={_rel:.5f} "
                                 f"(wrong so far: {STATS.get('wrong', 0)})\n")
        sys.stderr.flush()
    if CHECK_X:
        key = (N, K, int(x.shape[0]))
        if key not in _dbg_seen and len(_dbg_seen) < 60:
            _dbg_seen.add(key)
            xf = x.float()
            of = out.float()
            sys.stderr.write(f"[radiance.mxfp4.x] N={N} K={K} M={x.shape[0]} "
                             f"|x|={xf.abs().mean().item():.5f} xmax={xf.abs().max().item():.5f} "
                             f"x_nonfin={(~torch.isfinite(xf)).sum().item()} | "
                             f"|out|={of.abs().mean().item():.5f} "
                             f"outmax={of.abs().max().item():.5f} "
                             f"out_nonfin={(~torch.isfinite(of)).sum().item()} "
                             f"out_over448={(of.abs() > 448).sum().item()}\n")
            sys.stderr.flush()
    if FORCE_SYNC:
        torch.cuda.current_stream().synchronize()
    if FORCE_DEVSYNC:
        torch.cuda.synchronize()
    if DEBUG:
        _debug_compare(x, weight, weight_scale, weight_ref, out, x_fp8, x_scale)
    if REF_LINEAR:
        global _E2M1
        if _E2M1 is None:
            _E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                  -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                                 device=weight.device)
        codes = torch.stack([weight & 0x0F, (weight >> 4) & 0x0F], -1).reshape(N, K)
        w = _E2M1[codes.long()] * torch.pow(
            2.0, weight_scale.float() - 127.0).T.repeat_interleave(32, dim=1)
        return (x.float() @ w.T.float()).to(torch.bfloat16)
    return out.clone() if CLONE_OUT else out


@mxfp4_linear.register_fake
def _(x, weight, weight_scale, weight_ref):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


def make_row_ref(weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-output-row reference exponent, computed ONCE at load.

    Folding the MX block exponent into the weight needs a single reference per row; the kernel
    stores 2^(ref-127) back in the epilogue. weight_scale arrives as [K/32, N] (the layout the
    native path leaves behind), so the max is over dim 0."""
    return weight_scale.max(dim=0).values.contiguous()


# Per-layer accounting. A layer that fails layer_is_supported() still runs, on the kernel's slower
# per-block-rescale path, and a layer below MIN_M runs on aiter -- both silently. Count them so
# "no fallbacks" is something the log proves rather than something we assume.
STATS = {"fast": 0, "aiter": 0}
_stats_reported = [False]


def report_stats():
    """One-shot fast-path tally. Call ONLY from inside the custom op or at load time -- never from
    apply_weights, which dynamo traces: sys.stderr.write is a skipped builtin and compilation
    fails outright rather than falling back."""
    if _stats_reported[0]:
        return
    _stats_reported[0] = True
    n = STATS["fast"] + STATS["aiter"]
    sys.stderr.write(
        f"[radiance.mxfp4] linear layers: {STATS['fast']}/{n} on our kernel, "
        f"{STATS['aiter']} FORCED ONTO AITER (kernel cannot run them); "
        f"aiter below M={MIN_M} ({'DISABLED' if MIN_M <= 0 else 'ACTIVE'})\n")
    sys.stderr.flush()


def layer_is_supported(layer, K: int) -> bool:
    """Called ONCE per layer at load time, never in the forward path.

    Only K is a hard constraint: the launcher rejects K % BK, where BK=64. N is NOT -- the kernel
    masks a partial N tile, measured at N=48 (the gated-delta-net gate projection) as relRMSE
    0.00174 folded / 0.00155 per-block with zero out-of-bounds writes on either side of `out`.

    The old gate also demanded N % 64 == 0, which failed those 48 layers per rank and sent them to
    aiter -- NOT, as the name suggested, to this kernel's per-block path. That mattered: aiter's
    W4A4 Triton GEMM returns wrong values for some shape/M-band combinations here, so the strict
    gate was silently routing real layers onto a path that cannot be trusted."""
    try:
        return bool(ENABLED and K % 64 == 0
                    and layer.weight.shape[1] * 2 == K
                    and layer.weight_scale.dim() == 2
                    and layer.weight_scale.shape[0] == K // 32)
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# The vLLM 0.27 kernel plugin
# --------------------------------------------------------------------------------------------
# Deliberately NOT composed with AiterMxfp4LinearKernel. Its __init__ asserts is_supported(), which
# gates on current_platform.supports_mx() -- a CDNA4 (gfx950/gfx1250) allowlist that gfx1201 fails.
# The sub-MIN_M fallback calls torch.ops.vllm.gemm_with_dynamic_quant directly instead. That op is
# registered by vllm/model_executor/kernels/linear/mxfp4/aiter.py under
# `if is_aiter_found_and_supported():`, which does NOT consult supports_mx(), so it is present here.


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return bool(on_gfx12x())
    except Exception:
        return False


def _asm_gemm_enabled() -> bool:
    try:
        from vllm._aiter_ops import rocm_aiter_ops
        return bool(rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled())
    except Exception:
        return False


def _make_kernel_class():
    """Built lazily so importing this module never drags in vllm.model_executor.kernels."""
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearKernel,
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    class RadianceMxfp4W4A8LinearKernel(MxFp4LinearKernel):
        """MXFP4 weights x fp8 activations on gfx1201, via the hand-written fp8-WMMA GEMM."""

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not ENABLED or _ext is None:
                return False, "RADIANCE_MXFP4_W4A8 is not enabled, or the HIP extension is missing"
            if not _on_gfx12x():
                return False, "the radiance W4A8 MXFP4 kernel is compiled for gfx12x only"
            return True, None

        @classmethod
        def can_implement(cls, config: MxFp4LinearLayerConfig):
            if config.activation_quant_key != kMxfp4Dynamic:
                return False, "only supports MXFP4 dynamic activation"
            # The asm path stores weights shuffled (16,16) and the scale in a swizzled layout;
            # this kernel reads the plain packed weight and a [K/32, N] scale, and the sub-MIN_M
            # fallback would hand shuffled operands to the non-asm aiter GEMM. Decline instead of
            # silently computing the wrong thing -- AiterMxfp4LinearKernel takes it from here.
            if _asm_gemm_enabled():
                return False, "aiter asm fp4 GEMM is enabled; its weight layout is incompatible"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Preallocate the decode kernel's split-K partial slab on the FIRST layer, which is
            # weight-load time -- outside CUDA-graph capture.
            #
            # It cannot be done lazily inside the kernel launcher. Whenever the torch.compile cache
            # is warm ("Directly load AOT compilation from path ..."), vLLM skips the eager profile
            # run and the first GEMM call happens during capture, where hipMalloc is illegal. That
            # killed engine startup, and the error it surfaced through our own pybind call was a
            # thoroughly misleading "HIP runtime library (libamdhip64.so) not found ... TileLang's
            # ROCm backend". report_stats() is NOT a valid hook either -- it runs inside the custom
            # op, i.e. potentially under capture as well.
            if not _decode_scratch_ready[0] and DECODE_MAX_M > 0:
                _decode_scratch_ready[0] = True
                try:
                    # torch owns it: allocating from the .so put a hipMalloc inside CUDA-graph
                    # capture, and any C++ exception escaping our pybind module gets relabelled by
                    # quark's TileLang exception translator into a bogus "libamdhip64.so not found".
                    _decode_scratch[0] = torch.empty(
                        4 * 64 * 32768, dtype=torch.float32, device=layer.weight.device)
                    # One counter per output block for the fused split-K reduction. The
                    # last arriving block resets its counter, so this is zeroed once.
                    _decode_scratch[1] = torch.zeros(
                        32768 // 128 + 8, dtype=torch.int32, device=layer.weight.device)
                    _ext.set_decode_scratch(_decode_scratch[0].data_ptr(),
                                            _decode_scratch[0].numel() * 4,
                                            _decode_scratch[1].data_ptr())
                    sys.stderr.write(
                        f"[radiance.mxfp4] decode kernel ON (M<={DECODE_MAX_M}), "
                        f"{_decode_scratch[0].numel() * 4 >> 20} MiB split-K scratch\n")
                except Exception as _e:
                    import traceback
                    sys.stderr.write(f"[radiance.mxfp4] decode scratch FAILED: {_e!r}\n"
                                     + traceback.format_exc())
            # Same transpose AiterMxfp4LinearKernel's non-asm branch does: create_weights lays the
            # scale out as [N, K/32] and both the aiter GEMM and this kernel want [K/32, N].
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data.T.contiguous(), requires_grad=False)

            K = layer.weight.shape[1] * 2          # weights are 2 e2m1 codes per byte
            ok = layer_is_supported(layer, K)
            if ok and KERNEL_N is not None and int(layer.weight.shape[0]) not in KERNEL_N:
                ok = False          # bisect: hand this shape to aiter instead
            if ok and KERNEL_NK is not None and (int(layer.weight.shape[0]), K) not in KERNEL_NK:
                ok = False
            # Every layer carries the attribute so apply_weights never branches on hasattr, which
            # dynamo would have to guard. Ineligible layers get a 1-element placeholder, and the
            # op passes 0 for it, which makes the kernel take the per-block-rescale path.
            if ok:
                STATS["fast"] += 1
            else:
                STATS["aiter"] += 1
                sys.stderr.write(
                    f"[radiance.mxfp4] FALLBACK TO AITER (not our kernel): "
                    f"N={tuple(layer.weight.shape)[0]} K={K} "
                    f"ws={tuple(layer.weight_scale.shape)}\n")
            perblock = ok and PERBLOCK_NK is not None and \
                (int(layer.weight.shape[0]), K) in PERBLOCK_NK
            if perblock:
                STATS["fast"] -= 1
                STATS["perblock_forced"] = STATS.get("perblock_forced", 0) + 1
            if ok and not perblock:
                ref = make_row_ref(layer.weight_scale.data)          # folded
            else:
                # 2 elements = our kernel's per-block path; 1 = aiter (kernel cannot serve it)
                ref = torch.zeros(2 if perblock else 1,
                                  dtype=layer.weight_scale.dtype,
                                  device=layer.weight_scale.device)
            layer.radiance_wref = torch.nn.Parameter(ref, requires_grad=False)
            layer.radiance_w4a8_ok = bool(ok)   # record only; never read in the forward

        def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
            y = torch.ops.radiance.mxfp4_linear(
                x, layer.weight, layer.weight_scale, layer.radiance_wref)
            if bias is not None:
                y = y + bias
            return y

    return RadianceMxfp4W4A8LinearKernel


_KERNEL_CLS = None


def kernel_class():
    """The plugin class, built once. Returns None if anything about it is unavailable."""
    global _KERNEL_CLS
    if _KERNEL_CLS is None:
        try:
            _KERNEL_CLS = _make_kernel_class()
        except Exception as e:
            sys.stderr.write(f"[radiance.mxfp4] kernel class unavailable, disabled: {e!r}\n")
            _KERNEL_CLS = False
    return _KERNEL_CLS or None
