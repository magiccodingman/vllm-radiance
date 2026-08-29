"""RADIANCE fused gated-delta-net prefill dispatch.

Wraps R4D's `gdn_chunk_scan_k128_v128_c64_bf16`, which replaces three of the five Triton kernels
FLA runs per GDN layer during prefill -- recompute_w_u_fwd (the WY representation),
chunk_gated_delta_rule_fwd_h (the recurrent state scan) and chunk_fwd_o (the output) -- with one
kernel that keeps the recurrent state in WMMA accumulators across the whole chunk loop. h, w, u and
v_new never reach HBM, which is where the win comes from: the same maths reads about a third of the
bytes.

The cumsum, the K-gram and the triangular inverse (solve_tril) are NOT replaced; they stay on the
Triton path and their outputs feed this kernel.

Enabled with the rest of the library by RADIANCE_USE_R4D. The kernel name carries the geometry it
is compiled for, and this
module checks a model against it before dispatching: anything else falls back to FLA silently and
per call, so a model with a different GDN shape keeps working.
"""
import os
import sys

import torch

# RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py): with
# it off this module behaves exactly as it would in an image built without the library.
USE_R4D = (
    os.environ.get("RADIANCE_USE_R4D", "1") == "1"
    and os.environ.get("RADIANCE_USE_R4D_GDN", "1") == "1"
)
ENABLED = USE_R4D

try:
    import r4d as _r4d
except Exception as e:  # library missing or failed to load: stay on FLA
    _r4d = None
    ENABLED = False
    sys.stderr.write(f"[radiance.gdn] r4d import failed, fused GDN disabled: {e!r}\n")

# The geometry the kernel is compiled for, read from the library rather than copied here -- r4d
# reports it, and one source for it is the point. The entry point rejects a mismatch anyway, but
# checking first keeps the fallback silent instead of raising out of a forward pass.
HEAD_K = int(_r4d.GDN_HEAD_K) if _r4d is not None else 128
HEAD_V = int(_r4d.GDN_HEAD_V) if _r4d is not None else 128
CHUNK = int(_r4d.GDN_CHUNK) if _r4d is not None else 64
CONV_WIDTH = 4


def _bind(op: str, **geometry):
    """The entry point this build has for a geometry, or None if it has none."""
    if _r4d is None or not USE_R4D:
        return None   # asking would put a query in the report for a library that is switched off
    name = _r4d.select(op, **geometry)
    return getattr(_r4d, name) if name else None


# One handle per stage, resolved once. The entry-point names carry the geometry they are compiled
# for, so writing them here would be a second copy of the constants just read out of the library --
# ask it which kernel it has instead, and the names stay in the registry next to the kernels.
_CHUNK_SCAN = _bind("gdn_chunk_scan", head_k=HEAD_K, head_v=HEAD_V, chunk=CHUNK)
_CONV_PREP = _bind("gdn_conv_prep", conv_width=CONV_WIDTH, head_k=HEAD_K, head_v=HEAD_V,
                   chunk=CHUNK)
_CONV_UPDATE = _bind("gdn_conv_update", conv_width=CONV_WIDTH, head_k=HEAD_K, head_v=HEAD_V)
_KKT_SOLVE = _bind("gdn_kkt_solve", head_k=HEAD_K, chunk=CHUNK)
_RECURRENT_UPDATE = _bind("gdn_recurrent_update", head_k=HEAD_K, head_v=HEAD_V)
_FUSED_UPDATE = _bind("gdn_fused_update", conv_width=CONV_WIDTH, head_k=HEAD_K, head_v=HEAD_V)
# The fused decode step (conv -> grid barrier -> recurrent in ONE launch) removes one kernel
# boundary per GDN layer per forward. Bit-identical to the pair by construction. Off by default
# until the serving A/B has run; needs a build whose registry has gdn_fused_update.
FUSED_UPDATE_ON = os.environ.get("RADIANCE_GDN_FUSED_UPDATE", "0") == "1" and _FUSED_UPDATE is not None
# The barrier counter must exist BEFORE any CUDA-graph capture replays the kernel, and must NOT
# be allocated at import -- that grabs a CUDA context before vLLM sets the device and breaks its
# memory snapshot (the split-K decode scratch learned the same lesson; it allocates at weight
# load). init_fused_counter() is called from the post-load hook in radiance_gdnmerge.merge_model.
_FUSED_CNT = None


def init_fused_counter():
    global _FUSED_CNT, FUSED_UPDATE_ON
    if FUSED_UPDATE_ON and _FUSED_CNT is None:
        try:
            _FUSED_CNT = torch.zeros(1, dtype=torch.int32, device="cuda")
        except Exception as _e:                      # noqa: BLE001
            FUSED_UPDATE_ON = False
            sys.stderr.write(f"[radiance.gdn] fused update disabled, no counter: {_e!r}\n")
_GATED_RMSNORM = _bind("gdn_gated_rmsnorm", channels=HEAD_V)

if ENABLED and _CHUNK_SCAN is None:
    # Only reachable if the registry and the compiled-in constants disagree, which is a build bug
    # rather than a model this build does not cover -- but it is better said than launched.
    ENABLED = False
    sys.stderr.write(
        f"[radiance.gdn] no gdn_chunk_scan kernel for head_k {HEAD_K} head_v {HEAD_V} "
        f"chunk {CHUNK}, fused GDN disabled\n"
    )

if ENABLED:
    sys.stderr.write(
        f"[radiance.gdn] gdn_chunk_scan ENABLED (head_k {HEAD_K}, head_v {HEAD_V}, "
        f"chunk {CHUNK}; replaces wy + delta_h + chunk_o)\n"
    )

_warned = False


def _bail(why: str):
    """Report the first fallback and then stay quiet: it is per-call, so it must not spam."""
    global _warned
    if not _warned:
        _warned = True
        sys.stderr.write(f"[radiance.gdn] falling back to FLA for this shape: {why}\n")
    return None


def fused_prefill(q, k, v, A, g, beta, scale, initial_state, output_final_state, cu_seqlens,
                  core_attn_out, out=None):
    """Run the fused scan, or return None to let the caller keep the FLA path.

    q, k: [1, T, Hg, K] bf16   v: [1, T, H, V] bf16   A: [1, T, H, chunk] bf16 (solve_tril output)
    g, beta: [1, T, H] fp32    initial_state: [N, H, V, K] fp32    cu_seqlens: [N+1] int32
    Returns (o, final_state) with o shaped like v.
    """
    if cu_seqlens is None:
        return _bail("no cu_seqlens (the kernel is varlen-only)")
    if initial_state is None or not output_final_state:
        return _bail("the kernel always reads an initial state and writes a final one")
    if q.shape[0] != 1 or q.shape[-1] != HEAD_K or v.shape[-1] != HEAD_V:
        return _bail(f"head dims {q.shape[-1]}/{v.shape[-1]}, batch {q.shape[0]}")
    if A.shape[-1] != CHUNK:
        return _bail(f"chunk size {A.shape[-1]}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 \
            or A.dtype != torch.bfloat16:
        return _bail(f"dtypes q {q.dtype} k {k.dtype} v {v.dtype} A {A.dtype}")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return _bail(f"gate dtypes g {g.dtype} beta {beta.dtype}")
    if initial_state.dtype != torch.float32:
        return _bail(f"state dtype {initial_state.dtype}")
    # Every tensor is indexed by raw stride arithmetic, so a non-contiguous view would be read wrong
    # rather than slowly.
    for name, t in (("q", q), ("k", k), ("v", v), ("A", A), ("g", g), ("beta", beta),
                    ("initial_state", initial_state)):
        if not t.is_contiguous():
            return _bail(f"{name} is not contiguous")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        return _bail("cu_seqlens must be contiguous int32")

    num_seqs = cu_seqlens.numel() - 1
    if initial_state.shape[0] != num_seqs:
        return _bail(f"{initial_state.shape[0]} states for {num_seqs} sequences")

    H, Hg = v.shape[-2], q.shape[-2]
    if H % Hg != 0:
        return _bail(f"{H} v heads is not a multiple of {Hg} k heads")

    # chunk_fwd_o writes into the caller's buffer when it is given one, and the caller reads that
    # buffer rather than the returned tensor -- so the fused kernel has to write the same place.
    # `out` is the already-shaped form of that: the layer hook slices core_attn_out itself, because
    # the buffer is padded to the cudagraph batch and the flat slice below only lines up when it
    # is not.
    if out is not None:
        o = out
    elif core_attn_out is not None:
        if core_attn_out.numel() < v.numel():
            return _bail("core_attn_out smaller than v")
        o = core_attn_out[: v.numel()].view(*v.shape)
        if not o.is_contiguous():
            return _bail("core_attn_out view is not contiguous")
    else:
        o = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    _CHUNK_SCAN(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), A.data_ptr(), g.data_ptr(), beta.data_ptr(),
        initial_state.data_ptr(), o.data_ptr(), final_state.data_ptr(), cu_seqlens.data_ptr(),
        num_seqs, H, Hg, HEAD_K, HEAD_V, CHUNK, float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    return o, final_state


# =================================================================================================
# The whole layer: conv_prep -> kkt_solve -> chunk_scan on the prefill path, conv_update ->
# recurrent_update on the decode path. Nothing below calls a Triton kernel.
#
# This is installed as a prologue on QwenGatedDeltaNetAttention._forward_core (patch_r4d.py):
# it either handles the step and returns True, or returns False and lets the original body run
# unchanged. The gate is deliberately narrow -- the two shapes a step of this serve actually takes,
# a chunked-prefill batch and a speculative-decode batch -- because the shapes it does not take
# still have a correct implementation one return away, and because a wrong answer here is not
# recoverable downstream.
#
# What the fused path is worth over calling the same five R4D kernels through the op-level seams:
# the conv output (34 MB per layer per rank at this model's chunk size) never reaches HBM, which
# is most of the preamble's cost.
# =================================================================================================

ALL = ENABLED
# which step kinds the fused path takes; "both" in production, one of them when bisecting
PATHS = os.environ.get("RADIANCE_GDN_PATHS", "both")
# The gated RMS norm is off by default until it is priced in a trace: it sits INSIDE the compiled
# region, where vLLM's fuse_norm_quant pass may already have claimed it, and replacing a fused
# norm+quant with this norm plus a separate quant would be a regression dressed as a win.
NORM_FUSE = ALL and os.environ.get("RADIANCE_GDN_NORM", "0") == "1"
CHECK = os.environ.get("RADIANCE_GDN_CHECK", "0") == "1"
SOFTPLUS_THRESHOLD = 20.0          # FLA's, and the value both kernels are validated against
_fallbacks = {}
_seen = set()


def _first(kind: str):
    """Announce the first step of each kind the fused path takes, so a log shows what ran."""
    if kind not in _seen:
        _seen.add(kind)
        sys.stderr.write(f"[radiance.gdn] all-R4D {kind} path live\n")


def _fb(why: str):
    """Count why a step went back to the original path. Reported by fallback_report()."""
    _fallbacks[why] = _fallbacks.get(why, 0) + 1
    if _fallbacks[why] == 1:
        sys.stderr.write(f"[radiance.gdn] step not handled by the fused path: {why}\n")
    return False


def fallback_report():
    return dict(_fallbacks)


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _metadata(self):
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = md.get(self.prefix)
    return md


_conv_dim_first = None


def _conv_state_dim_first():
    global _conv_dim_first
    if _conv_dim_first is None:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        _conv_dim_first = bool(is_conv_state_dim_first())
    return _conv_dim_first


def _gate_params(self):
    """fp32 A_log and dt_bias, converted once and cached on the layer.

    A_log is declared fp32 but dt_bias is `torch.ones(...)` under vLLM's default dtype, so it
    arrives bf16 -- and a bf16 tensor read as fp32 is not a small error, it is noise. They are one
    value per head and never change, so the conversion is a one-off rather than a per-step cast.
    """
    p = getattr(self, "_radiance_gate_params", None)
    if p is None:
        p = (self.A_log.detach().float().contiguous(),
             self.dt_bias.detach().float().contiguous())
        self._radiance_gate_params = p
    return p


def _geometry_ok(self):
    return (self.head_k_dim == HEAD_K and self.head_v_dim == HEAD_V
            and self.conv_kernel_size == 4)


def conv_prep(x, conv_w, conv_bias, conv_state, cache_idx, has_init, a, b, A_log, dt_bias,
              cu, num_seqs, T, H, Hg):
    """causal conv + q/k/v split + qk l2norm + gating + per-chunk cumsum, in one kernel."""
    dev, dt = x.device, torch.bfloat16
    q = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    k = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    v = torch.empty((T, H, HEAD_V), device=dev, dtype=dt)
    g = torch.empty((T, H), device=dev, dtype=torch.float32)
    beta = torch.empty((T, H), device=dev, dtype=torch.float32)
    _CONV_PREP(
        x.data_ptr(), x.stride(0), conv_w.data_ptr(),
        conv_bias.data_ptr() if conv_bias is not None else 0,
        conv_state.data_ptr(), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        cache_idx.data_ptr(), cache_idx.stride(0),
        has_init.data_ptr() if has_init is not None else 0,
        a.data_ptr(), b.data_ptr(), a.stride(0), a.dtype == torch.bfloat16,
        A_log.data_ptr(), dt_bias.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        cu.data_ptr(), num_seqs, T, H, Hg, HEAD_K, HEAD_V, 4, SOFTPLUS_THRESHOLD, _stream())
    return q, k, v, g, beta


def conv_update(x, conv_w, conv_bias, conv_state, state_len_max, cache_idx, num_accepted,
                cu, num_seqs, T, H, Hg, max_query_len):
    """The decode convolution: rolling speculative window, writing q/k/v in their own layouts."""
    dev, dt = x.device, torch.bfloat16
    q = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    k = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    v = torch.empty((T, H, HEAD_V), device=dev, dtype=dt)
    _CONV_UPDATE(
        x.data_ptr(), x.stride(0), conv_w.data_ptr(),
        conv_bias.data_ptr() if conv_bias is not None else 0,
        conv_state.data_ptr(), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        state_len_max, cache_idx.data_ptr(), cache_idx.stride(0),
        num_accepted.data_ptr() if num_accepted is not None else 0,
        q.data_ptr(), k.data_ptr(), v.data_ptr(), cu.data_ptr(),
        num_seqs, H, Hg, HEAD_K, HEAD_V, CONV_WIDTH, max_query_len, _stream())
    return q, k, v


def fused_update(x, conv_w, conv_bias, conv_state, state_len_max, cache_idx, num_accepted,
                 cu, num_seqs, T, H, Hg, max_query_len, a, b, A_log, dt_bias, ssm_state, o,
                 sidx, scale):
    """conv_update + recurrent_update as one launch. Arguments are the union of the pair's."""
    dev, dt = x.device, torch.bfloat16
    q = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    k = torch.empty((T, Hg, HEAD_K), device=dev, dtype=dt)
    v = torch.empty((T, H, HEAD_V), device=dev, dtype=dt)
    _FUSED_UPDATE(
        x.data_ptr(), x.stride(0), conv_w.data_ptr(),
        conv_bias.data_ptr() if conv_bias is not None else 0,
        conv_state.data_ptr(), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        state_len_max, cache_idx.data_ptr(), cache_idx.stride(0),
        num_accepted.data_ptr() if num_accepted is not None else 0,
        cu.data_ptr(), num_seqs, H, Hg, HEAD_K, HEAD_V, CONV_WIDTH, max_query_len,
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        a.data_ptr(), b.data_ptr(), a.stride(0), a.dtype == torch.bfloat16,
        A_log.data_ptr(), dt_bias.data_ptr(),
        ssm_state.data_ptr(), ssm_state.stride(0), ssm_state.stride(1),
        o.data_ptr(), sidx.data_ptr(), sidx.stride(0),
        float(scale), SOFTPLUS_THRESHOLD, _FUSED_CNT.data_ptr(), _stream())


def kkt_solve(k, beta, g, cu, num_seqs, T, H, Hg):
    """A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk; the fp32 gram stays in LDS."""
    A = torch.empty((T, H, CHUNK), device=k.device, dtype=torch.bfloat16)
    _KKT_SOLVE(
        k.data_ptr(), beta.data_ptr(), g.data_ptr(), A.data_ptr(), cu.data_ptr(),
        num_seqs, T, H, Hg, HEAD_K, CHUNK, _stream())
    return A


def _norm_params(self):
    """(fp32 weight, eps, act code) for the layer's gated RMS norm, or None if it is not the shape
    the kernels implement. Cached: the weight is 128 values and never changes."""
    p = getattr(self, "_radiance_norm_params", "miss")
    if p == "miss":
        n = getattr(self, "norm", None)
        ok = (n is not None and getattr(n, "group_size", None) is None
              and getattr(n, "norm_before_gate", False) is True
              and getattr(n, "bias", None) is None
              and n.weight.numel() == HEAD_V
              and getattr(n, "activation", "silu") in ("silu", "swish", "sigmoid"))
        p = ((n.weight.detach().float().contiguous(), float(n.eps),
              1 if n.activation == "sigmoid" else 0) if ok else None)
        self._radiance_norm_params = p
    return p


def output_norm(self, core_attn_out, z):  # noqa: C901
    """The layer's gated RMS norm, in R4D. Returns the normalised tensor, or None to leave it to
    the original path.

    On the decode path the recurrent kernel has already applied it (its workgroup owns the whole
    128-channel row, so it costs a reduction rather than a kernel), and this returns the buffer
    untouched. On the prefill path it cannot be fused -- the chunked scan's workgroup owns 64 of
    the 128 channels of a row -- so it launches one kernel over the rows.
    """
    self.__dict__.pop("_radiance_z", None)          # never let the stash outlive the step
    if not NORM_FUSE or _r4d is None:
        return None
    if self.__dict__.pop("_radiance_normed", False):
        return core_attn_out                       # already normalised inside the decode kernel
    np_ = _norm_params(self)
    if np_ is None:
        return None
    w, eps, act = np_
    x = core_attn_out.reshape(-1, HEAD_V)
    zz = z.reshape(-1, HEAD_V)
    if (x.dtype != torch.bfloat16 or zz.dtype != torch.bfloat16
            or x.stride(1) != 1 or zz.stride(1) != 1):
        return None
    out = torch.empty_like(x)
    _GATED_RMSNORM(
        x.data_ptr(), zz.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0],
        x.stride(0), zz.stride(0), out.stride(0), HEAD_V, eps, act, _stream())
    return out


def recurrent_update(q, k, v, a, b, A_log, dt_bias, ssm_state, o, cu, sidx, num_accepted,
                     num_seqs, H, Hg, scale, z_gate=None, norm=None):
    # The slot and head strides come from the tensor: vLLM pads the mamba page to the attention
    # page size, so a slot is wider than H*V*K and deriving it from the shape reads the wrong
    # memory for every slot but the first.
    _RECURRENT_UPDATE(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        a.stride(0), a.dtype == torch.bfloat16, A_log.data_ptr(), dt_bias.data_ptr(),
        ssm_state.data_ptr(), ssm_state.stride(0), ssm_state.stride(1),
        o.data_ptr(), cu.data_ptr(), sidx.data_ptr(), sidx.stride(0),
        num_accepted.data_ptr() if num_accepted is not None else 0,
        z_gate.data_ptr() if z_gate is not None else 0,
        norm[0].data_ptr() if norm is not None else 0,
        norm[1] if norm is not None else 0.0, norm[2] if norm is not None else 0,
        num_seqs, H, Hg, HEAD_K, HEAD_V, float(scale), SOFTPLUS_THRESHOLD, _stream())


def _plan(self, mixed_qkv, b, a, core_attn_out):
    """Decide whether this step is one the R4D path covers, and gather what it needs.

    Everything that could raise on an unexpected metadata shape happens here, BEFORE any kernel
    runs, so an unhandled case costs a fallback rather than a half-updated state cache.
    """
    def no(why):
        _fb(why)
        return None

    md = _metadata(self)
    if md is None:
        return None                                   # warm-up pass: the caller compiles kernels
    if not _geometry_ok(self):
        return no(f"geometry head_k {self.head_k_dim} head_v {self.head_v_dim} "
                  f"conv {self.conv_kernel_size}")
    if mixed_qkv.dtype != torch.bfloat16:
        return no(f"mixed_qkv dtype {mixed_qkv.dtype}")

    T = int(md.num_actual_tokens)
    if T == 0:
        return None
    kv = self.kv_cache
    conv_state = kv[0] if _conv_state_dim_first() else kv[0].transpose(-1, -2)
    ssm_state = kv[1]
    if ssm_state.dtype != torch.float32:
        return no(f"ssm state dtype {ssm_state.dtype}")
    if ssm_state.stride(2) != ssm_state.shape[3] or ssm_state.stride(3) != 1:
        return no(f"ssm state [V,K] block is not packed: strides {tuple(ssm_state.stride())}")
    if conv_state.dtype != torch.bfloat16:
        return no(f"conv state dtype {conv_state.dtype}")
    if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype != a.dtype:
        return no(f"gate dtypes a {a.dtype} b {b.dtype}")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        return no("gate tensors are not contiguous along the head axis")
    if self.conv1d.weight.dtype != torch.bfloat16:
        return no(f"conv weight dtype {self.conv1d.weight.dtype}")
    if self.conv1d.bias is not None and self.conv1d.bias.dtype != torch.bfloat16:
        return no(f"conv bias dtype {self.conv1d.bias.dtype}")

    spec = md.spec_sequence_masks
    have_spec = spec is not None and md.num_spec_decodes > 0
    have_prefill = md.num_prefills > 0

    # A step of this serve is one of three shapes. With MTP the spec mask is present on a PREFILL
    # step too -- the drafter's sequences ride along -- so "prefill" here means prefill tokens plus
    # a spec group, and a batch with no spec at all is the rarer case, not the common one.
    if have_spec and not have_prefill and md.num_decodes == 0:
        kind = "decode"
    elif have_prefill and md.num_decodes == 0:
        kind = "prefill+spec" if have_spec else "prefill"
    else:
        return no(f"mixed batch: prefills {md.num_prefills} decodes {md.num_decodes} "
                  f"spec {spec is not None}")

    if kind == "decode" and PATHS not in ("both", "decode"):
        return no("decode path disabled by RADIANCE_GDN_PATHS")
    if kind != "decode" and PATHS not in ("both", "prefill"):
        return no("prefill path disabled by RADIANCE_GDN_PATHS")

    sidx = None
    if have_spec:
        sidx = md.spec_state_indices_tensor
        if sidx is None or sidx.dim() != 2:
            return no("spec state indices are not [seq, candidate]")
        if kind != "decode" and (md.spec_token_indx is None or md.non_spec_token_indx is None):
            return no("no spec/non-spec token index on the metadata")
    cu = None
    if kind != "decode":
        cu = md.prefill_query_start_loc
        if cu is None or cu.numel() != md.num_prefills + 1:
            return no("prefill_query_start_loc does not match num_prefills")
        if md.prefill_state_indices is None or md.prefill_has_initial_state is None:
            return no("no prefill state indices on the metadata")
    return (kind, T, conv_state, ssm_state, md, (sidx, cu))


def forward_core_fused(self, mixed_qkv, b, a, core_attn_out) -> bool:
    """Handle this step entirely in R4D, or return False and leave it to the original path."""
    if not ALL or _r4d is None:
        return False
    try:
        plan = _plan(self, mixed_qkv, b, a, core_attn_out)
    except Exception as e:                            # an unexpected metadata shape is a fallback
        return _fb(f"{type(e).__name__} while planning: {e}")
    if plan is None:
        return False

    # Take the stashed z and DROP the attribute in the same breath: holding it on the layer keeps
    # one per-step tensor alive per layer between steps, which at this model's chunk size is about
    # 25 MB x 48 layers of memory the allocator can never reuse -- enough to OOM a serve that runs
    # at 0.95 utilisation.
    z_stash = self.__dict__.pop("_radiance_z", None)
    kind, T, conv_state, ssm_state, md, (sidx, cu) = plan
    H = self.num_v_heads // self.tp_size
    Hg = self.num_k_heads // self.tp_size
    conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    A_log, dt_bias = _gate_params(self)
    mixed_qkv, a, b = mixed_qkv[:T], a[:T], b[:T]

    # ---- speculative decode: every token is a candidate of a spec sequence -------------------
    if kind == "decode":
        nseq = md.num_spec_decodes
        maxq = sidx.size(-1)
        cu = md.spec_query_start_loc[: nseq + 1]
        o = core_attn_out[:T].view(T, H, HEAD_V)
        if FUSED_UPDATE_ON:
            fused_update(mixed_qkv, conv_w, self.conv1d.bias, conv_state,
                         (4 - 1) + (maxq - 1), sidx[:, 0][:nseq], md.num_accepted_tokens,
                         cu, nseq, T, H, Hg, maxq, a, b, A_log, dt_bias, ssm_state, o, sidx,
                         HEAD_K ** -0.5)
            _first("decode(fused)")
            return True
        q, k, v = conv_update(mixed_qkv, conv_w, self.conv1d.bias, conv_state,
                              (4 - 1) + (maxq - 1), sidx[:, 0][:nseq],
                              md.num_accepted_tokens, cu, nseq, T, H, Hg, maxq)
        _first("decode")
        # Fold the layer's gated RMS norm into this kernel when z is visible: the workgroup owns
        # the whole 128-channel row here, so it is a reduction rather than a second pass. z is
        # stashed by the ROCm entry point (patch_r4d.py); without it the norm stays where it
        # was and output_norm() takes the prefill route.
        zg = z_stash if NORM_FUSE else None
        nrm = _norm_params(self) if NORM_FUSE else None
        if (zg is not None and nrm is not None and zg.dtype == torch.bfloat16
                and zg.shape[0] >= T and zg.stride(-1) == 1
                and zg.reshape(-1, HEAD_V).stride(0) == HEAD_V):
            recurrent_update(q, k, v, a, b, A_log, dt_bias, ssm_state, o, cu, sidx,
                             md.num_accepted_tokens, nseq, H, Hg, HEAD_K ** -0.5,
                             z_gate=zg[:T], norm=nrm)
            self._radiance_normed = True
        else:
            recurrent_update(q, k, v, a, b, A_log, dt_bias, ssm_state, o, cu, sidx,
                             md.num_accepted_tokens, nseq, H, Hg, HEAD_K ** -0.5)
        return True

    # ---- chunked prefill, with or without a spec group riding along --------------------------
    _first(kind)
    spec_o = None
    if kind == "prefill+spec":
        # The spec candidates are interleaved with the prefill tokens, so they are peeled out by
        # index, run through the decode pair, and written back by index at the end -- the same
        # split the original does, minus its extra merge buffer.
        si, ni = md.spec_token_indx, md.non_spec_token_indx
        nspec = md.num_spec_decodes
        maxq = sidx.size(-1)
        scu = md.spec_query_start_loc[: nspec + 1]
        x_spec = mixed_qkv.index_select(0, si)
        a_spec, b_spec = a.index_select(0, si), b.index_select(0, si)
        ts = x_spec.shape[0]
        sq, sk, sv = conv_update(x_spec, conv_w, self.conv1d.bias, conv_state,
                                 (4 - 1) + (maxq - 1), sidx[:, 0][:nspec],
                                 md.num_accepted_tokens, scu, nspec, ts, H, Hg, maxq)
        spec_o = torch.empty((ts, H, HEAD_V), device=mixed_qkv.device, dtype=torch.bfloat16)
        recurrent_update(sq, sk, sv, a_spec, b_spec, A_log, dt_bias, ssm_state, spec_o, scu,
                         sidx, md.num_accepted_tokens, nspec, H, Hg, HEAD_K ** -0.5)
        mixed_qkv = mixed_qkv.index_select(0, ni)
        a, b = a.index_select(0, ni), b.index_select(0, ni)

    nseq = md.num_prefills
    tp = mixed_qkv.shape[0]
    q, k, v, g, beta = conv_prep(
        mixed_qkv, conv_w, self.conv1d.bias, conv_state,
        md.non_spec_state_indices_tensor, md.has_initial_state, a, b,
        A_log, dt_bias, cu, nseq, tp, H, Hg)
    A = kkt_solve(k, beta, g, cu, nseq, tp, H, Hg)
    initial_state = ssm_state[md.prefill_state_indices]
    initial_state[~md.prefill_has_initial_state, ...] = 0
    o_buf = (core_attn_out[:tp].view(1, tp, H, HEAD_V) if spec_o is None
             else torch.empty((1, tp, H, HEAD_V), device=mixed_qkv.device,
                              dtype=torch.bfloat16))
    out = fused_prefill(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), A.unsqueeze(0),
                        g.unsqueeze(0), beta.unsqueeze(0), HEAD_K ** -0.5, initial_state,
                        True, cu, None, out=o_buf)
    if out is None:
        raise RuntimeError("radiance_gdn: chunk_scan declined a shape the preamble accepted")
    ssm_state[md.prefill_state_indices] = out[1].to(ssm_state.dtype)
    if spec_o is not None:
        dst = core_attn_out[:T]
        dst.index_copy_(0, md.spec_token_indx, spec_o)
        dst.index_copy_(0, md.non_spec_token_indx, o_buf.squeeze(0))
    return True
