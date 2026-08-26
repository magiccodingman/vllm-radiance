"""RADIANCE 4-bit drafter weights: pack the DFlash2 drafter to 4 bits and run it on r4d.

A speculative decode step reads the drafter's weights once and the target's once, and at a decode
batch of eight rows neither GEMM has any arithmetic intensity to speak of -- the drafter's MLP moves
44 MiB of weight against 82 KiB of activation. So the drafter's cost is its weight in bytes, and
halving them is the whole idea: the converted set is 763 MiB per rank per step at fp8 and 405 MiB
at 4.25 bpw -- four bits plus one 32-bit (scale, constant) dword per 128 K.

WHAT IS CONVERTED. The five decoder layers' qkv / o / gate_up / down, and nothing else. `fc`
(5120 x 25600) is declined by the config table on purpose: it is the one drafter linear that sees a
prefill token count every step, and a weight-streaming kernel re-reads the whole weight per chunk.
The grouped convolutions' kernel_projection and the candidate selector are built with
quant_config=None upstream are taken too. The embedding and
the LM head are the TARGET's -- load_dflash_model aliases them -- so they are not the drafter's
traffic and are not touched.

WHY THIS IS SAFE TO DO TO A DRAFTER AND NOT TO A TARGET. A drafter's output is a proposal that the
target verifies token by token, so a weight error costs acceptance, not correctness: every accepted
token is one the target itself produced. That makes the whole question empirical -- what does
acceptance do -- and it is measured on the corpus, not argued.

WHAT IT MEASURED. Round trip on the real weights is 1.01e-1 relative against the fp8 checkpoint's
2.65e-2, and the drafter's weights are Gaussian enough that no quantizer does better at four bits --
an offline sweep put every solver and every grid inside a 0.4% acceptance floor. Against the
block-fp8 path the kernel is 1.82x-2.00x at a decode batch of 8 and 1.45x-1.81x at 64 (the int8
kernel there), at the same GB/s in the first case: the win is the byte count, not the code. End to
end on the BetterBench corpus, four PAIRED compiles per arm, c1: the draft pass falls 9.1% at a
drafter batch of 64 and the step 1.14%.

Acceptance on this stack is BIMODAL and the mode is drawn per compile, in the control arm too --
eight compiles landed near either 2.85-2.93 or 3.04-3.06 -- so a single sample per arm reads a coin
flip as a tax. That is what kept this off by default for a long time; it is not a measured cost.

WHERE IT HOOKS. `begin_draft()` / `end_draft()` bracket the drafter's get_model (patch_dflash_w4.py),
so the load hook claims the drafter's linears and only the drafter's: the target is loaded outside
that bracket and never sees this. Conversion runs inside process_weights_after_loading, before the
preshuffle hook would have shuffled the weight, so there is no shuffled layout to invert. The layer
then carries `_radiance_w4` and its quant_method's apply is replaced; the preshuffle hook checks for
that attribute and leaves the layer alone.
"""
import os
import sys

import torch

USE_R4D = os.environ.get("RADIANCE_USE_R4D", "1") == "1"
# ONE switch for the whole tuned drafter stack, shared with radiance_drafthead: RADIANCE_FAST_DRAFT.
# There is deliberately no separate knob for the weight format, the kernel crossover or the clip
# search -- those were sweeps, they are finished, and the measured answer is baked below. Each half
# of the stack engages only where it applies: the 2-bit head works under any drafter, and this
# module's 4-bit weights only under a dflash one, because patch_dflash_w4 is what brackets the
# drafter's weight load. Under mtp nothing here fires.
ENABLED = USE_R4D and os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"
# Clip-search steps for the quantizer (0 would be plain min/max). 20 is the swept value.
_CLIP = 20

GROUP = 128          # K per (scale, zero) pair; must equal r4d.GEMM_W4_GROUP
_MAX_M = 64          # the kernel's band; overwritten from the library below when it loads
KPB = 64             # K per packed block: four k steps, sixteen bytes per lane
_ROWS = 2048         # output rows converted at a time, to bound the packer's temporaries

# k offset inside a 16-wide WMMA step for nibble 0..7 of a packed dword, before the lane's
# 4*(lane>>4). See r4d_gemm_w4a16_nt_m64.hip: the unpack emits nibbles 0,2,4,6,1,3,5,7 as fragment
# elements 0..7, and element e sits at k = 8*(e>>2) + (e&3).
_KOFF = (0, 8, 1, 9, 2, 10, 3, 11)

# (N, K) -> [(M limit, (WV, SK, MB))] in ascending M. Measured; a shape that is not here is declined
# and stays on the block-fp8 path it would have taken without this module.
#
# The drafter's `fc` (N=5120, K=25600) is deliberately absent. Every other drafter linear only ever
# sees the draft block -- num_reqs * (1 + num_speculative_tokens), 8 to 64 rows -- but
# combine_hidden_states runs fc over the TARGET's token count, which at prefill is a whole 4096-token
# chunk and a 1.07 TFLOP GEMM. That is a different kernel's problem; a weight-streaming kernel capped
# at 64 rows would re-read a 65 MiB weight sixty-four times to serve it.
_CFG: dict = {
    # (WV, SK, MB, NPW, NT), measured on gfx1201 against a pooled working set at each band's top M.
    # The band edges are the batch sizes this serve can produce: M = num_reqs * (1 + nspec), so 8
    # and 64 are the only two the shipped configuration actually runs and the rest interpolate.
    (3072, 5120):  [(16, (1, 4, 1, 1, 1)), (32, (1, 2, 1, 1, 1)),
                    (48, (1, 2, 1, 1, 0)), (64, (1, 8, 2, 4, 0))],   # qkv_proj      1.88x .. 1.15x
    (5120, 2048):  [(16, (2, 4, 1, 1, 1)), (32, (4, 2, 1, 1, 1)),
                    (48, (2, 2, 1, 1, 1)), (64, (1, 4, 2, 4, 1))],   # o_proj        1.82x .. 1.01x
    (17408, 5120): [(16, (2, 10, 1, 1, 1)), (32, (2, 2, 2, 4, 1)),
                    (48, (4, 1, 1, 4, 0)), (64, (1, 1, 2, 4, 0))],   # gate_up_proj  2.00x .. 1.00x
    (5120, 8704):  [(16, (1, 2, 1, 1, 1)), (32, (4, 1, 1, 1, 0)),
                    (48, (1, 4, 1, 4, 0)), (64, (1, 4, 2, 4, 0))],   # down_proj     1.94x .. 1.05x
}

# Above this many rows the int8-WMMA kernel takes over.  Below it the f16 kernel is already at the
# DRAM roofline -- 620 GB/s, 2.0x the block-fp8 path -- so there is nothing for a faster matrix
# instruction to win, and the activation quantiser would only add a launch.  At M=64 the f16 kernel
# is instead at 1.00-1.03x, because v_wmma_f32_16x16x16_f16 runs at 207 TF/s on this part against
# 407 for v_wmma_i32_16x16x16_iu8, and that is the whole reason the second kernel exists.
# NOTE the test below is `m > _A8_M`, so this is the last row the f16 kernel serves, not the first
# row the int8 one does.
_A8_M = 16

# The drafter's two UNQUANTISED projections. `DFlashGroupedConv.kernel_projection` (x10 per draft
# pass) and `candidate_selector.hidden_projection` are ReplicatedLinear with quant_config=None
# upstream, so they are bf16 AND unsharded: every rank reads all 131 MiB of them every step, and a
# Kineto window puts them at 12-15% of the draft pass at ~265 GB/s -- 44% of roofline, because
# rocBLAS lays a weight this small out as a handful of workgroups. Opt-in
# An explicit shape list rather than "anything with a config": these
# are the only bf16 linears left in the drafter and claiming an unexpected one would be silent.
# Mutually exclusive with RADIANCE_SKINNY_GEMM=all by construction -- that hook lives under
# UnquantizedLinearMethod.apply, and a converted layer no longer goes through it.


# (WV, SK, MB, NPW, NT) per M band, measured the same way as _CFG. Filled in by the isolation
# bench; an empty entry means the shape is declined rather than guessed at.
# Ratios below are against the same block-fp8 reference the other tables use, which is NOT what
# these layers actually run -- they run bf16 rocBLAS, ~33.7 us (kernel_projection) and ~89.5 us
# (hidden_projection) in the serve, or ~23.3 / ~6.9 through gemm_bf16_nt_m64 under
# RADIANCE_SKINNY_GEMM=all. Against the skinny path, 4 bits is another ~3x.
_CFG_UNQUANT: dict = {
    (1280, 5120): [(16, (1, 4, 1, 1, 1)), (32, (1, 4, 1, 1, 1)),
                   (48, (1, 4, 1, 1, 1)), (64, (1, 4, 1, 1, 0))],   # 8.19 .. 14.56 us
    (256, 5120):  [(16, (1, 8, 1, 1, 0)), (32, (1, 8, 1, 1, 0)),
                   (48, (2, 4, 1, 1, 0)), (64, (2, 4, 1, 1, 0))],   # 3.61 .. 5.09 us
}
_CFG_UNQUANT_A8: dict = {
    (1280, 5120): [(16, (1, 8, 1, 1, 1)), (32, (1, 4, 1, 1, 1)),
                   (48, (1, 4, 1, 1, 1)), (64, (1, 4, 1, 2, 1))],   # 7.88 .. 9.42 us
    (256, 5120):  [(16, (1, 10, 1, 1, 0)), (32, (1, 10, 1, 1, 0)),
                   (48, (2, 8, 1, 1, 1)), (64, (2, 8, 1, 1, 1))],   # 3.36 .. 4.29 us
}

# The same table for the int8 kernel, measured the same way. Only bands above _A8_M are consulted,
# so the 16-row entries exist to make the band structure identical and are never reached at the
# default threshold. The ratios are against the block-fp8 path at each band's top M.
_CFG_A8: dict = {
    (3072, 5120):  [(16, (1, 4, 1, 1, 1)), (32, (4, 2, 1, 1, 1)),
                    (48, (1, 2, 1, 2, 1)), (64, (4, 2, 1, 2, 1))],   # qkv_proj      1.90x .. 1.83x
    (5120, 2048):  [(16, (1, 4, 1, 1, 1)), (32, (4, 2, 1, 1, 1)),
                    (48, (16, 2, 1, 2, 1)), (64, (1, 2, 1, 2, 0))],  # o_proj        1.70x .. 1.48x
    (17408, 5120): [(16, (2, 10, 1, 1, 1)), (32, (2, 10, 2, 1, 1)),
                    (48, (4, 1, 1, 2, 0)), (64, (1, 1, 2, 4, 0))],   # gate_up_proj  1.98x .. 1.51x
    (5120, 8704):  [(16, (2, 2, 1, 1, 1)), (32, (2, 1, 1, 1, 1)),
                    (48, (2, 1, 1, 1, 1)), (64, (4, 1, 1, 2, 0))],   # down_proj     1.90x .. 1.73x
}

_CFG.update({k: v for k, v in _CFG_UNQUANT.items() if v})
_CFG_A8.update({k: v for k, v in _CFG_UNQUANT_A8.items() if v})

_LOADING_DRAFT = False
_seen: set = set()

try:
    import r4d as _r4d
except Exception as e:
    _r4d = None
    ENABLED = False
    sys.stderr.write(f"[radiance.w4] r4d import failed, disabled: {e!r}\n")

_GEMM = None
if ENABLED:
    _name = _r4d.select("gemm_nt", M=64, K=5120, N=5120, dtype="w4a16")
    if _name is None:
        ENABLED = False
        sys.stderr.write("[radiance.w4] no w4a16 gemm_nt kernel in this r4d build, disabled\n")
    else:
        _GEMM = getattr(_r4d, _name)
        _MAX_M = int(_r4d.GEMM_W4_MAX_M)
        if int(_r4d.GEMM_W4_GROUP) != GROUP:
            ENABLED = False
            sys.stderr.write("[radiance.w4] r4d group %d != %d, disabled\n"
                             % (int(_r4d.GEMM_W4_GROUP), GROUP))

# The int8-WMMA kernel reads the SAME packed weight (see pack() below) and is asked for separately,
# because a build can have one and not the other.
_A8 = _QACT = None
if ENABLED and _A8_M:
    _a8 = _r4d.select("gemm_nt", M=64, K=5120, N=5120, dtype="w4a8")
    _qa = _r4d.select("quant_act", K=5120, dtype="bf16")
    if _a8 is None or _qa is None or int(getattr(_r4d, "GEMM_W4A8_GROUP", 0)) != GROUP:
        sys.stderr.write("[radiance.w4] no w4a8 path in this r4d build, staying on f16\n")
    else:
        _A8, _QACT = getattr(_r4d, _a8), getattr(_r4d, _qa)


# ---- quantizer ---------------------------------------------------------------------------------

def _grid(w, amax):
    """(q, scale, sse) for one candidate clip, per (row, group). Signed, no stored zero point."""
    scale = (amax / 7.0).clamp_min(1e-12)
    s16 = scale.to(torch.float16).float()
    q = torch.round(w / s16[..., None]).clamp(-8, 7)
    err = q * s16[..., None] - w
    return q.to(torch.int8), scale, (err * err).sum(-1)


def quantize(W, group=GROUP, clip_steps=_CLIP):
    """W [N,K] -> (q int8 [N,K] in -8..7, scale f32 [N,K/group]).  w ~= scale * q.

    SYMMETRIC, per output channel per group, with no zero point stored at all.  That is not a
    quality choice -- at four bits the offline sweep put every grid and every solver inside a 0.4%
    acceptance floor -- it is what lets ONE quantised weight feed both kernels.  The int8 kernel
    shifts the nibble into its byte's high half and reads the byte signed, which needs a two's
    complement code; the f16 kernel builds 1024+n and subtracts a constant, which needs offset
    binary and a fixed zero.  A symmetric grid is the one both can express, with the f16 kernel
    converting between them in two instructions (see R4D_GEMM_W4_TWOS).

    The range is not the absolute max.  A group of 128 weights is a sample and its extremes are
    where the sample is thinnest, so spending a level reaching them costs more in the middle than it
    saves at the ends.  This searches shrink factors down to half and keeps, per (row, group)
    INDEPENDENTLY, the smallest squared error against the value the kernel will actually compute --
    the same clip search as before, on a symmetric grid.
    """
    N, K = W.shape
    assert K % group == 0, (K, group)
    out_q = torch.empty((N, K), dtype=torch.int8, device=W.device)
    out_s = torch.empty((N, K // group), dtype=torch.float32, device=W.device)
    for r0 in range(0, N, _ROWS):
        r1 = min(r0 + _ROWS, N)
        w = W[r0:r1].float().reshape(r1 - r0, K // group, group)
        amax = w.abs().amax(-1)
        best = _grid(w, amax)
        for i in range(1, max(clip_steps, 1)):
            cand = _grid(w, amax * (1.0 - 0.5 * i / clip_steps))
            take = cand[2] < best[2]
            best = (torch.where(take[..., None], cand[0].to(torch.int16),
                                best[0].to(torch.int16)).to(torch.int8),
                    torch.where(take, cand[1], best[1]),
                    torch.where(take, cand[2], best[2]))
        out_q[r0:r1] = best[0].reshape(r1 - r0, K)
        out_s[r0:r1] = best[1]
    return out_q, out_s


def dequantize(q, scale, group=GROUP):
    N, K = q.shape
    s16 = scale.to(torch.float16).float()
    return ((q.reshape(N, K // group, group).float() * s16[..., None])
            .reshape(N, K).to(torch.bfloat16))


def pack(q, scale, group=GROUP):
    """(q, scale) -> (wq uint32 [N*K/8], wsz uint32 [N/16 * K/group * 16]).

    The kernel does no shuffling: a lane's sixteen bytes are the four WMMA B fragments it needs, in
    the order its unpack produces them. That order is built here and nowhere else, and BOTH kernels
    read it -- the nibble is the 4-bit two's complement code and the scale dword carries the f16
    scale in its low half with the f16 of -(1024 + 8) in its high half, which is the constant that
    turns the f16 kernel's 1024 + n into the signed value.
    """
    N, K = q.shape
    dev = q.device
    a = (q.to(torch.int16) & 0xF).reshape(N // 16, 16, K // KPB, 4, 16)   # t, r, kb, s, kk
    idx = (4 * torch.arange(2, device=dev)[:, None]
           + torch.tensor(_KOFF, device=dev)[None, :])              # [lh, i] -> kk
    a = a[..., idx.reshape(-1)].reshape(N // 16, 16, K // KPB, 4, 2, 8)
    a = a.permute(0, 2, 4, 1, 3, 5).contiguous().to(torch.int64)    # t, kb, lh, r, s, i
    wq = torch.zeros(a.shape[:-1], dtype=torch.int64, device=dev)
    for i in range(8):
        wq |= a[..., i] << (4 * i)
    wq = wq.to(torch.int32).reshape(-1)

    sc = scale.to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF
    nz = int(torch.tensor(-(1024.0 + 8.0), dtype=torch.float16).view(torch.int16)) & 0xFFFF
    sz = (sc | (nz << 16)).to(torch.int32)                          # [N, K/group]
    sz = sz.reshape(N // 16, 16, K // group).permute(0, 2, 1).contiguous().reshape(-1)
    return wq, sz


def prepare(W, group=GROUP, clip_steps=_CLIP):
    """bf16/f32 [N,K] -> (wq, wsz), packed in chunks of rows so the temporaries stay bounded."""
    N, K = W.shape
    wqs, szs = [], []
    for r0 in range(0, N, _ROWS):
        chunk = W[r0:r0 + _ROWS]
        q, scale = quantize(chunk, group, clip_steps)
        a, b = pack(q, scale, group)
        wqs.append(a); szs.append(b)
    return torch.cat(wqs), torch.cat(szs)


# ---- the linear ------------------------------------------------------------------------------

if ENABLED:
    @torch.library.custom_op("radiance::w4_gemm", mutates_args=())
    def w4_gemm(a: torch.Tensor, wq: torch.Tensor, wsz: torch.Tensor,
                N: int, WV: int, SK: int, MB: int, NPW: int, NT: int) -> torch.Tensor:
        M, K = a.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        _GEMM(a.data_ptr(), wq.data_ptr(), wsz.data_ptr(), c.data_ptr(),
              M, K, N, WV, SK, MB, NPW, NT, torch.cuda.current_stream().cuda_stream)
        return c

    @w4_gemm.register_fake
    def _(a, wq, wsz, N, WV, SK, MB, NPW, NT):
        return torch.empty((a.shape[0], N), dtype=torch.bfloat16, device=a.device)

if ENABLED and _A8 is not None:
    @torch.library.custom_op("radiance::w4a8_gemm", mutates_args=())
    def w4a8_gemm(a: torch.Tensor, wq: torch.Tensor, wsz: torch.Tensor,
                  N: int, WV: int, SK: int, MB: int, NPW: int, NT: int) -> torch.Tensor:
        M, K = a.shape
        q = torch.empty((M, K), device=a.device, dtype=torch.int8)
        sc = torch.empty(M, device=a.device, dtype=torch.float32)
        st = torch.cuda.current_stream().cuda_stream
        _QACT(a.data_ptr(), q.data_ptr(), sc.data_ptr(), M, K, st)
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        _A8(q.data_ptr(), sc.data_ptr(), wq.data_ptr(), wsz.data_ptr(), c.data_ptr(),
            M, K, N, WV, SK, MB, NPW, NT, st)
        return c

    @w4a8_gemm.register_fake
    def _(a, wq, wsz, N, WV, SK, MB, NPW, NT):
        return torch.empty((a.shape[0], N), dtype=torch.bfloat16, device=a.device)


def config_for(n, k, m, table=None):
    """(WV, SK, MB, NPW, NT) for this shape and batch, or None to decline it."""
    if _GEMM is None or not (1 <= m <= int(_r4d.GEMM_W4_MAX_M)):
        return None
    band = (_CFG if table is None else table).get((n, k))
    if band is None:
        return None
    for lim, cfg in band:
        if m <= lim:
            mtiles = (m + 15) // 16
            return (cfg[0], cfg[1], min(cfg[2], mtiles), cfg[3], cfg[4])
    return None


class _W4LinearMethod:
    """Wraps the layer's own quant method: everything but apply() is the original's."""

    def __init__(self, inner, wq, wsz, N, K):
        self._inner = inner
        self.wq, self.wsz, self.N, self.K = wq, wsz, N, K

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _run(self, x2):
        m = x2.shape[0]
        if _A8 is not None and m > _A8_M:
            cfg = config_for(self.N, self.K, m, _CFG_A8)
            if cfg is not None:
                return torch.ops.radiance.w4a8_gemm(
                    x2.to(torch.bfloat16).contiguous(), self.wq, self.wsz, self.N, *cfg)
        cfg = config_for(self.N, self.K, m)
        if cfg is None:
            raise RuntimeError("radiance.w4: no config for N=%d K=%d M=%d"
                               % (self.N, self.K, m))
        return torch.ops.radiance.w4_gemm(
            x2.to(torch.float16).contiguous(), self.wq, self.wsz, self.N, *cfg)

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, self.K)
        m = x2.shape[0]
        if m <= _MAX_M:
            out = self._run(x2)
        else:
            # The one caller past the decode band is the fused context-KV build, which recovers W^T
            # by running an identity through qkv_proj once at first use (patch_dflash_fused_kv_fp8).
            # A weight-streaming kernel re-reads the whole weight per chunk, so this is only
            # acceptable BECAUSE it happens once: ~2 ms per layer at startup. `fc` -- the one
            # drafter linear that sees a prefill token count every step -- is deliberately not
            # converted, so no hot path reaches here.
            out = torch.cat([self._run(x2[i:i + _MAX_M]) for i in range(0, m, _MAX_M)])
        out = out.reshape(*x.shape[:-1], self.N)
        return out if bias is None else out + bias


# The CHECKPOINT's block-fp8 tile, which is a property of the file and not of our quantiser. It
# happens to equal GROUP today; defaulting it to GROUP would silently mis-dequantise every drafter
# weight the moment our group moved.
_FP8_BLOCK = 128


def _dequant_block_fp8(weight, scale_inv, block=_FP8_BLOCK):
    """The checkpoint's fp8 weight and its [N/128, K/128] scale back to bf16."""
    N, K = weight.shape
    s = torch.repeat_interleave(torch.repeat_interleave(scale_inv.float(), block, 0), block, 1)
    return (weight.float() * s[:N, :K]).to(torch.bfloat16)


def maybe_convert(method, layer):
    """Convert one just-loaded block-fp8 linear to 4 bits. True if this layer is now ours."""
    if not (ENABLED and _LOADING_DRAFT):
        return False
    try:
        if not getattr(method, "block_quant", False):
            return False
        w = getattr(layer, "weight", None)
        ws = getattr(layer, "weight_scale_inv", None)
        if ws is None:
            ws = getattr(layer, "weight_scale", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or ws is None:
            return False
        N, K = int(w.shape[0]), int(w.shape[1])
        if N % 16 or K % GROUP or config_for(N, K, 8) is None:
            if (N, K) not in _seen:
                _seen.add((N, K))
                sys.stderr.write("[radiance.w4] declined N=%d K=%d (no measured config)\n" % (N, K))
            return False
        bf = _dequant_block_fp8(w.data, ws.data)
        wq, wsz = prepare(bf)
        del bf
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn,
                                                      device=w.device), requires_grad=False)
        layer._radiance_w4 = (N, K)
        layer._radiance_N = N
        layer.quant_method = _W4LinearMethod(method, wq, wsz, N, K)
        if (N, K) not in _seen:
            _seen.add((N, K))
            sys.stderr.write("[radiance.w4] converted N=%d K=%d  %.1f MiB -> %.1f MiB\n"
                             % (N, K, N * K / 2**20, (wq.numel() + wsz.numel()) * 4 / 2**20))
        return True
    except Exception as e:
        sys.stderr.write(f"[radiance.w4] conversion skipped a layer: {e!r}\n")
        return False


def maybe_convert_bf16(method, layer):
    """Convert one just-loaded bf16 linear to 4 bits. True if this layer is now ours.

    Two populations reach this and the fp8 hook sees neither. Under an fp8 drafter checkpoint it is
    only the layers upstream hardcodes with quant_config=None -- kernel_projection and the candidate
    selector -- listed in _CFG_UNQUANT. Under a BF16 drafter
    checkpoint it is every linear in the drafter, and the four decoder projections are claimed from
    the bf16 weight directly, which is 2.6-2.9% less error on the real weights than going through
    the fp8 checkpoint (measured per layer, layer 0: 1.033e-1 vs 1.061e-1 on q_proj).

    Either way the gate is a measured config, so a shape nobody has swept is declined rather than
    guessed at.
    """
    if not (ENABLED and _LOADING_DRAFT):
        return False
    try:
        w = getattr(layer, "weight", None)
        if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
            return False
        N, K = int(w.shape[0]), int(w.shape[1])
        if N % 16 or K % GROUP or config_for(N, K, 8) is None:
            return False
        wq, wsz = prepare(w.data)
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device),
                                          requires_grad=False)
        layer._radiance_w4 = (N, K)
        layer._radiance_N = N
        layer.quant_method = _W4LinearMethod(method, wq, wsz, N, K)
        if ("bf16", N, K) not in _seen:
            _seen.add(("bf16", N, K))
            sys.stderr.write("[radiance.w4] converted bf16 N=%d K=%d  %.1f MiB -> %.1f MiB\n"
                             % (N, K, N * K * 2 / 2**20, (wq.numel() + wsz.numel()) * 4 / 2**20))
        return True
    except Exception as e:
        sys.stderr.write(f"[radiance.w4] bf16 conversion skipped a layer: {e!r}\n")
        return False


def begin_draft():
    global _LOADING_DRAFT
    _LOADING_DRAFT = True


def end_draft():
    global _LOADING_DRAFT
    _LOADING_DRAFT = False


def install_load_hook():
    """Wrap Fp8LinearMethod.process_weights_after_loading so a drafter linear is packed to 4 bits
    as soon as its fp8 weight exists. Idempotent; must run before the model loads."""
    if not ENABLED:
        return
    import vllm.model_executor.layers.quantization.fp8 as _f8
    if getattr(_f8.Fp8LinearMethod, "_radiance_w4_wrapped", False):
        return
    _orig = _f8.Fp8LinearMethod.process_weights_after_loading

    def _wrapped(self, layer):
        _orig(self, layer)
        maybe_convert(self, layer)

    _f8.Fp8LinearMethod.process_weights_after_loading = _wrapped
    _f8.Fp8LinearMethod._radiance_w4_wrapped = True
    sys.stderr.write("[radiance.w4] 4-bit drafter hook installed\n")

    # With a bf16 drafter checkpoint this is the main path, not a side one: every drafter linear
    # arrives here rather than at the fp8 hook above.
    import vllm.model_executor.layers.linear as _lin
    if not getattr(_lin.UnquantizedLinearMethod, "_radiance_w4_wrapped", False):
        _orig_u = _lin.UnquantizedLinearMethod.process_weights_after_loading

        def _wrapped_u(self, layer):
            _orig_u(self, layer)
            maybe_convert_bf16(self, layer)

        _lin.UnquantizedLinearMethod.process_weights_after_loading = _wrapped_u
        _lin.UnquantizedLinearMethod._radiance_w4_wrapped = True
        sys.stderr.write("[radiance.w4] bf16 drafter hook installed\n")
    sys.stderr.flush()
