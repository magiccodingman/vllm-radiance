"""FP8 residual-stream contract: fuse each RowParallel linear's post-all-reduce epilogue.

WHAT. In the stock graph every o_proj/out_proj/down_proj does: all-reduce -> traced
(residual add + Gemma rms_norm) -> traced per-token fp8 quant -> the next radiance linear.
At decode M=8 that epilogue is 2-3 inductor kernels tiled for the M=8192 compile hint
(XBLOCK=1: ~4.4 us active + 2 dispatch gaps per site, 128 sites per step). This module
replaces it with ONE custom op per site: AR (r4d, unchanged) + radiance_add_rms_quant, a HIP
kernel measured BIT-IDENTICAL to the traced reference on every production shape (q codes,
scales, and residual all exact — see ~/mxfp4_work escha? no: scratchpad arnq_test.py) at
~2.4 us/launch.

CONTRACT. Layers exchange (partial_down, residual) instead of (hidden, residual): layer i's
down_proj stops reducing, and layer i+1 STARTS by finishing the epilogue with its OWN
input_layernorm weight — no cross-layer weight reach, and the model loop's 2-tuple contract
survives because a partial sum has the same shape as a reduced one. Inside a layer, the
epilogue's (q_fp8, scale) tuple flows through vLLM's Linear.forward untouched (verified: both
Column/RowParallel pass input_ straight to quant_method.apply) into our apply_weights, which
unpacks it onto the pq kernel path. The LAST layer keeps the stock contract end-to-end so
model.norm and the dflash drafter's last_hidden_states see exactly what they see today.

GUARDS (install skips, loudly, if any fail): tp>1, pp==1, no aux_hidden_state_layers, no
layer_scale, no sequence-parallel MoE, MLP.expert_gate is None, gdnmerge merged the GDN
in_proj (its forward is ours to make tuple-aware), and every consumer linear of a streamed
site runs the radiance W4A8 kernel (radiance_wref present).

RADIANCE_FP8_STREAM=1 enables. Changes the traced graph -> cache dir must be keyed (-fp8s).
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_FP8_STREAM", "0") == "1"


def _log(msg):
    sys.stderr.write(f"[radiance.fp8stream] {msg}\n")
    sys.stderr.flush()


def _ext():
    import radiance_mxfp4
    return radiance_mxfp4._ext


_COMM = [None]


def _radiance_comm():
    """The RadianceAllreduce instance, if it exists, is enabled, and carries the nq kernel."""
    if _COMM[0] is None:
        try:
            from vllm.distributed.parallel_state import get_tp_group
            comm = getattr(get_tp_group().device_communicator, "radiance_comm", None)
            ok = (comm is not None and not comm.disabled
                  and hasattr(comm._ext, "ar_oneshot_2rank_exact_nq"))
            _COMM[0] = comm if ok else False
        except Exception:                           # noqa: BLE001
            _COMM[0] = False
    return _COMM[0] or None


@torch.library.custom_op("radiance::ar_add_rms_quant", mutates_args=())
def ar_add_rms_quant(y: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                     eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = y.shape
    residual = residual.contiguous()
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=y.device)
    sc = torch.empty((M,), dtype=torch.float32, device=y.device)
    ro = torch.empty((M, K), dtype=residual.dtype, device=y.device)
    stream = torch.cuda.current_stream().cuda_stream
    c = _radiance_comm()
    # Decode fast path: AR + the whole epilogue in ONE kernel (libr4d exact_nq, one block per
    # row -- rx4). The branch is shape/dtype-only, so both TP ranks take the same side every
    # call and the per-block seq counters stay in lockstep. Prefill (M > 512) falls through to
    # the plain AR (which routes big messages to wht6) + the standalone epilogue kernel.
    if (c is not None and y.dtype == torch.bfloat16 and y.is_contiguous()
            and M <= 512 and K % 8 == 0 and K <= 10240
            and y.numel() * 2 <= c.max_bytes):
        c._ext.ar_oneshot_2rank_exact_nq(
            c._peer_scratch, c._scratch, c._peer_flags, c._flags,
            c._seq.data_ptr(), c.slot16, y.data_ptr(), residual.data_ptr(),
            weight.data_ptr(), q.data_ptr(), sc.data_ptr(), ro.data_ptr(),
            M, K, eps, stream, c.drain, c.acq)
        return q, sc, ro
    from vllm.distributed import tensor_model_parallel_all_reduce as _ar
    y = _ar(y)
    _ext().launch_add_rms_quant(y.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                                q.data_ptr(), sc.data_ptr(), ro.data_ptr(), M, K, eps,
                                stream)
    return q, sc, ro


@ar_add_rms_quant.register_fake
def _(y, residual, weight, eps):
    M, K = y.shape
    return (torch.empty((M, K), dtype=torch.float8_e4m3fn, device=y.device),
            torch.empty((M,), dtype=torch.float32, device=y.device),
            torch.empty((M, K), dtype=residual.dtype, device=y.device))


@torch.library.custom_op("radiance::silu_mul_quant", mutates_args=())
def silu_mul_quant(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N2 = gu.shape
    N = N2 // 2
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=gu.device)
    sc = torch.empty((M,), dtype=torch.float32, device=gu.device)
    _ext().launch_silu_mul_quant(gu.contiguous().data_ptr(), q.data_ptr(), sc.data_ptr(),
                                 M, N, torch.cuda.current_stream().cuda_stream)
    return q, sc


@silu_mul_quant.register_fake
def _(gu):
    M, N2 = gu.shape
    return (torch.empty((M, N2 // 2), dtype=torch.float8_e4m3fn, device=gu.device),
            torch.empty((M,), dtype=torch.float32, device=gu.device))


class _SiluMulQuant(torch.nn.Module):
    """Drop-in for the MLP's SiluAndMul: returns (q_fp8, scale) for the pq-path down_proj.
    3 fp8 codes per ~1.1M differ from the traced chain (expf vs aten sigmoid, 1 ulp); scales
    exact."""

    def forward(self, gu):
        return torch.ops.radiance.silu_mul_quant(gu)


def _stream_forward(self, hidden_states, residual, positions=None, **kwargs):
    """Patched decoder-layer forward under the fp8-stream contract. Mirrors the stock body
    (qwen3_next.py Qwen3NextDecoderLayer.forward) minus the guarded-away branches (sequence
    parallel, layer_scale), which install() proves are dead for this serve."""
    if residual is not None and self._rad_fp8_in:
        # hidden_states is the PREVIOUS layer's un-reduced down_proj output.
        q, qs, residual = torch.ops.radiance.ar_add_rms_quant(
            hidden_states, residual, self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon)
        hs = (q, qs)
    else:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hs = hidden_states

    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn(hidden_states=hs)
    else:
        attn_out = self.self_attn(hidden_states=hs, positions=positions)

    if self._rad_fp8_mid:
        # attn_out is this layer's un-reduced o_proj/out_proj output.
        q2, qs2, residual = torch.ops.radiance.ar_add_rms_quant(
            attn_out, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon)
        hidden_states = self.mlp((q2, qs2))
    else:
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
    # If this layer's down_proj was un-reduced (self is not the last layer), hidden_states is
    # a partial sum; the NEXT layer's _rad_fp8_in epilogue finishes it. Same 2-tuple contract.
    return hidden_states, residual


def _linear_is_ours(lin) -> bool:
    return (lin is not None and getattr(lin, "radiance_wref", None) is not None)


def _consumers_ok(layer) -> bool:
    """Every linear that would receive a (q, scale) tuple must be on the radiance kernel."""
    if layer.layer_type == "linear_attention":
        la = layer.linear_attn
        if not getattr(la, "_rad_merged", False):        # gdnmerge marker, set below at install
            return False
    else:
        if not _linear_is_ours(layer.self_attn.qkv_proj):
            return False
    return _linear_is_ours(layer.mlp.gate_up_proj)


def install(model) -> None:
    if not ENABLED:
        return
    try:
        from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() <= 1:
            _log("tp=1, skipping (epilogue exists to absorb the AR)")
            return
        if get_pp_group().world_size > 1:
            _log("pp>1, skipping (inter-layer contract crosses pp boundary)")
            return
    except Exception as e:                              # noqa: BLE001
        _log(f"distributed introspection failed, skipping: {e!r}")
        return
    # The decoder core hides at different depths per wrapper (ForCausalLM: model.model;
    # ForConditionalGeneration: model.language_model.model). Find it structurally: the module
    # that owns BOTH the layer list and the aux-tap config is Qwen3NextModel.
    core = None
    for m in model.modules():
        if hasattr(m, "layers") and hasattr(m, "aux_hidden_state_layers"):
            core = m
            break
    if core is None:
        _log("no decoder core with .layers/.aux_hidden_state_layers found, skipping")
        return
    layers = core.layers
    if tuple(getattr(core, "aux_hidden_state_layers", ()) or ()):
        _log("aux hidden state taps are set, skipping")
        return
    import types
    n_in = n_mid = n_down = 0
    L = len(layers)
    for i, layer in enumerate(layers):
        if getattr(layer, "layer_scale", False) or layer.use_attn_reduce_scatter_for_moe:
            _log(f"layer {i}: layer_scale/SP branch live, left stock")
            layer._rad_fp8_in = layer._rad_fp8_mid = False
            layer.forward = types.MethodType(_stream_forward, layer)
            continue
        if getattr(layer.mlp, "expert_gate", None) is not None:
            _log(f"layer {i}: mlp.expert_gate present, left stock")
            layer._rad_fp8_in = layer._rad_fp8_mid = False
            layer.forward = types.MethodType(_stream_forward, layer)
            continue
        row = (layer.linear_attn.out_proj if layer.layer_type == "linear_attention"
               else layer.self_attn.o_proj)
        cons_ok = _consumers_ok(layer)
        # mid epilogue: this layer's attention-side AR + post norm + quant into its own mlp
        mid_ok = cons_ok and _linear_is_ours(row) and row.bias is None and row.reduce_results
        if mid_ok:
            row.reduce_results = False
            layer._rad_fp8_mid = True
            n_mid += 1
        else:
            layer._rad_fp8_mid = False
        # down stream: this layer's down_proj stays partial, the NEXT layer's input epilogue
        # finishes it. Never on the last layer (model.norm and the drafter read its output).
        nxt = layers[i + 1] if i + 1 < L else None
        down = layer.mlp.down_proj
        down_ok = (nxt is not None and _linear_is_ours(down) and down.bias is None
                   and down.reduce_results and _consumers_ok(nxt)
                   and not getattr(nxt, "layer_scale", False)
                   and not nxt.use_attn_reduce_scatter_for_moe)
        if down_ok:
            down.reduce_results = False
            nxt._rad_fp8_in = True
            n_down += 1
            n_in += 1
        # activation epilogue: silu*up + quant in one kernel, independent of the AR contract --
        # any layer whose down_proj is ours can take it (the last layer included; only the
        # INPUT side of down_proj changes).
        if _linear_is_ours(layer.mlp.down_proj) and (layer.mlp.down_proj.input_size_per_partition % 8 == 0):
            layer.mlp.act_fn = _SiluMulQuant()
            n_act = getattr(install, "_n_act", 0) + 1
            install._n_act = n_act
        layer.forward = types.MethodType(_stream_forward, layer)
        if not hasattr(layer, "_rad_fp8_in"):
            layer._rad_fp8_in = False
    # make sure every layer got flags (layer 0 has no upstream partial)
    for layer in layers:
        if not hasattr(layer, "_rad_fp8_in"):
            layer._rad_fp8_in = False
        if not hasattr(layer, "_rad_fp8_mid"):
            layer._rad_fp8_mid = False
    _log(f"fp8 stream installed: {n_mid} mid epilogues, {n_down} down streams, "
         f"{getattr(install, '_n_act', 0)} act epilogues over {L} layers")
