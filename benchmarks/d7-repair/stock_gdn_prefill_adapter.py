"""Experimental causal prefill using the same stock transition as decode.

Admits one prefill sequence. This bypasses cumulative gate subtraction and
future-dependent whole-chunk fallbacks. Unsupported mixed batches fail
explicitly in qualification instead of silently reverting to different math.
"""

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class StockPrefillAdapter:
    def __init__(self, torch, native, convolution, scan, original):
        self.torch = torch
        self.native = native
        self.convolution = convolution
        self.scan = scan
        self.original = original
        self.calls = 0
        self.rows = 0

    def __call__(self, layer, mixed_qkv, b, a, core_attn_out):
        plan = self.native._plan(layer, mixed_qkv, b, a, core_attn_out)
        if plan is None or plan[0] == "decode":
            return self.original(layer, mixed_qkv, b, a, core_attn_out)
        kind, count, conv_state, ssm_state, md, (_, cu) = plan
        if kind != "prefill" or md.num_prefills != 1:
            raise DiagnosticError("stock prefill qualification requires one unmixed sequence")
        t = self.torch
        if (
            mixed_qkv.shape[1:] != (10240,)
            or mixed_qkv.dtype != t.bfloat16
            or layer.A_log.dtype != t.float32
            or layer.dt_bias.dtype != t.bfloat16
            or layer.conv1d.bias is not None
            or tuple(md.prefill_state_indices.shape) != (1,)
            or tuple(md.prefill_has_initial_state.shape) != (1,)
            or md.has_initial_state is None
            or tuple(md.has_initial_state.shape) != (1,)
        ):
            raise DiagnosticError("unsupported stock prefill metadata or arithmetic")
        conv_ids = md.non_spec_state_indices_tensor
        if conv_ids.ndim == 2:
            conv_ids = conv_ids[:, 0]
        if conv_ids.shape != (1,):
            raise DiagnosticError("stock prefill requires one convolution state index")
        # Use the original convolution/state address maps, retaining unrelated
        # slots and selecting zero history only for a fresh sequence.
        old_conv = conv_state.index_select(0, conv_ids.long())
        conv_state.index_copy_(
            0,
            conv_ids.long(),
            t.where(md.has_initial_state[:, None, None], old_conv, t.zeros_like(old_conv)),
        )
        selected = ssm_state.index_select(0, md.prefill_state_indices.long())
        initial = t.where(
            md.prefill_has_initial_state[:, None, None, None], selected, t.zeros_like(selected)
        )[0].contiguous()
        source = mixed_qkv[:count]
        packed = t.empty_like(source)
        weights = layer.conv1d.weight.view(10240, 4)
        self.convolution(
            source,
            conv_state,
            weights,
            None,
            "silu",
            conv_state_indices=conv_ids,
            query_start_loc=cu,
            max_query_len=count,
            validate_data=False,
            out=packed,
        )
        result = self.scan.run(initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias)
        core_attn_out[:count].view(count, 48, 128).copy_(result.outputs)
        ssm_state.index_copy_(0, md.prefill_state_indices.long(), result.final_state.unsqueeze(0))
        layer.__dict__.pop("_radiance_z", None)
        self.calls += 1
        self.rows += count
        return True
