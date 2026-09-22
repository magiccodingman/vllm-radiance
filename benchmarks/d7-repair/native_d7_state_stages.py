"""Isolate native convolution, recurrence and attention through their real APIs."""

import ast
import copy
import functools
import importlib
import inspect
from pathlib import Path

import torch

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def packed(q, k, v):
    all_rows = torch.cat([t.reshape(t.shape[0], -1) for t in (q, k, v)], dim=1)
    parts = all_rows.split((2048, 2048, 6144), dim=1)
    return tuple(t.view(-1, h, 128) for t, h in zip(parts, (16, 16, 48), strict=True))


class StateStages:
    def __init__(self, runner, repairs):
        import radiance_gdn as gdn
        import radiance_r4d_attn as attention

        self.runner = runner
        self.repairs = repairs
        self.gdn = gdn
        # Recover only the two unchanged, source-bound public native wrappers;
        # do not re-import the module and reinstall its process-wide patches.
        tree = ast.parse(Path(gdn.__file__).read_text())
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("conv_update", "recurrent_update")
        ]
        if len(functions) != 2:
            raise DiagnosticError("original GDN ABI is missing")
        namespace = dict(vars(gdn))
        exec(compile(ast.Module(body=functions, type_ignores=[]), gdn.__file__, "exec"), namespace)
        self.old_conv = namespace["conv_update"]
        self.old_recur = namespace["recurrent_update"]
        original = [
            old for owner, key, old, *_ in repairs.hooks.entries if key == "causal_conv1d_update"
        ]
        if len(original) != 1:
            raise DiagnosticError("old M1 convolution binding is ambiguous")
        self.old_m1_conv = original[0]
        self.fixed_m1_conv = repairs.convolution.update
        self.old_attention = inspect.unwrap(attention.R4DAttentionImpl.forward)
        self.stock = importlib.import_module(
            "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
        )

    def conv_serial(self, update, *args):
        (
            x,
            w,
            bias,
            history,
            state_len,
            indices,
            accepted,
            _cu,
            nseq,
            count,
            heads,
            qheads,
            maxlen,
        ) = args
        if (nseq, count, heads, qheads, maxlen, state_len) != (1, 8, 48, 16, 8, 10):
            raise DiagnosticError("serial convolution cut is outside the pinned ABI")
        slot = int(indices[0])
        offset = int(accepted[0]) - 1
        initial = history[slot, :, offset : offset + 3].clone()
        pool = torch.empty((2, 10240, 3), device=x.device, dtype=x.dtype)
        pool[1].copy_(initial)
        ones = torch.ones(1, device=x.device, dtype=torch.int32)
        boundaries = torch.tensor([0, 1], device=x.device, dtype=torch.int32)
        out = torch.empty_like(x)
        for row in range(8):
            actual = update(
                x[row : row + 1],
                pool,
                w,
                bias,
                "silu",
                conv_state_indices=ones,
                query_start_loc=boundaries,
                max_query_len=1,
                validate_data=False,
                out=out[row : row + 1],
            )
            if actual.data_ptr() != out[row : row + 1].data_ptr():
                raise DiagnosticError("serial convolution replaced its output")
        history[slot].copy_(torch.cat((initial[:, 1:], x.T), dim=1))
        if not torch.equal(pool[1], history[slot, :, -3:]):
            raise DiagnosticError("serial convolution logical history did not match its remapping")
        return tuple(
            t.view(8, h, 128)
            for t, h in zip(out.split((2048, 2048, 6144), 1), (16, 16, 48), strict=True)
        )

    def conv_old(self, *args, **kwargs):
        return packed(*self.old_conv(*args, **kwargs))

    def recurrent_old(self, q, k, v, *args, **kwargs):
        return self.old_recur(q.contiguous(), k.contiguous(), v.contiguous(), *args, **kwargs)

    def recurrent_m1(
        self,
        q,
        k,
        v,
        a,
        b,
        alog,
        bias,
        state,
        out,
        cu,
        indices,
        accepted,
        nseq,
        heads,
        qheads,
        scale,
        z_gate=None,
        norm=None,
    ):
        if (nseq, heads, qheads) != (1, 48, 16) or z_gate is not None or norm is not None:
            raise DiagnosticError("serial recurrence cut changed")
        initial = int(indices[0, int(accepted[0]) - 1])
        slots = [int(s) for s in indices[0, :8]]
        # Use the same native packed M1 kernel and row strides as ordinary
        # target decode; a private state slot prevents an early destination
        # overwrite from altering the starting committed state.
        storage = torch.empty(2 * 802816, dtype=torch.float32, device=q.device)
        pool = storage.as_strided((2, 48, 128, 128), (802816, 16384, 128, 1))
        pool[1].copy_(state[initial])
        sid = torch.ones(1, dtype=torch.int32, device=q.device)
        mixed = torch.empty((1, 16384), dtype=q.dtype, device=q.device)
        aa = torch.empty((1, 96), dtype=a.dtype, device=q.device)
        bb = torch.empty_like(aa)
        dt = bias.bfloat16()
        kernel = self.stock.fused_recurrent_gated_delta_rule_packed_decode_kernel
        for row in range(8):
            mixed[:, :10240].copy_(
                torch.cat((q[row].flatten(), k[row].flatten(), v[row].flatten())).view(1, -1)
            )
            aa[:, :48].copy_(a[row])
            bb[:, :48].copy_(b[row])
            kernel[(4, 48, 1)](
                mixed_qkv=mixed,
                a=aa,
                b=bb,
                A_log=alog,
                dt_bias=dt,
                o=out[row : row + 1].view(1, 1, 48, 128),
                h0=pool,
                ht=pool,
                ssm_state_indices=sid,
                scale=scale,
                stride_mixed_qkv_tok=16384,
                stride_a_tok=96,
                stride_b_tok=96,
                stride_init_state_token=802816,
                stride_final_state_token=802816,
                stride_indices_seq=1,
                H=16,
                HV=48,
                K=128,
                V=128,
                BK=128,
                BV=32,
                SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=True,
                SPLIT_BATCH_HEAD_GRID=False,
                num_warps=1,
                num_stages=3,
                enable_fp_fusion=True,
                allow_flush_denorm=False,
            )
            state[slots[row]].copy_(pool[1])

    def core_variant(self, function, name, replacement, *args, **kwargs):
        original = getattr(self.gdn, name)
        setattr(self.gdn, name, replacement)
        try:
            return function(*args, **kwargs)
        finally:
            setattr(self.gdn, name, original)

    def attention_variant(self, function, serial, *args, **kwargs):
        from vllm.forward_context import get_forward_context

        schema = function._schema
        values = {
            arg.name: args[i] if i < len(args) else kwargs.get(arg.name)
            for i, arg in enumerate(schema.arguments)
        }
        name = values["layer_name"]
        if not isinstance(name, str):
            from vllm.model_executor.layers.attention.attention import _resolve_layer_name

            name = _resolve_layer_name(name)
        module = get_forward_context().no_compile_layers[name]
        impl = module.impl
        current = impl.forward
        owned = "forward" in vars(impl)

        def apply(
            layer,
            query,
            key,
            value,
            kv_cache,
            md,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            if not serial:
                return self.old_attention(
                    impl,
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    md,
                    output,
                    output_scale,
                    output_block_scale,
                )
            for row in range(8):
                one = copy.copy(md)
                one.r4d_plan = ((0, 1, 1, 0),)
                one.seq_lens = md.seq_lens[:1] - 7 + row
                one.r4d_max_ctx = md.r4d_max_ctx - 7 + row
                one.num_actual_tokens = 1
                self.old_attention(
                    impl,
                    layer,
                    query[row : row + 1].contiguous(),
                    key,
                    value,
                    kv_cache,
                    one,
                    output[row : row + 1],
                    output_scale,
                    output_block_scale,
                )
            return output

        impl.forward = apply
        try:
            return function(*args, **kwargs)
        finally:
            if owned:
                impl.forward = current
            else:
                del impl.forward

    def cases(self, call):
        from native_d7_stage_matrix import shared_versions

        fn = call.function
        args, _kwargs = call.cut[0][0].thaw(call.cut[0])
        if call.name == "vllm.qwen_gdn_attention_core.default":
            name = args[4]
            if not isinstance(name, str):
                from vllm.model_executor.layers.attention.attention import _resolve_layer_name

                name = _resolve_layer_name(name)
            original = functools.partial(self.core_variant, fn, "conv_update", self.conv_old)
            old_m1 = functools.partial(
                self.core_variant,
                fn,
                "conv_update",
                functools.partial(self.conv_serial, self.old_m1_conv),
            )
            fixed_m1 = functools.partial(
                self.core_variant,
                fn,
                "conv_update",
                functools.partial(self.conv_serial, self.fixed_m1_conv),
            )
            versions = shared_versions(fn, fixed_m1)
            versions.update(old_m1=old_m1, old_m8=original)
            yield "GDN convolution", name, versions
            original = functools.partial(
                self.core_variant, fn, "recurrent_update", self.recurrent_old
            )
            m1 = functools.partial(self.core_variant, fn, "recurrent_update", self.recurrent_m1)
            versions = shared_versions(fn, m1)
            versions["old_m8"] = original
            yield "GDN recurrence and gates", name, versions
        elif call.name == "vllm.unified_attention_with_output.default":
            name = args[4]
            if not isinstance(name, str):
                from vllm.model_executor.layers.attention.attention import _resolve_layer_name

                name = _resolve_layer_name(name)
            old = functools.partial(self.attention_variant, fn, False)
            m1 = functools.partial(self.attention_variant, fn, True)
            versions = shared_versions(fn, m1)
            versions["old_m8"] = old
            yield "Attention decode and split-KV merge", name, versions
