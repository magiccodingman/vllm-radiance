"""Stock-arithmetic speculative GDN with device-side acceptance and state writes.

There are no synchronizing tensor-to-host reads, per-row launches or state
copies. Metadata validation remains the dispatcher's obligation, just as for
the original R4D entry point; qualification exercises that obligation separately.
"""

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class StockIndexedAdapter:
    def __init__(self, torch):
        self.torch = torch
        self.calls = 0
        self.rows = 0

    def __call__(
        self,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        output,
        cu,
        indices,
        accepted,
        num_seqs,
        heads,
        query_heads,
        scale,
        z_gate=None,
        norm=None,
    ):
        t = self.torch
        count = q.shape[0]
        if (
            heads != 48
            or query_heads != 16
            or scale != 128**-0.5
            or z_gate is not None
            or norm is not None
            or num_seqs < 1
            or indices.ndim != 2
            or indices.shape[0] != num_seqs
            or not 1 <= indices.shape[1] <= 8
            or cu.shape != (num_seqs + 1,)
            or accepted is None
            or accepted.shape != (num_seqs,)
            or state.ndim != 4
            or tuple(state.shape[1:]) != (48, 128, 128)
            or state.dtype != t.float32
            or state.stride()[1:] != (16384, 128, 1)
            or a_log.shape != (48,)
            or a_log.dtype != t.float32
            or dt_bias.shape != (48,)
            or dt_bias.dtype != t.float32
            or any(x.dtype != t.int32 for x in (cu, indices, accepted))
        ):
            raise DiagnosticError("unsupported indexed stock GDN invocation")
        for x, shape in (
            (q, (count, 16, 128)),
            (k, (count, 16, 128)),
            (v, (count, 48, 128)),
            (output, (count, 48, 128)),
        ):
            if tuple(x.shape) != shape or x.dtype != t.bfloat16 or not x.is_contiguous():
                raise DiagnosticError("unsupported indexed stock GDN QKV/output layout")
        if any(
            x.shape[0] < count or x.shape[1:] != (48,) or x.dtype != t.bfloat16 or x.stride(-1) != 1
            for x in (a, b)
        ):
            raise DiagnosticError("unsupported indexed stock GDN gate layout")
        if any(
            x.device != q.device
            for x in (k, v, a, b, a_log, dt_bias, state, output, cu, indices, accepted)
        ):
            raise DiagnosticError("indexed stock GDN tensors must share a device")
        if q.device.type != "cuda":
            raise DiagnosticError("indexed stock GDN requires a GPU")
        from stock_gdn_scan_kernel import stock_gdn_scan_kernel

        packed = t.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
        # The native ABI widens a read-only BF16 parameter. Round back to the
        # declared input type before entering the same stock transition.
        bias = dt_bias.to(t.bfloat16)
        stock_gdn_scan_kernel[(4, 48, num_seqs)](
            packed,
            a,
            b,
            a_log,
            bias,
            state,
            output,
            state,
            indices,
            accepted,
            cu,
            count,
            scale,
            packed.stride(0),
            a.stride(0),
            b.stride(0),
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=32,
            SAVE_ROWS=True,
            INDEXED=True,
            stride_state=state.stride(0),
            stride_index_seq=indices.stride(0),
            stride_index_row=indices.stride(1),
            num_warps=1,
            num_stages=3,
            enable_fp_fusion=True,
            allow_flush_denorm=False,
        )
        self.calls += 1
        self.rows += count
