"""One-launch candidate for the declared stock M1 GDN arithmetic contract.

Importing this module does not initialize a GPU. This remains an experimental
operator until native comparisons and full-model qualification succeed.
"""

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def retained_rows(rows, keep):
    keep = tuple(keep)
    if any(type(i) is not int or not 0 <= i < rows for i in keep):
        raise DiagnosticError("retained GDN row is outside the sequence")
    if len(set(keep)) != len(keep):
        raise DiagnosticError("duplicate retained GDN row")
    return keep


class StockScan:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = torch.device(device)

    def run(self, initial, mixed_qkv, a, b, a_log, dt_bias, *, retain_rows=()):
        from stock_gdn_sequence import SequenceResult

        t = self.torch
        if not isinstance(mixed_qkv, t.Tensor) or mixed_qkv.ndim != 2:
            raise DiagnosticError("stock scan requires packed QKV rows")
        rows = mixed_qkv.shape[0]
        keep = retained_rows(rows, retain_rows)
        for x, shape, dtype in (
            (initial, (48, 128, 128), t.float32),
            (mixed_qkv, (rows, 10240), t.bfloat16),
            (a, (rows, 48), t.bfloat16),
            (b, (rows, 48), t.bfloat16),
            (a_log, (48,), t.float32),
            (dt_bias, (48,), t.bfloat16),
        ):
            if (
                not isinstance(x, t.Tensor)
                or tuple(x.shape) != shape
                or x.dtype != dtype
                or x.device != self.device
                or x.stride(-1) != 1
            ):
                raise DiagnosticError("unsupported stock scan tensor representation")
        if not initial.is_contiguous():
            raise DiagnosticError("stock scan requires contiguous logical initial state")
        outputs = t.empty((rows, 48, 128), dtype=t.bfloat16, device=self.device)
        if rows == 0:
            return SequenceResult(outputs, initial.clone(), {}, 0)
        if self.device.type != "cuda":
            raise DiagnosticError("stock scan native execution requires a GPU")
        from stock_gdn_scan_kernel import stock_gdn_scan_kernel

        # Prefill writes only its final state. D7 retains each after-row state;
        # the caller chooses the accepted prefix, never the final proposed row.
        states = t.empty((rows if keep else 1, 48, 128, 128), dtype=t.float32, device=self.device)
        stock_gdn_scan_kernel[(4, 48)](
            mixed_qkv,
            a,
            b,
            a_log,
            dt_bias,
            initial,
            outputs,
            states,
            None,
            None,
            None,
            rows,
            128**-0.5,
            mixed_qkv.stride(0),
            a.stride(0),
            b.stride(0),
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=32,
            SAVE_ROWS=bool(keep),
            INDEXED=False,
            stride_state=0,
            stride_index_seq=0,
            stride_index_row=0,
            num_warps=1,
            num_stages=3,
            enable_fp_fusion=True,
            allow_flush_denorm=False,
        )
        return SequenceResult(outputs, states[-1], {i: states[i] for i in keep}, rows)

    def d7(self, initial, mixed_qkv, a, b, a_log, dt_bias):
        if tuple(mixed_qkv.shape) != (8, 10240):
            raise DiagnosticError("D7 requires the pending input plus seven proposals")
        return self.run(initial, mixed_qkv, a, b, a_log, dt_bias, retain_rows=range(8))
