"""Diagnostic Radiance recurrent-call adapter for the pinned stock GDN fold.

Admits one unfused sequence of one through eight rows. It computes privately
before copying verified results into the diagnostic worker's state pool. It is
not installed in production and does not claim graph or concurrent support.
"""

from __future__ import annotations

import math

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class StockRecurrentAdapter:
    def __init__(self, torch, sequence):
        self.torch = torch
        self.sequence = sequence
        self.calls = 0
        self.rows = 0

    def __call__(
        self,
        q,
        k,
        v,
        a,
        b,
        A_log,  # noqa: N803 - preserve the intercepted Radiance call ABI
        dt_bias,
        ssm_state,
        o,
        cu,
        sidx,
        num_accepted,
        num_seqs,
        H,  # noqa: N803 - preserve the intercepted Radiance call ABI
        Hg,  # noqa: N803 - preserve the intercepted Radiance call ABI
        scale,
        z_gate=None,
        norm=None,
    ):
        t = self.torch
        if num_seqs != 1 or H != 48 or Hg != 16 or z_gate is not None or norm is not None:
            raise DiagnosticError(
                "stock recurrent diagnostic requires one unfused 48/16-head sequence"
            )
        if not math.isfinite(scale) or scale != 128**-0.5:
            raise DiagnosticError("stock recurrent diagnostic requires the pinned query scale")
        tensors = (q, k, v, a, b, A_log, dt_bias, ssm_state, o, cu, sidx)
        if not all(isinstance(x, t.Tensor) and x.device == self.sequence.device for x in tensors):
            raise DiagnosticError("stock recurrent diagnostic tensors must share the oracle device")
        if q.ndim != 3 or not 1 <= q.shape[0] <= 8:
            raise DiagnosticError("stock recurrent diagnostic admits one through eight rows")
        count = q.shape[0]
        for x, shape in (
            (q, (count, 16, 128)),
            (k, (count, 16, 128)),
            (v, (count, 48, 128)),
            (o, (count, 48, 128)),
        ):
            if tuple(x.shape) != shape or x.dtype != t.bfloat16:
                raise DiagnosticError(
                    "stock recurrent diagnostic QKV/output representation changed"
                )
        if (
            any(
                x.ndim != 2 or x.shape[0] < count or x.shape[1] != 48 or x.dtype != t.bfloat16
                for x in (a, b)
            )
            or A_log.shape != (48,)
            or A_log.dtype != t.float32
            or dt_bias.shape != (48,)
            or dt_bias.dtype != t.float32
            or ssm_state.ndim != 4
            or tuple(ssm_state.shape[1:]) != (48, 128, 128)
            or ssm_state.dtype != t.float32
        ):
            raise DiagnosticError("stock recurrent diagnostic gate/state representation changed")
        if cu.shape != (2,) or cu.dtype != t.int32 or cu.tolist() != [0, count]:
            raise DiagnosticError("stock recurrent diagnostic sequence boundaries changed")
        if sidx.ndim != 2 or sidx.shape[0] != 1 or sidx.shape[1] < count or sidx.dtype != t.int32:
            raise DiagnosticError("stock recurrent diagnostic state map changed")
        if num_accepted is None:
            accepted = 1
        elif (
            isinstance(num_accepted, t.Tensor)
            and num_accepted.device == self.sequence.device
            and num_accepted.dtype == t.int32
            and num_accepted.shape == (1,)
        ):
            accepted = int(num_accepted.item())
        else:
            raise DiagnosticError("stock recurrent diagnostic acceptance metadata changed")
        if not 1 <= accepted <= sidx.shape[1]:
            raise DiagnosticError(
                "stock recurrent diagnostic previous acceptance outside state map"
            )
        initial_slot = int(sidx[0, accepted - 1].item())
        destinations = [int(x) for x in sidx[0, :count].tolist()]
        if (
            not all(0 < x < ssm_state.shape[0] for x in [initial_slot, *destinations])
            or len(set(destinations)) != count
        ):
            raise DiagnosticError("stock recurrent diagnostic invalid or aliased state slots")
        rounded_bias = dt_bias.to(t.bfloat16)
        if not t.equal(rounded_bias.float().view(t.uint8), dt_bias.contiguous().view(t.uint8)):
            raise DiagnosticError("FP32 bias is not an exact widening of the stock BF16 bias")
        packed = t.cat((q.reshape(count, -1), k.reshape(count, -1), v.reshape(count, -1)), dim=1)
        # Clone the old committed state before touching any speculative slot.
        initial = ssm_state[initial_slot].clone()
        result = self.sequence.run(
            initial,
            packed,
            a[:count],
            b[:count],
            A_log,
            rounded_bias,
            retain_rows=range(count),
        )
        if result.rows != count or set(result.after_rows) != set(range(count)):
            raise DiagnosticError("stock recurrent diagnostic returned an incomplete sequence")
        # Validate every tentative result before publishing any part of it.
        values = [result.outputs, *result.after_rows.values()]
        if (
            tuple(result.outputs.shape) != (count, 48, 128)
            or result.outputs.dtype != t.bfloat16
            or any(
                tuple(x.shape) != (48, 128, 128) or x.dtype != t.float32
                for x in result.after_rows.values()
            )
            or any(
                x.device != self.sequence.device or not bool(t.isfinite(x).all()) for x in values
            )
        ):
            raise DiagnosticError("stock recurrent diagnostic returned invalid tentative values")
        for row, slot in enumerate(destinations):
            ssm_state[slot].copy_(result.after_rows[row])
        o.copy_(result.outputs)
        self.calls += 1
        self.rows += count
