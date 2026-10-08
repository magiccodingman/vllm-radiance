"""Experimental ordered GDN replay through the pinned stock M1 executable.

This is an operator repair candidate, not an installed backend hook. It uses
private state, retains D7 after-row states and never writes a live session.
The serial launches intentionally prioritize arithmetic identity over speed.
Importing this file does not import Torch/Triton or access a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import threading
from pathlib import Path

from qwen_r9700_lab.conformance_gdn_contract import (
    CONTRACT_ID,
    CONTRACT_SHA256,
    audit_artifacts,
    committed_gdn_row,
    read_contract,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, integer

MODULE = "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
KERNEL = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
STATE_SHAPE = (48, 128, 128)
STATE_ELEMENTS = 48 * 128 * 128
STATE_STRIDE = 802816
GRID = (4, 48, 1)


def _contract(contract):
    authenticate(contract)
    if contract.get("id") != CONTRACT_ID or contract["sha256"] != CONTRACT_SHA256:
        raise DiagnosticError("unreviewed stock GDN sequence contract")


def check_executable(contract, binary: bytes, metadata: dict):
    """Check the executable returned by warmup before allowing any launch.

    This binds an invocation to the chosen arithmetic artifact. It is not a
    functional proof of the artifact or an attestation of physical execution.
    """
    _contract(contract)
    authority = contract["arithmetic_authority"]
    expected = authority["files"][KERNEL + ".hsaco"]
    if not isinstance(binary, bytes) or hashlib.sha256(binary).hexdigest() != expected:
        raise DiagnosticError("stock GDN executable differs from the pinned oracle")
    if metadata.get("target") != authority["target"] or any(
        metadata.get(key) != value for key, value in authority["compiler"].items()
    ):
        raise DiagnosticError("stock GDN runtime compiler metadata changed")
    return expected


@dataclasses.dataclass(frozen=True)
class SequenceResult:
    outputs: object
    final_state: object
    after_rows: dict[int, object]
    rows: int

    def accepted_state(self, accepted_proposals: int):
        if self.rows != 8 or set(self.after_rows) != set(range(8)):
            raise DiagnosticError("accepted-prefix selection requires a complete D7 capture")
        # Row zero consumes the old pending token. The new bonus/correction
        # remains pending; there is deliberately no extra transition here.
        return self.after_rows[committed_gdn_row(accepted_proposals)].clone()


class _Sequence:
    """Shared implementation; the public native factory supplies a bound launcher.

    Tests use CPU Torch tensors and a deliberately non-GDN transition to check
    isolation, ordering and fault handling without claiming native arithmetic.
    """

    def __init__(self, torch, contract, device, launch_factory):
        _contract(contract)
        self.torch = torch
        self.contract = contract
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self.storage = torch.full((3 * STATE_STRIDE,), 17.0, dtype=torch.float32, device=device)
        self.pool = self.storage.as_strided((3, *STATE_SHAPE), (STATE_STRIDE, 128 * 128, 128, 1))
        # The single-row arithmetic does not depend on the row stride, but the
        # executable identity does. Preserve the captured QKVZ-backed packing.
        self.qkv_storage = torch.full((1, 16384), 17.0, dtype=torch.bfloat16, device=device)
        self.qkv = self.qkv_storage[:, :10240]
        self.a_storage = torch.full((1, 96), 17.0, dtype=torch.bfloat16, device=device)
        self.b_storage = self.a_storage.clone()
        self.a, self.b = self.a_storage[:, :48], self.b_storage[:, :48]
        self.A_log = torch.empty(48, dtype=torch.float32, device=device)
        self.dt_bias = torch.empty(48, dtype=torch.bfloat16, device=device)
        self.output = torch.empty((1, 1, 48, 128), dtype=torch.bfloat16, device=device)
        self.indices = torch.ones(1, dtype=torch.int32, device=device)
        self._launch = launch_factory(self)

    def _tensor(self, value, shape, dtype, name):
        if (
            not isinstance(value, self.torch.Tensor)
            or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype
            or value.device != self.device
            or not bool(self.torch.isfinite(value).all())
        ):
            raise DiagnosticError(f"unsupported stock GDN replay tensor: {name}")

    def _guards(self):
        if (
            not bool((self.storage[:STATE_STRIDE] == 17).all())
            or not bool((self.storage[STATE_STRIDE + STATE_ELEMENTS :] == 17).all())
            or not bool((self.a_storage[:, 48:] == 17).all())
            or not bool((self.b_storage[:, 48:] == 17).all())
            or not bool((self.qkv_storage[:, 10240:] == 17).all())
            or not bool((self.indices == 1).all())
        ):
            raise DiagnosticError("stock GDN replay modified a guard or state index")

    def run(self, initial, mixed_qkv, a, b, a_log, dt_bias, *, retain_rows=()):
        """Return tentative results from one sequence; empty input is identity."""
        if not self._lock.acquire(blocking=False):
            raise DiagnosticError("stock GDN scratch workspace is already in use")
        try:
            return self._run(initial, mixed_qkv, a, b, a_log, dt_bias, retain_rows)
        finally:
            self._lock.release()

    def _run(self, initial, mixed_qkv, a, b, a_log, dt_bias, retain_rows):
        t = self.torch
        if not isinstance(mixed_qkv, t.Tensor) or mixed_qkv.ndim != 2:
            raise DiagnosticError("stock GDN replay requires a matrix of packed input rows")
        rows = mixed_qkv.shape[0]
        keep = tuple(retain_rows)
        for row in keep:
            integer(row)
            if row >= rows:
                raise DiagnosticError("retained GDN state row is outside the sequence")
        if len(set(keep)) != len(keep):
            raise DiagnosticError("duplicate retained GDN state row")
        for value, shape, dtype, name in (
            (initial, STATE_SHAPE, t.float32, "initial state"),
            (mixed_qkv, (rows, 10240), t.bfloat16, "mixed QKV"),
            (a, (rows, 48), t.bfloat16, "a"),
            (b, (rows, 48), t.bfloat16, "b"),
            (a_log, (48,), t.float32, "A_log"),
            (dt_bias, (48,), t.bfloat16, "dt_bias"),
        ):
            self._tensor(value, shape, dtype, name)
        self.storage.fill_(17)
        self.qkv_storage.fill_(17)
        self.a_storage.fill_(17)
        self.b_storage.fill_(17)
        self.indices.fill_(1)
        self.pool[1].copy_(initial)
        self.A_log.copy_(a_log)
        self.dt_bias.copy_(dt_bias)
        outputs = t.empty((rows, 48, 128), dtype=t.bfloat16, device=self.device)
        retained = {}
        for row in range(rows):
            self.qkv.copy_(mixed_qkv[row : row + 1])
            self.a.copy_(a[row : row + 1])
            self.b.copy_(b[row : row + 1])
            self.output.fill_(float("nan"))
            self._launch()
            # These synchronizing checks are intentional in this diagnostic
            # implementation. No unchecked result reaches the caller.
            self._guards()
            if not bool(t.isfinite(self.output).all()) or not bool(t.isfinite(self.pool[1]).all()):
                raise DiagnosticError("stock GDN replay produced incomplete or non-finite results")
            for copied, original in (
                (self.qkv, mixed_qkv[row : row + 1]),
                (self.a, a[row : row + 1]),
                (self.b, b[row : row + 1]),
                (self.A_log, a_log),
                (self.dt_bias, dt_bias),
            ):
                if not t.equal(copied.view(t.uint8), original.contiguous().view(t.uint8)):
                    raise DiagnosticError("stock GDN replay modified an input")
            outputs[row].copy_(self.output[0, 0])
            if row in keep:
                retained[row] = self.pool[1].clone()
        self._guards()
        return SequenceResult(outputs, self.pool[1].clone(), retained, rows)

    def d7(self, initial, mixed_qkv, a, b, a_log, dt_bias):
        if not isinstance(mixed_qkv, self.torch.Tensor) or mixed_qkv.shape != (8, 10240):
            raise DiagnosticError("D7 requires the pending input plus seven proposals")
        return self.run(initial, mixed_qkv, a, b, a_log, dt_bias, retain_rows=range(8))


def native_sequence(contract_path: Path, compiled_root: Path, stock_source: Path, op_source: Path):
    """Create a GPU runner only under the caller's already acquired GPU lease.

    The factory compiles but does not launch the GDN kernel until its binary has
    matched the existing oracle. There is no approximate or alternate fallback.
    """
    contract = read_contract(contract_path)
    audit_artifacts(
        contract, compiled_root, {"fused_recurrent.py": stock_source, "fla_op.py": op_source}
    )
    import torch
    import triton
    from triton.runtime import driver

    stock = importlib.import_module(MODULE)
    op = importlib.import_module(MODULE.rsplit(".", 1)[0] + ".op")
    authority = contract["arithmetic_authority"]
    for module, name in ((stock, "fused_recurrent.py"), (op, "fla_op.py")):
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != authority["source"][name]:
            raise DiagnosticError("loaded stock GDN sources differ from the oracle")
    if stock.exp is not op.exp or triton.__version__ != authority["compiler"]["triton_version"]:
        raise DiagnosticError("loaded stock GDN math or compiler differs from the oracle")
    target = driver.active.get_current_target()
    if dataclasses.asdict(target) != authority["target"]:
        raise DiagnosticError("stock GDN replay requires the pinned gfx1201 target")

    def factory(workspace):
        arguments = {
            "mixed_qkv": workspace.qkv,
            "a": workspace.a,
            "b": workspace.b,
            "A_log": workspace.A_log,
            "dt_bias": workspace.dt_bias,
            "o": workspace.output,
            "h0": workspace.pool,
            "ht": workspace.pool,
            "ssm_state_indices": workspace.indices,
            "scale": 128**-0.5,
            "stride_mixed_qkv_tok": 16384,
            "stride_a_tok": 96,
            "stride_b_tok": 96,
            "stride_init_state_token": STATE_STRIDE,
            "stride_final_state_token": STATE_STRIDE,
            "stride_indices_seq": 1,
            "H": 16,
            "HV": 48,
            "K": 128,
            "V": 128,
            "BK": 128,
            "BV": 32,
            "SOFTPLUS_THRESHOLD": 20.0,
            "USE_QK_L2NORM_IN_KERNEL": True,
            "SPLIT_BATCH_HEAD_GRID": False,
        }
        kernel = getattr(stock, KERNEL)
        if set(kernel.arg_names) != set(arguments):
            raise DiagnosticError("unreviewed stock GDN native call ABI")
        options = {k: v for k, v in authority["compiler"].items() if k != "triton_version"}
        compiled = kernel.warmup(**arguments, grid=GRID, **options)
        metadata = compiled.metadata._asdict()
        if dataclasses.is_dataclass(metadata.get("target")):
            metadata["target"] = dataclasses.asdict(metadata["target"])
        # Triton places the version in compiler metadata in this pinned build.
        check_executable(contract, compiled.kernel, metadata)
        launch = compiled[GRID]
        values = tuple(arguments[name] for name in kernel.arg_names)
        return lambda: launch(*values)

    return _Sequence(torch, contract, torch.device("cuda", torch.cuda.current_device()), factory)
