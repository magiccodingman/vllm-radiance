"""CPU checks of capture bookkeeping; no claim to exercise GPU dispatch hooks."""

import ast
import functools
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


class Tensor:
    """A small NumPy transport for testing selection and file ownership only."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    dtype = np.dtype("float32")

    def numel(self):
        return self.array.size

    def detach(self):
        return self

    def to(self, **kwargs):
        assert kwargs == {"device": "cpu", "copy": True}
        return Tensor(self.array.copy())

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))

    def stride(self):
        return tuple(v // 4 for v in self.array.strides)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.array.ctypes.data)

    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))

    def __getitem__(self, index):
        return Tensor(self.array[index])


def save(tensors, path):
    with path.open("xb") as f:
        np.savez(f, **{k: v.array for k, v in tensors.items()})


@pytest.fixture
def capture_class():
    # Execute the actual class bodies while excluding GPU-only import/init code.
    directory = Path(__file__).resolve().parents[1]
    env = {
        "torch": SimpleNamespace(
            Tensor=Tensor,
            bfloat16="bfloat16",
            float16=np.dtype("float16"),
            float32=Tensor.dtype,
            float8_e4m3fn="float8",
            save=save,
        ),
        "Path": Path,
        "hashlib": hashlib,
        "Counter": Counter,
        "HookSet": HookSet,
        "DiagnosticError": DiagnosticError,
        "seal": seal,
        "private_json": private_json,
        "write_private": write_private,
        "functools": functools,
    }
    for filename, names in (
        ("isolated_d7_capture.py", {"row_tensor", "BoundaryCapture"}),
        ("execution_mode_d7_worker.py", {"ModeBoundaryCapture"}),
    ):
        path = directory / filename
        env["__file__"] = str(path)
        module = ast.parse(path.read_text(), filename=str(path))
        module.body = [node for node in module.body if getattr(node, "name", None) in names]
        exec(compile(module, str(path), "exec"), env)
    return env["ModeBoundaryCapture"]


def setup_capture(cls, tmp_path):
    probe = SimpleNamespace(
        pending_positions=None,
        schedule=SimpleNamespace(prefix=list(range(12)), output=list(range(321))),
        runner=SimpleNamespace(model=SimpleNamespace(named_parameters=list, named_buffers=list)),
    )
    return probe, cls(
        probe, SimpleNamespace(draft=False), tmp_path / "capture", require_compiled=False
    )


def invoke(capture, probe, positions, *, flattened=False):
    probe.pending_positions = list(positions)
    count = len(probe.pending_positions)
    tensor = Tensor(np.arange(count * (48 if flattened else 1) * 4).reshape(-1, 4))
    capture.invoke("aten.add.Tensor", lambda x: Tensor(x.array + 1), (tensor,), {})


def complete_decode(capture, probe):
    for start in range(12, 332, 8):
        invoke(capture, probe, range(start, start + 8))


def test_prefill_and_decode_files_cannot_overwrite_each_other(capture_class, tmp_path):
    probe, capture = setup_capture(capture_class, tmp_path)
    invoke(capture, probe, range(6))
    invoke(capture, probe, range(6, 12), flattened=True)
    complete_decode(capture, probe)
    result = capture.finish()
    assert result["positions"] == 320
    decode = private_json(capture.root / "manifest.json")
    prefill = private_json(capture.root / "prefill-manifest.json")
    assert prefill["positions"] == [*range(8), 11]
    assert [b["positions"] for b in prefill["batches"]] == [list(range(6)), [6, 7, 11]]
    batches = decode["batches"] + prefill["batches"]
    assert len({b["file"] for b in batches}) == 42
    for record in batches:
        path = capture.root / record["file"]
        metadata = private_json(path.with_suffix(".json"))
        authenticate(metadata)
        assert metadata["sha256"] == record["sha256"]
        assert metadata["tensor_file"] == path.name
        assert metadata["tensor_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as tensors:
            expected_rows = len(record["positions"]) * (48 if path.name == "prefill-0001.pt" else 1)
            assert all(t.shape == (expected_rows, 4) for t in tensors.values())


def test_missing_prefill_samples_cannot_claim_complete_capture(capture_class, tmp_path):
    probe, capture = setup_capture(capture_class, tmp_path)
    invoke(capture, probe, range(6))
    invoke(capture, probe, [11])  # Deliberately omit positions 6 and 7.
    complete_decode(capture, probe)
    with pytest.raises(DiagnosticError, match="prefill boundary coverage"):
        capture.finish()
    assert not (capture.root / "prefill-manifest.json").exists()


def test_unsampled_prefill_and_drafter_do_not_create_capture_files(capture_class, tmp_path):
    probe, capture = setup_capture(capture_class, tmp_path)
    invoke(capture, probe, [8, 9, 10])
    capture.observation.draft = True
    invoke(capture, probe, range(12, 20))
    assert not capture.events and not capture.tensors
    assert list(capture.root.iterdir()) == []


def test_duplicate_decode_positions_cannot_replace_missing_positions(capture_class, tmp_path):
    probe, capture = setup_capture(capture_class, tmp_path)
    invoke(capture, probe, range(12))
    complete_decode(capture, probe)
    capture.flush()
    capture.batches[1]["positions"] = capture.batches[0]["positions"]
    with pytest.raises(DiagnosticError, match="captured position domain"):
        capture.finish()
    assert not (capture.root / "manifest.json").exists()


@pytest.mark.parametrize("fail_attach", [False, True])
def test_mode_probe_creates_parent_before_evidence_and_cleans_failed_hooks(tmp_path, fail_attach):
    path = (
        Path(__file__).resolve().parents[1]
        / "execution_mode_d7_worker.py"
    )
    module = ast.parse(path.read_text())
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ExecutionModeWorker"
    )
    cls.bases = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "qwen_optimized_begin"
    ]
    calls = []

    class Observation:
        def __init__(self, runner, private, profile):
            Path(private).mkdir()
            calls.append("observation")

        def close(self):
            calls.append("observation_closed")

    class Probe:
        def __init__(self, runner, task):
            (tmp_path / "pass-00/correctness").mkdir()
            calls.append("probe")

        def attach(self):
            if fail_attach:
                raise DiagnosticError("injected attachment failure")

        def close(self):
            calls.append("probe_closed")

    env = {
        "GraphObservation": Observation,
        "ExecutionModeProbe": Probe,
        "DiagnosticError": DiagnosticError,
        "Path": Path,
        "private_json": lambda p: {},
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), env)
    worker = env["ExecutionModeWorker"]()
    worker.model_runner = object()
    if fail_attach:
        with pytest.raises(DiagnosticError, match="injected"):
            worker.qwen_optimized_begin(tmp_path / "pass-00", task_path="task.json")
        assert calls == ["observation", "probe", "probe_closed", "observation_closed"]
        assert not hasattr(worker, "_qwen_observation")
        assert not hasattr(worker, "_qwen_forced")
    else:
        worker.qwen_optimized_begin(tmp_path / "pass-00", task_path="task.json")
        assert calls == ["observation", "probe"]
