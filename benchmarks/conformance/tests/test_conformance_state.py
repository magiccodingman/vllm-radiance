import hashlib

import numpy as np
import pytest

from qwen_r9700_lab.conformance_state import (
    FrameWriter,
    archive_frame,
    compare_frames,
    load_arrays,
    read_frame,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest


def frame(root, arrays, *, consumed=1, pending=2, logical=None):
    writer = FrameWriter(
        root,
        contract=digest("contract"),
        execution=digest("build"),
        adapter=digest("adapter"),
        input_digest=digest("same input"),
        phase="prefill",
        consumed=consumed,
        pending=pending,
        expected=list(arrays),
        logical=logical,
    )
    for k, v in arrays.items():
        writer.array(k, v)
    return writer.finish()


def test_exact_bytes_despite_zero_numeric_distance(tmp_path):
    frame(tmp_path / "a", {"state": np.asarray([0.0], dtype=np.float32)})
    frame(tmp_path / "b", {"state": np.asarray([-0.0], dtype=np.float32)})
    result = compare_frames(tmp_path / "a", tmp_path / "b", chunk_bytes=8)
    assert not result["equal"]
    assert result["components"][0]["max_abs"] == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_never_passes_even_if_equal(tmp_path, value):
    for name in ("a", "b"):
        frame(tmp_path / name, {"state": np.asarray([value], np.float32)})
    assert not compare_frames(tmp_path / "a", tmp_path / "b")["equal"]


def test_comparison_streams_and_localizes_first_boundary(tmp_path):
    a = {"conv": np.arange(150, dtype=np.float32), "gdn": np.ones((5, 10), dtype=np.float32)}
    b = {k: v.copy() for k, v in a.items()}
    b["conv"][31] += 1
    b["gdn"][0, 0] += 100
    frame(tmp_path / "a", a)
    frame(tmp_path / "b", b)
    result = compare_frames(tmp_path / "a", tmp_path / "b", chunk_bytes=16)
    assert result["first_difference"]["boundary"] == "conv"
    assert 124 <= result["first_difference"]["byte_offset"] < 128
    assert result["components"][0]["max_abs"] == 1


def test_corrupt_snapshot_and_incomplete_capture_refused(tmp_path):
    d = frame(tmp_path / "a", {"state": np.ones(4, np.float32)})
    path = tmp_path / "a" / d["components"]["state"]["file"]
    path.write_bytes(b"bad")
    with pytest.raises(DiagnosticError):
        load_arrays(tmp_path / "a")
    with pytest.raises(DiagnosticError):
        archive_frame(tmp_path / "a", tmp_path / "copy")
    path.write_bytes(np.ones(4, np.float32).tobytes())
    path.chmod(0o644)
    with pytest.raises(DiagnosticError):
        load_arrays(tmp_path / "a")


def test_byte_tamper_with_forged_equal_hash_not_accepted(tmp_path, monkeypatch):
    # Descriptor equality is insufficient: the comparator reads actual bytes.
    a = frame(tmp_path / "a", {"state": np.zeros(4, np.float32)})
    b = frame(tmp_path / "b", {"state": np.ones(4, np.float32)})
    real_read = read_frame

    def lying_read(path):
        doc = real_read(path)
        doc["components"]["state"]["sha256"] = a["components"]["state"]["sha256"]
        return doc

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.read_frame", lying_read)
    with pytest.raises(DiagnosticError, match="sealed descriptor"):
        compare_frames(tmp_path / "a", tmp_path / "b")
    assert a["sha256"] != b["sha256"]


def test_f64_diagnostic_overflow_does_not_crash_or_pass(tmp_path):
    for name, sign in (("a", 1), ("b", -1)):
        frame(tmp_path / name, {"state": np.asarray([sign * 1e308], np.float64)})
    result = compare_frames(tmp_path / "a", tmp_path / "b")
    assert not result["equal"]
    assert result["components"][0]["max_abs"] == "overflow"


def test_archive_is_independent_and_retains_identity(tmp_path):
    a = frame(tmp_path / "a", {"state": np.arange(128, dtype=np.int64)})
    b = archive_frame(tmp_path / "a", tmp_path / "b")
    assert a == b
    name = a["components"]["state"]["file"]
    assert (tmp_path / "a" / name).stat().st_ino != (tmp_path / "b" / name).stat().st_ino
    assert (
        hashlib.sha256((tmp_path / "b" / name).read_bytes()).hexdigest()
        == a["components"]["state"]["sha256"]
    )
