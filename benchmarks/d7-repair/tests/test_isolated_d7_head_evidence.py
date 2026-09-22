"""Interrupted native probes must preserve counts without certifying a pass."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import compare_rows, summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate

SPEC = importlib.util.spec_from_file_location(
    "head_probe",
    Path(__file__).resolve().parents[1] / "probe_isolated_d7_head.py",
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def comparison():
    ref = summarize_logits(np.arange(32, dtype=np.float32))
    return compare_rows(ref, ref)


@pytest.mark.parametrize("positions", [0, 8, 320])
def test_partial_counts_never_claim_completed_qualification(positions):
    record = comparison()
    result = PROBE.partial_progress(
        {arm: [record] * positions for arm in ("old", "fixed")}, {"binding": "pinned"}
    )
    authenticate(result)
    assert result["status"] == "INCOMPLETE"
    assert result["pending_checks"]
    assert result["results"]["fixed"]["top20_order_exact"] == positions
    assert "isolated_inputs_verified" not in result["results"]["fixed"]


def test_partial_counts_reject_incomplete_or_different_arms():
    record = comparison()
    for a, b in [(8, 0), (1, 1), (328, 328)]:
        with pytest.raises(DiagnosticError):
            PROBE.partial_progress({"old": [record] * a, "fixed": [record] * b}, {})


def test_streamed_digest_agrees_without_whole_file_read(tmp_path, monkeypatch):
    data = bytes(range(256)) * 8193
    path = tmp_path / "capture.bin"
    path.write_bytes(data)

    def no_whole_file_read(*_):
        raise AssertionError("evidence hash must not allocate a whole file copy")

    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    assert PROBE.file_digest(path) == hashlib.sha256(data).hexdigest()
