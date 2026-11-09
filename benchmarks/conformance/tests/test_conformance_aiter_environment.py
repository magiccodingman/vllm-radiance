from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_runtime import worker_environment
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def spec(environment):
    return {"environment": environment, "binding": {"live_data_abi": "synthetic-test"}}


def test_source_copy_and_jit_cache_are_separate(tmp_path, monkeypatch):
    monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
    source = tmp_path / "source"
    (source / "csrc").mkdir(parents=True)
    (source / "csrc/kernel.cpp").write_text("original")
    case = tmp_path / "case"
    env = worker_environment(spec({"AITER_META_DIR": str(source)}), case)
    copied = Path(env["AITER_META_DIR"])
    assert copied != source
    assert (copied / "csrc/kernel.cpp").read_text() == "original"
    assert Path(env["AITER_JIT_DIR"]).is_dir()
    assert Path(env["AITER_JIT_DIR"]) != copied
    (copied / "csrc/kernel.cpp").write_text("compiled-case")
    assert (source / "csrc/kernel.cpp").read_text() == "original"


def test_ignored_legacy_root_and_missing_source_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
    with pytest.raises(DiagnosticError, match="AITER_ROOT_DIR is ignored"):
        worker_environment(spec({"AITER_ROOT_DIR": str(tmp_path)}), tmp_path / "legacy")
    with pytest.raises(DiagnosticError, match="must contain csrc"):
        worker_environment(spec({"AITER_META_DIR": str(tmp_path)}), tmp_path / "missing")
