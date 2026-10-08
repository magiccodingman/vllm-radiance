import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


@pytest.fixture
def adapter():
    path = Path(__file__).parents[1] / "optimized_d7_performance.py"
    spec = importlib.util.spec_from_file_location("performance_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_receipt_requires_the_same_binary_and_successful_check(adapter, tmp_path):
    build = seal({"status": "BUILT_UNTESTED", "binary_sha256": "a" * 64})
    evidence = seal({"status": "SAMPLE_CHECKED", "build": build["sha256"]})
    a, b = tmp_path / "build", tmp_path / "evidence"
    a.mkdir()
    b.mkdir()
    write_private(a / "build.json", build)
    write_private(b / "result.json", evidence)
    entry = {
        "build": str(a),
        "qualification": str(b),
        "build_sha256": build["sha256"],
        "qualification_sha256": evidence["sha256"],
    }
    assert adapter.qualified_stage(entry) == build
    with pytest.raises(DiagnosticError, match="build changed"):
        adapter.qualified_stage({**entry, "build_sha256": "b" * 64})
    with pytest.raises(DiagnosticError, match="evidence changed"):
        adapter.qualified_stage({**entry, "qualification_sha256": "b" * 64})
    (b / "result.json").unlink()
    bad = seal({"status": "SAMPLE_CHECKED", "build": "b" * 64})
    write_private(b / "result.json", bad)
    with pytest.raises(DiagnosticError, match="different build"):
        adapter.qualified_stage({**entry, "qualification_sha256": bad["sha256"]})


@pytest.mark.parametrize(
    "change",
    [
        {"r4d_plan": [(0, 1, 7, 0)]},
        {"r4d_plan": [(0, 2, 8, 0)]},
        {"r4d_plan": [(0, 1, 8, 8)]},
        {"r4d_plan": [(0, 1, 8, 0), (1, 1, 8, 8)]},
        {"r4d_max_ctx": 1023},
        {"causal": False},
    ],
)
def test_unqualified_attention_domains_keep_repaired_fallback(adapter, change):
    impl = SimpleNamespace(num_heads=24, num_kv_heads=4, head_size=256, scale=256**-0.5)
    state = {"r4d_plan": ((0, 1, 8, 0),), "r4d_max_ctx": 60008, "causal": True}
    assert adapter.shared_attention_admitted(impl, SimpleNamespace(**state), None, None)
    assert not adapter.shared_attention_admitted(
        impl,
        SimpleNamespace(**{**state, **change}),
        None,
        None,
    )
    assert not adapter.shared_attention_admitted(impl, SimpleNamespace(**state), 1.0, None)
