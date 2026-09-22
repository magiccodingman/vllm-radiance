"""The full-model rotary experiment must preserve all other requested controls."""

import importlib.util
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "rotary_experiment", root / "benchmark_rotary_contract_d7.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("capture", [False, True])
def test_only_selects_declared_rotary_worker(driver, capture):
    spec = {"native_config": {"max_num_seqs": 2, "seed": 19, "max_num_batched_tokens": 2048}}
    kwargs = {"execution_mode": "eager", "isolated_capture": capture}
    original = driver.BASE_CONFIG(spec, "fixed-bf16", **kwargs)
    actual = driver.make_config(spec, "fixed-bf16", **kwargs)
    assert actual.pop("worker_cls") == "rotary_mode_d7_worker.RotaryRne" + (
        "CaptureWorker" if capture else "Worker"
    )
    original.pop("worker_cls")
    assert actual == original


@pytest.mark.parametrize("options", [{"execution_mode": "compiled"}, {"speculation": False}])
def test_rejects_other_modes_and_serial_arm(driver, options):
    with pytest.raises(DiagnosticError):
        driver.make_config({"native_config": {}}, "fixed-bf16", **options)


def test_binds_the_reused_driver(driver):
    assert (
        driver.hashlib.sha256(Path(driver.benchmark.__file__).read_bytes()).hexdigest()
        == driver.BASE_DRIVER_SHA256
    )
