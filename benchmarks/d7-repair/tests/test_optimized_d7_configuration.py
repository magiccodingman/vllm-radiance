import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    path = Path(__file__).parents[1] / "benchmark_optimized_d7.py"
    spec = importlib.util.spec_from_file_location("optimized_d7_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiled_profile_keeps_production_capture_sizes_and_does_not_mutate_spec(driver):
    source = {
        "native_config": {
            "enforce_eager": True,
            "async_scheduling": False,
            "kv_transfer_config": {"private": "store"},
            "max_num_seqs": 2,
            "speculative_config": {"method": "dflash", "num_speculative_tokens": 7},
        }
    }
    config = driver.make_config(source, "old-bf16")
    assert config["enforce_eager"] is False
    assert config["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    assert "async_scheduling" not in config
    assert "kv_transfer_config" not in config
    assert config["max_num_seqs"] == 2
    assert config["speculative_config"]["num_speculative_tokens"] == 7
    assert source["native_config"]["enforce_eager"] is True
    assert "kv_transfer_config" in source["native_config"]


def test_old_and_fixed_use_identical_compilation_configuration(driver):
    spec = {"native_config": {"max_num_seqs": 2}}
    assert driver.make_config(spec, "old-bf16") == driver.make_config(spec, "fixed-bf16")


def test_m1_control_only_removes_speculation_and_keeps_graphs(driver):
    config = driver.make_config(
        {"native_config": {"speculative_config": {"method": "dflash"}}},
        "old-bf16",
        speculation=False,
    )
    assert "speculative_config" not in config
    assert config["async_scheduling"] is False
    assert config["enforce_eager"] is False
    assert config["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
