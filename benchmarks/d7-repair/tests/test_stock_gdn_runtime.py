"""CPU validation of the exact repair bundle admitted by the Pi replay."""

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1])
    )
    module = importlib.import_module("stock_gdn_runtime")
    for name in module.REQUIRED_FILES:
        (tmp_path / name).write_text(name)
    convolution = tmp_path / "convolution.py"
    convolution.write_text("reviewed convolution")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "stock_gdn_runtime.py"))
    monkeypatch.setattr(module, "CONV_SOURCE", hashlib.sha256(convolution.read_bytes()).hexdigest())
    manifest = {
        "schema": "urn:qwen:d7-stock-repair-bundle:v4",
        "prefill": True,
        "stock_norm": True,
        "stock_head": True,
        "stock_attention": True,
        "head_group_size": 4,
        "convolution": str(convolution),
        "sources": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in module.REQUIRED_FILES
        },
    }
    return module, tmp_path, manifest


def test_validating_the_bundle_requires_no_torch_or_gpu(bundle):
    module, root, manifest = bundle
    sealed = seal(manifest)
    path = root / "manifest.json"
    write_private(path, sealed)
    assert module.validate_manifest(path) == sealed


@pytest.mark.parametrize(
    "fault",
    [
        "omitted-source",
        "changed-source",
        "changed-convolution",
        "invalid-prefill",
        "missing-norm",
        "missing-head-group",
        "invalid-head-group",
        "boolean-head-group",
    ],
)
def test_incomplete_or_changed_bundle_cannot_be_used(bundle, fault):
    module, root, manifest = bundle
    if fault == "omitted-source":
        manifest["sources"].pop("stock_gdn_scan_kernel.py")
    elif fault == "changed-source":
        (root / "stock_gdn_scan_kernel.py").write_text("different arithmetic")
    elif fault == "changed-convolution":
        Path(manifest["convolution"]).write_text("different rounding")
    elif fault == "invalid-prefill":
        manifest["prefill"] = "false"
    elif fault == "missing-norm":
        manifest.pop("stock_norm")
    elif fault == "missing-head-group":
        manifest.pop("head_group_size")
    else:
        manifest["head_group_size"] = True if fault == "boolean-head-group" else 8
    path = root / "manifest.json"
    write_private(path, seal(manifest))
    with pytest.raises(DiagnosticError):
        module.validate_manifest(path)


@pytest.mark.parametrize("limit", [0, 1, 31, 32, 47])
def test_global_fusion_flag_with_unreachable_branch_is_admitted(bundle, limit):
    module, _, _ = bundle
    module.validate_separate_gdn_dispatch(
        SimpleNamespace(FUSED_UPDATE_ON=True, FUSED_MAX_ITEMS=limit, NORM_FUSE=False)
    )


@pytest.mark.parametrize("limit", [48, 64, 96])
def test_reachable_fusion_cannot_bypass_the_repair(bundle, limit):
    module, _, _ = bundle
    with pytest.raises(DiagnosticError, match="separate convolution"):
        module.validate_separate_gdn_dispatch(
            SimpleNamespace(FUSED_UPDATE_ON=True, FUSED_MAX_ITEMS=limit, NORM_FUSE=False)
        )


def test_disabled_fusion_and_enabled_norm_fusion_have_distinct_admission(bundle):
    module, _, _ = bundle
    native = SimpleNamespace(FUSED_UPDATE_ON=False, FUSED_MAX_ITEMS=96, NORM_FUSE=False)
    module.validate_separate_gdn_dispatch(native)
    native.NORM_FUSE = True
    with pytest.raises(DiagnosticError, match="separate convolution"):
        module.validate_separate_gdn_dispatch(native)
