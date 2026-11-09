#!/usr/bin/env python3
"""GPU-free immutable LFM3 payload and fail-closed installation contracts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import patch_lfm3_backport as patch


def main():
    vendor = ROOT / "upstream/vllm-lfm3"
    manifest = json.loads((vendor / "SOURCE_PROVENANCE.json").read_text())
    assert manifest["commit"] == patch.COMMIT
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == expected, name
    assert manifest["sha256"][patch.PAYLOAD.name] == patch.PATCH_SHA256
    assert manifest["kernel_verification"]["sha256"] == patch.KERNEL_HASHES
    # Matching forward code is not evidence that the decode hunk was applied.
    # This reproduced the missing width-7 arithmetic/state branches on GPU.
    source = "def forward(x):\n    return x + 1\n\ndef decode(x):\n    return x\n"
    before, after = "    return x\n", "    return x + 1\n"
    fixed = patch._apply_hunk(source, before, after, "decode")
    assert fixed == source.replace("def decode(x):\n" + before,
                                   "def decode(x):\n" + after)
    assert patch._apply_hunk(fixed, before, after, "decode") == fixed
    drifted = source.replace("def decode(x):\n" + before,
                             "def decode(x):\n    return x + 2\n")
    try:
        patch._apply_hunk(drifted, before, after, "decode")
        raise AssertionError("matching forward code hid decode source drift")
    except RuntimeError as exc:
        assert "source drift" in str(exc)
    kernel_source = "\n".join(
        f"def {name}():\n    if True:\n        if True:\n" + patch.PRODUCT_BEFORE
        for name in patch.KERNEL_HASHES
    )
    adapted = patch.adapt_kernel_precision(kernel_source)
    assert patch.canonical_kernel_source(adapted) == kernel_source
    partial = adapted.replace(patch.PRODUCT_AFTER, patch.PRODUCT_BEFORE, 1)
    try:
        patch.canonical_kernel_source(partial)
        raise AssertionError("partially installed precision correction accepted")
    except RuntimeError as exc:
        assert "incomplete FP32 product adaptation" in str(exc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        site = root / "site"
        site.mkdir()
        marker = site / "marker"
        marker.write_text("unchanged")
        try:
            patch.apply_backport(site)
            raise AssertionError("missing stable source accepted")
        except RuntimeError as exc:
            assert "source drift" in str(exc)
        assert list(site.iterdir()) == [marker]
        altered = root / "altered.patch"
        altered.write_bytes(patch.PAYLOAD.read_bytes() + b"\n")
        try:
            patch.apply_backport(site, altered)
            raise AssertionError("altered payload accepted")
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc)
        assert marker.read_text() == "unchanged"
    print("LFM3 upstream hashes and fail-closed installation: PASS (CPU only)")


if __name__ == "__main__":
    main()
