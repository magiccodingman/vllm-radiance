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
