#!/usr/bin/env python3
"""Compile the registered source set without shell-dependent glob expansion."""

from __future__ import annotations

import compileall
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def check_sources(repo: Path) -> bool:
    paths = [repo / "_patchlib.py", repo / "install_radiance_hooks.py"]
    for pattern in ("patch_*.py", "radiance_*.py"):
        matches = sorted(repo.glob(pattern))
        if not matches:
            print(f"FAIL: no sources match {pattern}")
            return False
        paths.extend(matches)
    paths.extend([repo / "benchmarks/bin", repo / "tools"])

    passed = True
    for path in paths:
        if not path.exists():
            print(f"FAIL: source path missing: {path}")
            passed = False
        elif path.is_dir():
            if not compileall.compile_dir(str(path), quiet=1, force=True):
                passed = False
        elif not compileall.compile_file(str(path), quiet=1, force=True):
            passed = False
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if check_sources(REPO) else 1)
