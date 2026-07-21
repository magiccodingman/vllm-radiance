#!/usr/bin/env python3
"""Shared helper for the radiance source patches.

Every `patch_*.py` applies an idempotent one-shot string replacement to an installed vLLM /
transformers source file; they all used a byte-identical `apply()`. It lives here once instead
of being copy-pasted into each patch. Each patch runs as `python patch_X.py` from the directory
that holds this file (the build's `/opt/patches`, or the repo root when run standalone), so
`from _patchlib import apply` resolves as a sibling import.
"""
import ast


def apply(path, anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new` in `path`. Skips if
    `sentinel` is already present; a missing file or non-unique anchor is fatal."""
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({path})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    path.write_text(s)
    print(f"  OK    {label}")
