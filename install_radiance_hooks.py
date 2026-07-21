#!/usr/bin/env python3
"""Add radiance_kernels.install_all() to vLLM's plugin loader.
load_general_plugins() runs once per process (parent + engine-core + every TP worker) after
torch/vllm/aiter are imported but before the model loads, which is exactly when the runtime hooks
need to install. Idempotent, env-gated per hook."""
import ast
import sysconfig
from pathlib import Path

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/plugins/__init__.py"
ANCHOR = ("    # general plugins, we only need to execute the loaded functions\n"
          "    for func in plugins.values():\n"
          "        func()")
NEW = ANCHOR + "\n\n" + (
    "    # radiance runtime hooks (env-gated, idempotent; vllm/aiter imported, model not yet loaded)\n"
    "    try:\n"
    "        import radiance_kernels\n"
    "        radiance_kernels.install_all()\n"
    "    except Exception as _e:\n"
    "        import sys as _s\n"
    "        _s.stderr.write(f\"[radiance] install_all failed: {_e!r}\\n\")")


def main():
    if not F.exists():
        raise SystemExit(f"FAIL: {F} missing")
    s = F.read_text()
    if "radiance_kernels.install_all" in s:
        print("  NOOP  radiance plugin hook already installed"); return
    if ANCHOR not in s:
        raise SystemExit(f"  FAIL  anchor not found in {F} (vLLM version drift?)")
    out = s.replace(ANCHOR, NEW, 1)
    ast.parse(out)  # never write a file that would not parse
    F.write_text(out)
    print(f"  OK    radiance install_all hook -> vllm/plugins/__init__.py")


if __name__ == "__main__":
    main()
