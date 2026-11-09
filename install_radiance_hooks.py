#!/usr/bin/env python3
"""Install Radiance through vLLM's native general-plugin entry point.

The per-process loader already runs after torch/platform setup and before model
loading. No edit to vllm/plugins/__init__.py is needed. The individual hooks
remain idempotent and retain their environment gates.
"""
import sysconfig
from pathlib import Path


def main():
    site = Path(sysconfig.get_paths()["purelib"])
    info = site / "radiance_runtime-1.2.0rc1.dist-info"
    info.mkdir(exist_ok=True)
    files = {
        "METADATA": "Metadata-Version: 2.1\nName: radiance-runtime\nVersion: 1.2.0rc1\n",
        "entry_points.txt": "[vllm.general_plugins]\nradiance = radiance_kernels:install_all\n",
    }
    for name, text in files.items():
        (info / name).write_text(text)
    (info / "RECORD").write_text("".join(
        f"{info.name}/{name},,\n" for name in (*files, "RECORD")
    ))
    print("OK Radiance native vLLM general-plugin entry point")


if __name__ == "__main__":
    main()
