"""Build the isolated interleaved M4 head experiment without using a GPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def build(output):
    output.mkdir(mode=0o700)
    source = Path(__file__).with_name("stock_m1_head_pair.hip")
    local = output / source.name
    local.write_bytes(source.read_bytes())
    command = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-ffp-contract=off",
        "-mcumode",
        "-cuid=qwen_head_pair_" + hashlib.sha256(local.read_bytes()).hexdigest(),
        str(local),
        "-o",
        str(output / "candidate.so"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    report = seal(
        {
            "status": "BUILT_UNTESTED" if result.returncode == 0 else "BUILD_FAILED",
            "returncode": result.returncode,
            "command": command,
            "gpu_used": False,
            "kernel_abi": "qwen-stock-m1-head-pair-v1",
            "source_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256((output / "candidate.so").read_bytes()).hexdigest()
            if result.returncode == 0
            else None,
            "scope": "gfx1201 BF16 [8,5120] x [248320,5120]; no bias; experimental",
        }
    )
    write_private(output / "build.json", report)
    print(json.dumps(report), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(build(parser.parse_args().output))
