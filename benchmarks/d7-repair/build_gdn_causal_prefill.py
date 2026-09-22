"""Build an isolated causal-prefill candidate; never alter installed libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private

CONV_SHA256 = "dc9f1a270f06f88716f1e95bf541546b1f2cb90a028ce823cf929f8967f81de8"
HEADERS = {
    "r4d.h": "473e7dfeb5df2a6a7715442c7ae8a95f8a00a8f9ece0fe0629eb91114f6ab25e",
    "r4d_gdn_wmma.h": "9c3c97f5f48d494f15648f3a0fb64fd7ae6ce5ce4961e83617c7d59a7d3ce104",
}


def raw_convolution(source: bytes) -> str:
    if hashlib.sha256(source).hexdigest() != CONV_SHA256:
        raise ValueError("convolution source preimage mismatch")
    text = source.decode()
    begin = text.index("      // inclusive wave scan of the per-lane pair sums")
    end = text.index("      if (i0 < rows) {", begin)
    text = text[:begin] + text[end:]
    text = text.replace(" = pre + g0;", " = g0;")
    text = text.replace(" = pre + g0 + g1;", " = g1;")
    begin = text.index("      float ss = 0.0f;")
    end = text.index("      unsigned short* dst", begin)
    text = (
        text[:begin]
        + "      const float inv = 1.0f; // Normalize in the recurrent kernel.\n"
        + text[end:]
    )
    # Keep the producer ABI, including strided gates and convolution cache handling.
    text = text.replace("r4d_gdn_conv_prep_w4_h128_bf16", "qwen_gdn_conv_raw")
    text = text.replace("r4d_gdn_conv_update_w4_h128_bf16", "qwen_gdn_conv_update_reference")
    return text


def build(source: Path, output: Path):
    conv = source / "r4d_gdn_conv_w4_h128_bf16.hip"
    candidate = raw_convolution(conv.read_bytes())
    headers = {name: (source / name).read_bytes() for name in HEADERS}
    if any(hashlib.sha256(value).hexdigest() != HEADERS[name] for name, value in headers.items()):
        raise ValueError("header preimage mismatch")
    recurrence = Path(__file__).with_name("gdn_causal_prefill.hip").read_bytes()
    output.mkdir(mode=0o700)
    (output / "conv.hip").write_text(candidate)
    (output / "recurrence.hip").write_bytes(recurrence)
    (output / "candidate.hip").write_text('#include "conv.hip"\n#include "recurrence.hip"\n')
    for name, value in headers.items():
        (output / name).write_bytes(value)
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
        "-cuid=qwen_causal_" + hashlib.sha256(candidate.encode() + recurrence).hexdigest(),
        str(output / "candidate.hip"),
        "-o",
        str(output / "candidate.so"),
    ]
    started = time.monotonic()
    done = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    (output / "compiler.log").write_text(done.stdout + done.stderr)
    report = {
        "status": "BUILT_UNTESTED" if done.returncode == 0 else "BUILD_FAILED",
        "command": command,
        "returncode": done.returncode,
        "seconds": time.monotonic() - started,
        "gpu_used": False,
        "original_conv_sha256": CONV_SHA256,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.suffix in (".hip", ".h")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Raw convolution/gates plus causal FP32 recurrence; experimental, not deployed.",
    }
    if done.returncode == 0:
        report["binary_sha256"] = hashlib.sha256((output / "candidate.so").read_bytes()).hexdigest()
    report = seal(report)
    write_private(output / "build.json", report)
    print(json.dumps(report), flush=True)
    return done.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(build(args.source, args.output))
