"""Build the isolated stock-M1 normalization candidate without using a GPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def gather_reduction(source):
    """Keep the logical reduction tree but gather its upper levels in one wave."""
    begin = "  partial[tid] = total;\n  // Match native"
    end = "  if (tid < 32) {\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("normalization reduction source changed")
    start, stop = source.index(begin), source.index(end)
    return (
        source[:start]
        + """  partial[tid] = total;
  // Every lane gathers the same inter-wave reduction tree. The arithmetic
  // order is unchanged, but only one barrier is needed to publish partials.
  __syncthreads();
  if (tid < 32) {
    float values[THREADS / 32];
#pragma unroll
    for (int wave = 0; wave < THREADS / 32; ++wave)
      values[wave] = partial[tid + wave * 32];
#pragma unroll
    for (int offset = THREADS / 64; offset > 0; offset >>= 1) {
#pragma unroll
      for (int wave = 0; wave < offset; ++wave)
        values[wave] += values[wave + offset];
    }
    total = values[0];
  }
"""
        + source[stop:]
    )


def retain_inputs(source):
    """Reuse the exact FP32 values already squared instead of loading them twice."""
    declaration = "  float accum[4] = {0, 0, 0, 0};"
    square = "      const float square = z * z;"
    begin = "  for (int column = tid; column < WIDTH; column += THREADS) {"
    end = "\n}\n\ntemplate <int WIDTH, int THREADS>"
    for anchor in (declaration, square, begin, end):
        if source.count(anchor) != 1:
            raise ValueError("normalization input reuse source changed")
    source = source.replace(
        declaration, declaration + "\n  float retained[(WIDTH / 4 + THREADS - 1) / THREADS][4];"
    )
    source = source.replace(square, "      retained[vector / THREADS][component] = z;\n" + square)
    start, stop = source.index(begin), source.index(end)
    return (
        source[:start]
        + """  for (int vector = tid; vector < WIDTH / 4; vector += THREADS) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const int column = vector * 4 + component;
      const float z = retained[vector / THREADS][component];
      if (RESIDUAL) residual_output[row * WIDTH + column] = narrow(z);
      const float normalized = z * inverse;
      const float weighted = normalized * (widen(weight[column]) + 1.0f);
      output[row * WIDTH + column] = narrow(weighted);
    }
  }"""
        + source[stop:]
    )


def build(output, *, gather=False, retain=False):
    output.mkdir(mode=0o700)
    source = Path(__file__).with_name("stock_m1_gemma_norm.hip")
    local = output / source.name
    text = source.read_text()
    if gather:
        text = gather_reduction(text)
    if retain:
        text = retain_inputs(text)
    local.write_text(text)
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
        "-cuid=qwen_stock_norm_" + hashlib.sha256(local.read_bytes()).hexdigest(),
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
            "reduction": "gathered-identical-tree" if gather else "distributed-barriers",
            "retained_input_registers": retain,
            "kernel_abi": "qwen-stock-m1-norm-v3",
            "source_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256((output / "candidate.so").read_bytes()).hexdigest()
            if result.returncode == 0
            else None,
            "scope": (
                "BF16 Gemma norm: hidden width 5120 and Q/K groups 24/4 by 256; "
                "stock ROCm M1 reduction; experimental."
            ),
        }
    )
    write_private(output / "build.json", report)
    print(json.dumps(report), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gather-reduction", action="store_true")
    parser.add_argument("--retain-inputs", action="store_true")
    args = parser.parse_args()
    raise SystemExit(build(args.output, gather=args.gather_reduction, retain=args.retain_inputs))
