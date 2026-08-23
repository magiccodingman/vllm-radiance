#!/usr/bin/env python3
"""gfx1201 (RDNA4) gated-delta-net fp16-WMMA for the linear-attention triangular block-inverse.

Since vLLM 0.26.0 the KKt-gram cast is handled upstream: chunk_scaled_dot_kkt.py casts the gram dot to
the key dtype on RDNA (`_CAST_DOT_TO_K_DTYPE = on_gfx1x()`, i.e. `tl.dot(b_kb.to(b_k.dtype), ...)`),
which selects the WMMA matrix cores, so it is not patched here. Upstream casts to the key's bf16
(7-bit mantissa); an fp16 cast (10-bit) yields a finer gram (~6.9e-5, near TF32).
Keys are L2-normalized, so bf16 is expected to be within tolerance; if the bf16 gram regresses on a
correctness check, add an fp16 cast in chunk_scaled_dot_kkt.py. Upstream gates on on_gfx1x() (always
on for gfx1201), so only the solve_tril half below is patched here.

The solve_tril triangular block-inverse is still on the fp32 path upstream (all 16 dots use
input_precision=DOT_PRECISION), so this patch keeps covering it. The 64x64 block-inverse kernel's
fp32-operand tl.dot()s are routed through a helper that casts both operands to fp16, selecting
WMMA. The dots are chained (each dot-result feeds the next), so the helper casts its inputs on every
call, which also downcasts the intermediate fp32 results. Operands are the bounded gram and its
unit-diagonal inverse (O(1), fp16-safe) and the inverse is stored bf16 anyway, so fp16 is finer than
the storage target.

vLLM 0.26.0 moved the FLA ops from vllm/model_executor/layers/fla/ops/ to
vllm/third_party/flash_linear_attention/ops/. Idempotent string patch of the installed site-packages
copy; safe to re-run (skips if `_tril_dot` is already present); exits nonzero before writing on source
drift."""
import ast
import sysconfig
from pathlib import Path

SOLVE = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/third_party/flash_linear_attention/ops/solve_tril.py"
)


def transform_solve_tril_source(s):
    """Pure string transform of solve_tril.py source -> fp16-WMMA source. Shared by the installer
    (`patch_solve_tril`) and the isolation A/B harness so the tested kernel is byte-identical to the
    shipped one. Raises SystemExit if the vendored source drifted from the exact shapes below."""
    counts = (
        s.count("tl.dot("),
        s.count(", input_precision=DOT_PRECISION)"),
        s.count("        input_precision=DOT_PRECISION,\n    )"),
        s.count("    DOT_PRECISION: tl.constexpr,\n):"),
        s.count("        DOT_PRECISION=FLA_TRIL_PRECISION,\n    )"),
    )
    if counts != (18, 11, 7, 3, 1):
        raise SystemExit(
            f"  FAIL  solve_tril: unexpected anchor counts {counts}, expected (18, 11, 7, 3, 1) "
            "- source changed, re-verify before patching"
        )
    # The cast is unconditional, so only two edits are needed: route every dot through the helper,
    # then define it. All 18 dots already forward input_precision, which the helper accepts and
    # ignores (fp16 operands select WMMA regardless of the requested fp32 precision).
    # 1. route every dot through the fp16-casting helper.
    s = s.replace("tl.dot(", "_tril_dot(")
    # 2. Inject the helper AFTER step 1 so its own tl.dot is not rewritten.
    anchor = (
        '    f"FLA_TRIL_PRECISION must be one of {ALLOWED_TRIL_PRECISIONS}, '
        'but got {FLA_TRIL_PRECISION}"\n)\n'
    )
    if s.count(anchor) != 1:
        raise SystemExit("  FAIL  solve_tril: helper injection anchor not unique")
    helper = anchor + (
        "\n\n"
        "@triton.jit\n"
        "def _tril_dot(a, b, input_precision: tl.constexpr):\n"
        "    # gfx1201 (RDNA4) has no fp32/tf32 matrix-core path, so an fp32-operand dot\n"
        "    # lowers to a slow scalar loop. Casting both operands to fp16 selects WMMA.\n"
        "    # The block-inverse operands are the L2-normalized-key gram and its unit-\n"
        "    # diagonal inverse (all O(1)); the result is stored bf16 downstream, so fp16\n"
        "    # (10-bit mantissa) intermediates are strictly finer than the store target.\n"
        "    return tl.dot(a.to(tl.float16), b.to(tl.float16))\n"
    )
    return s.replace(anchor, helper, 1)


def patch_solve_tril():
    """Apply the fp16-WMMA treatment to the gated-delta-net triangular solve.
    Idempotent: skips if `_tril_dot` is already present. Fails loudly (before writing) if the vendored
    source drifted from the exact dot/signature/launcher shapes the string edits expect."""
    if not SOLVE.exists():
        raise SystemExit(f"  FAIL  solve_tril: {SOLVE} missing")
    s = SOLVE.read_text()
    if "_tril_dot" in s:
        print("  NOOP  solve_tril already applied")
        return
    s = transform_solve_tril_source(s)  # raises on any source drift
    ast.parse(s)  # never write a file that would not parse
    SOLVE.write_text(s)
    print("  OK    solve_tril fp16-WMMA")


def main():
    # KKt gram is upstream since 0.26.0 (see module docstring); only the triangular solve remains.
    patch_solve_tril()


if __name__ == "__main__":
    main()
