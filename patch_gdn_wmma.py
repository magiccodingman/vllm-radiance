#!/usr/bin/env python3
"""gfx1201 (RDNA4) gated-delta-net fp16-WMMA: run the fp32-operand tl.dot()s in the linear-attention
prefill on the fp16 matrix cores (WMMA) instead of the slow RDNA4 scalar fp32 path. Covers two
kernels -- the chunk_scaled_dot_kkt K*K^T gram and the solve_tril triangular block-inverse -- both
behind the single RADIANCE_GDN_WMMA env (default on; set 0 to keep the stock fp32 dots). Idempotent
string patches of the installed site-packages copies; re-running is safe.

KKt gram (chunk_scaled_dot_kkt.py): keys are bf16 but beta is fp32, so `b_kb = k * beta` is fp32 and
the stock `tl.dot(b_kb, trans(k).to(b_kb.dtype))` is an fp32 matmul. RDNA4 has no fp32 matrix-core
path, so that dot lowers to a scalar/vector loop that is slow to run and very slow to compile (some
autotune configs take minutes each, which dominates cold start). Casting both operands to fp16 selects
the WMMA matrix cores. fp16 keeps a 10-bit mantissa (vs bf16's 7) and the keys are L2-normalized, so
every dot operand is well within fp16 range.

Triangular solve (solve_tril.py): the 64x64 block-inverse kernel runs 16 fp32-operand tl.dot()s
(input_precision="ieee") with the same RDNA4 fp32 penalty. Every dot is routed through a helper that
casts both operands to fp16 when the gate is on; the dots are chained, so the helper also downcasts
each intermediate fp32 dot-result. Operands are the bounded gram and its unit-diagonal inverse (O(1),
fp16-safe) and the inverse is stored bf16 anyway, so fp16 is strictly finer than the storage target."""
import ast
import sysconfig
from pathlib import Path

KKT = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py"
SOLVE = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/fla/ops/solve_tril.py"


def apply(anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new`. Skips if `sentinel`
    is already present; a missing file or non-unique anchor is fatal."""
    if not KKT.exists():
        raise SystemExit(f"  FAIL  {label}: {KKT} missing")
    s = KKT.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({KKT})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    KKT.write_text(s)
    print(f"  OK    {label}")


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
    # 1. inline dots: add the CAST_FP16 arg, keep input_precision forwarded for the stock branch.
    s = s.replace(
        ", input_precision=DOT_PRECISION)",
        ", CAST_FP16, input_precision=DOT_PRECISION)",
    )
    # 2. multi-line dots: insert CAST_FP16 before the trailing input_precision kwarg.
    s = s.replace(
        "        input_precision=DOT_PRECISION,\n    )",
        "        CAST_FP16,\n        input_precision=DOT_PRECISION,\n    )",
    )
    # 3. route every dot through the fp16-casting helper.
    s = s.replace("tl.dot(", "_tril_dot(")
    # 4. thread CAST_FP16 through all three kernel signatures.
    s = s.replace(
        "    DOT_PRECISION: tl.constexpr,\n):",
        "    DOT_PRECISION: tl.constexpr,\n    CAST_FP16: tl.constexpr,\n):",
    )
    # 5. pass the env gate from the shared launcher.
    s = s.replace(
        "        DOT_PRECISION=FLA_TRIL_PRECISION,\n    )",
        "        DOT_PRECISION=FLA_TRIL_PRECISION,\n        CAST_FP16=RADIANCE_GDN_WMMA,\n    )",
    )
    # 6. Inject the env gate + helper AFTER step 3 so the helper's own tl.dot is not rewritten.
    anchor = (
        '    f"FLA_TRIL_PRECISION must be one of {ALLOWED_TRIL_PRECISIONS}, '
        'but got {FLA_TRIL_PRECISION}"\n)\n'
    )
    if s.count(anchor) != 1:
        raise SystemExit("  FAIL  solve_tril: env/helper injection anchor not unique")
    helper = anchor + (
        "\n"
        'RADIANCE_GDN_WMMA = os.environ.get("RADIANCE_GDN_WMMA", "1") == "1"\n'
        "\n\n"
        "@triton.jit\n"
        "def _tril_dot(a, b, CAST_FP16: tl.constexpr, input_precision: tl.constexpr):\n"
        "    # gfx1201 (RDNA4) has no fp32/tf32 matrix-core path, so an fp32-operand dot\n"
        "    # lowers to a slow scalar loop. Casting both operands to fp16 selects WMMA.\n"
        "    # The block-inverse operands are the L2-normalized-key gram and its unit-\n"
        "    # diagonal inverse (all O(1)); the result is stored bf16 downstream, so fp16\n"
        "    # (10-bit mantissa) intermediates are strictly finer than the store target.\n"
        "    if CAST_FP16:\n"
        "        return tl.dot(a.to(tl.float16), b.to(tl.float16))\n"
        "    else:\n"
        "        return tl.dot(a, b, input_precision=input_precision)\n"
    )
    return s.replace(anchor, helper, 1)


def patch_solve_tril():
    """Extend the same RADIANCE_GDN_WMMA fp16-WMMA treatment to the gated-delta-net triangular solve.
    solve_tril's 64x64 block-inverse kernel runs 16 fp32-operand tl.dot()s that hit the same slow
    RDNA4 scalar path as the stock KKt gram. This is a multi-step transform (not a single anchor
    replace), so it has its own function rather than going through `apply`.

    Every block dot is routed through a `_tril_dot` helper that casts both operands to fp16 when the
    gate is on. The dots are chained (the result of one dot is an operand of the next), so casting only
    the loaded operands would leave the outer dots on the fp32 path -- the helper casts its inputs each
    call, which downcasts the intermediate fp32 dot-results too. The stock fp32 behaviour (with the
    original input_precision) is preserved verbatim in the else branch when the gate is off.

    Idempotent: skips if `_tril_dot` is already present. Fails loudly (before writing) if the vendored
    source drifted from the exact dot/signature/launcher shapes the string edits below expect."""
    if not SOLVE.exists():
        raise SystemExit(f"  FAIL  solve_tril: {SOLVE} missing")
    s = SOLVE.read_text()
    if "_tril_dot" in s:
        print("  NOOP  solve_tril already applied")
        return
    s = transform_solve_tril_source(s)  # raises on any source drift
    ast.parse(s)  # never write a file that would not parse
    SOLVE.write_text(s)
    print("  OK    solve_tril fp16-WMMA (gated on RADIANCE_GDN_WMMA)")


def main():
    # 1. Module-level env gate.
    apply(
        "from .utils import FLA_CHUNK_SIZE\n",
        "from .utils import FLA_CHUNK_SIZE\n"
        "\n"
        "import os as _os\n"
        'RADIANCE_GDN_WMMA = _os.environ.get("RADIANCE_GDN_WMMA", "1") == "1"\n',
        "RADIANCE_GDN_WMMA =",
        "env gate",
    )
    # 2. Kernel constexpr flag.
    apply(
        "    IS_VARLEN: tl.constexpr,\n    USE_G: tl.constexpr,\n):",
        "    IS_VARLEN: tl.constexpr,\n    USE_G: tl.constexpr,\n    CAST_FP16: tl.constexpr,\n):",
        "CAST_FP16: tl.constexpr,",
        "kernel flag",
    )
    # 3. fp16 matrix-core dot branch.
    apply(
        "        b_kb = b_k * b_beta[:, None]\n"
        "        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))\n",
        "        b_kb = b_k * b_beta[:, None]\n"
        "        if CAST_FP16:\n"
        "            b_A += tl.dot(b_kb.to(tl.float16), tl.trans(b_k).to(tl.float16))\n"
        "        else:\n"
        "            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))\n",
        "if CAST_FP16:",
        "fp16 dot branch",
    )
    # 4. Pass the flag from the launcher.
    apply(
        "        K=K,\n        BT=BT,\n    )\n    return A",
        "        K=K,\n        BT=BT,\n        CAST_FP16=RADIANCE_GDN_WMMA,\n    )\n    return A",
        "CAST_FP16=RADIANCE_GDN_WMMA,",
        "pass flag",
    )
    # 5. Same fp16-WMMA treatment for the gated-delta-net triangular block-inverse.
    patch_solve_tril()


if __name__ == "__main__":
    main()
