#!/usr/bin/env python3
"""Native MXFP4 (OCP micro-scaling) linear GEMM on gfx1201, for Quark W4A4 checkpoints
such as amd/Qwen3.8-27B-Quark-AWQ-MXFP4.

Stock vLLM emulates MXFP4 on this card: every weight tensor is materialised in bf16 on each
forward and the linear runs in high precision. Two things stand in the way of the real kernel,
and neither of them is the compiler:

  1. `RocmPlatform.supports_mx()` allowlists gfx95 (CDNA4), so `AiterMxfp4LinearKernel` reports
     itself unsupported and selection falls through to `EmulationMxfp4LinearKernel`. But Triton
     3.6 *does* lower `tl.dot_scaled` on gfx12x -- it upconverts the e2m1 operands and runs bf16
     WMMA -- which is exactly what aiter's `gemm_afp4wfp4` rides on. Measured on gfx1201:
     bit-identical output to the emulated path (the activation quant is the same either way) and
     ~2.4-4x faster at decode shapes.
  2. vLLM imports the aiter fp4 GEMM from `aiter.ops.triton.gemm_afp4wfp4`, a module path that
     aiter 0.1.17 moved to `aiter.ops.triton.gemm.basic.gemm_afp4wfp4`. Left alone, the native
     branch would raise ImportError the moment it was reached. (`aiter.ops.triton.quant` still
     re-exports dynamic_mxfp4_quant, so that import is left as-is.)

aiter's own `arch_info.is_fp4_avail()` allowlists gfx950/gfx1250 too; it is relaxed in the same
place, lazily, inside the op body -- so no aiter import is added at plugin-load time, where it
would initialise HIP early and force the engine core to spawn instead of fork.

A second, separate opt-in (RADIANCE_MXFP4_W4A8=1) routes large-M linears to a hand-written
fp8-WMMA kernel. Triton will not emit gfx1201's fp8 matrix instruction -- measured
register-resident, fp8 WMMA runs 325.2 TFLOP/s vs f16's 160.2, while Triton's own fp8 tl.dot
manages only 43.3 because it upconverts -- so the 2x is only reachable by hand. Measured against
the tuned aiter path it replaces: 1.47-2.26x faster AND 4.2x more accurate (relative error 0.0265
vs 0.1119 against exact arithmetic), because fp8 activations beat the mxfp4 ones aiter quantizes
to. It is opt-in anyway, since it makes the layer W4A8 rather than the checkpoint's declared W4A4.

Gated by RADIANCE_MXFP4=1 (default off). With it unset the checkpoint still loads and serves,
just on the stock emulated path, and no other quantization scheme is touched.

WHAT CHANGED AT vLLM 0.27
-------------------------
0.26 had all of this inline in `QuarkOCP_MX` and this patch carried seven hunks against that one
file. 0.27 replaced it with a kernel plugin ABC: `MxFp4LinearKernel` (is_supported /
can_implement / process_weights_after_loading / apply_weights), instantiated by
`init_mxfp4_linear_kernel()` from the first entry of `_POSSIBLE_MXFP4_KERNELS[platform]` that
accepts the config. So the patch is now three small hunks against a documented interface:
register our plugin, relax aiter's CDNA4 gate, and fix aiter's moved module path.

RADIANCE_MXFP4_MAX_M is retired. It used to hand large batches back to emulation, where a single
amortised bf16 dequant beat the fp4 kernel past M~256. That crossover only ever mattered against
the *aiter W4A4* path; with W4A8 on, large M goes to the fp8-WMMA kernel, which beats both. It
was also unusable in practice on this stack -- the fallback reached quark's TileLang backend,
which dies with `HIP runtime library (libamdhip64.so) not found` inside the vLLM worker, and the
branch got specialised into the torch.compile graph during the M=8192 profile run, killing
startup rather than one request. Reintroducing it would also mean a Python branch inside
apply_weights, which is exactly the dynamo graph break that cost 33% of decode once already.

Tiles come from mxfp4-configs/gfx1201-GEMM-AFP4WFP4.json, which the Dockerfile drops into aiter's
config dir: every band there pins matrix_instr_nonkdim to 16. aiter's gfx1250 table uses 32 for
M>=64, and gfx1201's WMMA is 16x16x16 only, so those bands fail to compile with
"no matching matrix core intrinsic due to unsupported element type: A='bf16' B='bf16' C='f32'".
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KL = SP / "vllm/model_executor/kernels/linear/__init__.py"
KA = SP / "vllm/model_executor/kernels/linear/mxfp4/aiter.py"


# --- 1. register the radiance W4A8 plugin at the head of the ROCm MXFP4 list ----------------
# Done inside init_mxfp4_linear_kernel rather than at the _POSSIBLE_MXFP4_KERNELS literal: that
# dict is built at module import, in the parent process during config parsing, and importing
# radiance_mxfp4 there would pull in the HIP extension and initialise HIP before the engine core
# forks. init_mxfp4_linear_kernel runs at model load, in the worker, where HIP is already up.
REGISTER_ANCHOR = (
    "    config = MxFp4LinearLayerConfig(\n"
    "        activation_quant_key=activation_quant_key,\n"
    "    )\n"
    "\n"
    "    linear_backend = _get_linear_backend()\n"
    "\n"
    "    platform = current_platform._enum\n"
    "    possible = list(_POSSIBLE_MXFP4_KERNELS.get(platform, []))\n"
)
REGISTER_NEW = (
    "    config = MxFp4LinearLayerConfig(\n"
    "        activation_quant_key=activation_quant_key,\n"
    "    )\n"
    "\n"
    "    linear_backend = _get_linear_backend()\n"
    "\n"
    "    platform = current_platform._enum\n"
    "    possible = list(_POSSIBLE_MXFP4_KERNELS.get(platform, []))\n"
    "\n"
    "    # --- radiance (patch_quark_mxfp4.py): the gfx1201 W4A8 fp8-WMMA kernel ---\n"
    "    # Imported here, not at module scope: this runs in the worker at model load, whereas the\n"
    "    # module is imported in the parent during config parsing, where initialising HIP would\n"
    "    # force the engine core to spawn instead of fork. It declines via is_supported() unless\n"
    "    # RADIANCE_MXFP4_W4A8=1 on gfx12x, so the list is unchanged everywhere else.\n"
    "    try:\n"
    "        import radiance_mxfp4 as _radiance_mxfp4\n"
    "\n"
    "        _radiance_cls = _radiance_mxfp4.kernel_class()\n"
    "        if _radiance_cls is not None:\n"
    "            possible.insert(0, _radiance_cls)\n"
    "    except Exception as _radiance_exc:  # never block model load on our own kernel\n"
    '        logger.warning_once("[radiance] MXFP4 W4A8 kernel unavailable: %r", _radiance_exc)\n'
)


# --- 2. relax aiter's CDNA4 gate so the Triton fp4 GEMM is reachable on gfx1201 -------------
# This is what RADIANCE_MXFP4=1 buys on its own: without it AiterMxfp4LinearKernel declines and
# selection falls through to EmulationMxfp4LinearKernel. Output is bit-identical either way --
# the activation quantization is the same -- so this is a speed change only.
SUPPORTS_ANCHOR = (
    "        if not current_platform.supports_mx():\n"
    '            return False, "current platform does not support native MXFP4 computation"\n'
)
SUPPORTS_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): gfx1201 native MXFP4, RADIANCE_MXFP4=1 ---\n"
    "        # supports_mx() is a CDNA4 allowlist, but Triton 3.6 lowers tl.dot_scaled on gfx12x\n"
    "        # (upconvert + bf16 WMMA), verified bit-identical against the emulated path.\n"
    "        _radiance_mx = current_platform.supports_mx()\n"
    "        if not _radiance_mx:\n"
    "            import os as _radiance_os\n"
    '            if _radiance_os.environ.get("RADIANCE_MXFP4", "0") == "1":\n'
    "                from vllm.platforms.rocm import on_gfx12x\n"
    "\n"
    "                _radiance_mx = bool(on_gfx12x())\n"
    "                if _radiance_mx:\n"
    "                    logger.warning_once(\n"
    '                        "[radiance] native MXFP4 enabled on gfx12x "\n'
    '                        "(aiter gemm_afp4wfp4 via tl.dot_scaled); the emulation notice "\n'
    '                        "elsewhere in the log does not apply to mxfp4 x mxfp4 layers"\n'
    "                    )\n"
    "        if not _radiance_mx:\n"
    '            return False, "current platform does not support native MXFP4 computation"\n'
)


# --- 3. aiter 0.1.17 moved the fp4 GEMM module, and gates fp4 on its own arch allowlist -----
IMPORT_ANCHOR = (
    "        from aiter.ops.triton.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffled_weight_scales,\n"
    "        )\n"
)
IMPORT_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): aiter 0.1.17 moved this module ---\n"
    "        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffled_weight_scales,\n"
    "        )\n"
    "\n"
    "        # aiter allowlists gfx950/gfx1250 for fp4; gfx1201 lowers tl.dot_scaled correctly\n"
    "        # (verified bit-identical against the emulated path), so relax the assert. Done\n"
    "        # here, lazily, to keep aiter out of the plugin-load import graph.\n"
    "        import aiter.ops.triton.utils._triton.arch_info as _radiance_arch\n"
    "\n"
    "        if not _radiance_arch.is_fp4_avail():\n"
    "            _radiance_arch.is_fp4_avail = lambda: True\n"
)


def main():
    apply(KL, REGISTER_ANCHOR, REGISTER_NEW,
          "[radiance] MXFP4 W4A8 kernel unavailable",
          "mxfp4: register the radiance W4A8 kernel")
    apply(KA, SUPPORTS_ANCHOR, SUPPORTS_NEW,
          "[radiance] native MXFP4 enabled on gfx12x",
          "mxfp4: relax aiter's CDNA4 gate")
    apply(KA, IMPORT_ANCHOR, IMPORT_NEW,
          "aiter 0.1.17 moved this module",
          "mxfp4: aiter moved-module + fp4 arch allowlist")


if __name__ == "__main__":
    main()
