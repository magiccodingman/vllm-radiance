#!/usr/bin/env python3
"""Install the radiance (gfx1201) custom kernel dispatcher. Run once before serving.

Two idempotent source patches:
  1. Fix AITER's compute_splitk_params: shrink NUM_KSPLIT until each K-split is 128-aligned, so the
     block-FP8 scale blocks stay aligned (K=8704 was picking KSPLIT=8 and page-faulting).
  2. Point vLLM's block-FP8 GEMM chokepoint at the dispatcher:
     TritonFp8BlockScaledMMKernel.apply_block_scaled_mm -> radiance_kernels.block_scaled_mm.

The dispatch table itself (radiance_kernels.py) is placed into site-packages separately (the
Dockerfile COPYs it there). After this, routing changes are edits to radiance_kernels.py, no
vLLM re-patch.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
CFG = SP / "aiter/ops/triton/utils/gemm_config_utils.py"
KERN = SP / "vllm/model_executor/kernels/linear/scaled_mm/triton.py"

CFG_ANCHOR = (
    "    add_default_gemm_config_params(config)\n"
    "\n"
    '    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])\n'
    "\n"
    '    if "BLOCK_SIZE_K" in config:'
)
CFG_NEW = (
    "    add_default_gemm_config_params(config)\n"
    "\n"
    '    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])\n'
    "\n"
    "    # --- radiance fix (patch_radiance_dispatch.py): scale-alignment guard ---\n"
    "    # AITER's gfx1201 config picks NUM_KSPLIT=8 at K=8704 -> SPLITK_BLOCK_SIZE=1088,\n"
    "    # not a multiple of the 128 block-FP8 scale block -> GPU page fault. Shrink\n"
    "    # NUM_KSPLIT until each K-split is 128-aligned (drops 8->4 for K=8704).\n"
    '    while config["NUM_KSPLIT"] > 1 and (K % (128 * config["NUM_KSPLIT"]) != 0):\n'
    '        config["NUM_KSPLIT"] = config["NUM_KSPLIT"] // 2\n'
    '    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])\n'
    "\n"
    '    if "BLOCK_SIZE_K" in config:'
)

KERN_ANCHOR = (
    "    ) -> torch.Tensor:\n"
    "        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(\n"
    "            A,\n"
    "            B,\n"
    "            As,\n"
    "            Bs,\n"
    "            list(self.weight_group_shape),\n"
    "            self.config.out_dtype,\n"
    "        )"
)
KERN_NEW = (
    "    ) -> torch.Tensor:\n"
    "        # --- gfx1201 custom kernel dispatcher (radiance; patch_radiance_dispatch.py) ---\n"
    "        from radiance_kernels import block_scaled_mm as _radiance_block_scaled_mm\n"
    "        return _radiance_block_scaled_mm(self, A, B, As, Bs)"
)


def main():
    apply(CFG, CFG_ANCHOR, CFG_NEW, "radiance fix (patch_radiance_dispatch.py)", "compute_splitk_params scale-alignment fix")
    apply(KERN, KERN_ANCHOR, KERN_NEW, "gfx1201 custom kernel dispatcher", "apply_block_scaled_mm -> dispatcher hook")


if __name__ == "__main__":
    main()
