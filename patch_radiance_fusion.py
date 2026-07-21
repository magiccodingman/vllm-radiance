#!/usr/bin/env python3
"""Install the RADIANCE rms_norm + group-fp8-quant fusion coverage fix (gfx1201).

vLLM's RocmAiterRMSNormQuantFusionPass registers the *group* rms+quant patterns
with only the aiter-quant matcher (match_aiter_quant=True). On gfx1201 our graph
emits torch.ops._C.per_token_group_fp8_quant (the NATIVE quant op), so the aiter
matcher matches 0 patterns and the standalone bf16->fp8 group-quant kernels are
never folded into the rms epilogue. This registers the native (match_aiter_quant
=False) variant too, as vLLM already does for the sibling per-token DYNAMIC quant
patterns below, reusing the SAME is_quant_fp8_enabled duplicate-pattern guard
(with quant_fp8 disabled both matchers trace the same native impl, so registering
both would create duplicate Inductor patterns).

Runtime-gated by RADIANCE_FUSE_RMS_QUANT (default "1"; "0" for stock behaviour).
Numerically exact: match_aiter_quant only selects WHICH quant op is matched; the
key.quant (per-128-group, symmetric, fp32 scale, fp8) and the fused replacement op
(get_rmsnorm_group_fused_quant_op) are identical either way. Idempotent. Run once
pre-serve / at image build."""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/compilation/passes/fusion/rocm_aiter_fusion.py"
)

ANCHOR = (
    "            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant\n"
    "            AiterRMSFp8GroupQuantPattern(\n"
    "                epsilon, FP8_DTYPE, GroupShape(1, 128)\n"
    "            ).register(self.patterns)\n"
    "\n"
    "            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant\n"
    "            AiterFusedAddRMSFp8GroupQuantPattern(\n"
    "                epsilon, FP8_DTYPE, GroupShape(1, 128)\n"
    "            ).register(self.patterns)"
)

NEW = (
    "            # --- RADIANCE: rms_norm(+fused_add) + group fp8 quant coverage fix ---\n"
    "            # Stock registers only the aiter-quant matcher; on gfx1201 our graph\n"
    "            # emits torch.ops._C.per_token_group_fp8_quant (native), so that matches\n"
    "            # 0. Register the native (match_aiter_quant=False) variant too, reusing\n"
    "            # the same is_quant_fp8_enabled duplicate-pattern guard vLLM applies to\n"
    "            # the dynamic-quant patterns below. Gated by RADIANCE_FUSE_RMS_QUANT.\n"
    "            import os as _radiance_os\n"
    '            if _radiance_os.environ.get("RADIANCE_FUSE_RMS_QUANT", "1") == "1":\n'
    "                _radiance_maq = (\n"
    "                    [True, False]\n"
    '                    if config.compilation_config.is_custom_op_enabled("quant_fp8")\n'
    "                    else [False]\n"
    "                )\n"
    "            else:\n"
    "                _radiance_maq = [True]  # stock behaviour\n"
    "            for _maq in _radiance_maq:\n"
    "                AiterRMSFp8GroupQuantPattern(\n"
    "                    epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant=_maq\n"
    "                ).register(self.patterns)\n"
    "                AiterFusedAddRMSFp8GroupQuantPattern(\n"
    "                    epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant=_maq\n"
    "                ).register(self.patterns)"
)

MARKER = "RADIANCE: rms_norm(+fused_add) + group fp8 quant"


def main():
    apply(F, ANCHOR, NEW, MARKER, "rms+quant fusion coverage fix")


if __name__ == "__main__":
    main()
