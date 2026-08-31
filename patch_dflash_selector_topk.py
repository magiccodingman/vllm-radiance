#!/usr/bin/env python3
"""Make the DFlash2 candidate selector's top_k env-tunable.

The int2 rerank work established that RADIANCE_DRAFT_RERANK's pool saturates at 4x the
selector's top_k -- "this is a ceiling to raise with selector_top_k, not a free parameter."
The checkpoint ships selector_top_k=16 as a config value, but the selector is a trained
SCORING head (rank-256); top_k only truncates its ranked output at inference, so raising it
admits more candidates without touching weights. RADIANCE_DFLASH_SELECTOR_TOPK overrides
(0/unset = checkpoint value). Pair it with RADIANCE_DRAFT_RERANK >= 4x the new k.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

apply(
    SP / "vllm" / "model_executor" / "models" / "qwen3_dflash2.py",
    '                top_k=int(draft_config["selector_top_k"]),\n',
    '                # radiance (patch_dflash_selector_topk.py): inference-time override; the\n'
    '                # selector is a trained scorer, top_k only truncates its ranking.\n'
    '                top_k=(int(__import__("os").environ.get("RADIANCE_DFLASH_SELECTOR_TOPK") or 0)\n'
    '                       or int(draft_config["selector_top_k"])),\n',
    "RADIANCE_DFLASH_SELECTOR_TOPK",
    "dflash selector top_k override",
)

# The worker-side speculator caches its OWN copy of the value (used for the candidate reshape
# and the Triton walk kernels' constexpr); both readers must agree or the reshape throws
# "shape [B, S, 16] invalid" the moment the model returns 32 candidates.
apply(
    SP / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dflash2" / "speculator.py",
    '        self.selector_top_k = int(draft_config["selector_top_k"])\n',
    '        # radiance (patch_dflash_selector_topk.py): must match the model-side override.\n'
    '        self.selector_top_k = (\n'
    '            int(__import__("os").environ.get("RADIANCE_DFLASH_SELECTOR_TOPK") or 0)\n'
    '            or int(draft_config["selector_top_k"])\n'
    '        )\n',
    "RADIANCE_DFLASH_SELECTOR_TOPK",
    "dflash selector top_k override (speculator)",
)
