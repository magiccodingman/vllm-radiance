#!/usr/bin/env python3
"""Bracket the DFlash drafter's get_model so radiance_w4 can tell a drafter linear from a target one.

radiance_w4 packs block-fp8 weights to 4 bits from inside process_weights_after_loading, which is
the only point where the fp8 weight exists and nothing has shuffled it yet. That callback gets a
layer and no idea which model it belongs to, and the drafter's prefixes are indistinguishable from
the target's -- both are `model.layers.N.mlp.down_proj`. The drafter is loaded by exactly one call,
so bracketing that call is the whole discriminator, and it needs no heuristic that could one day
claim a target layer of the same shape.

The bracket is a try/finally: a load that raises must not leave the flag set, or the next model
built in this process would be packed as if it were the drafter.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
UTIL = SP / "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"

OLD = """    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )
"""
NEW = """    with set_model_tag("dflash_head"):
        # --- RADIANCE 4-bit drafter weights (patch_dflash_w4.py) ---
        try:
            import radiance_w4 as _radiance_w4
        except Exception:
            _radiance_w4 = None
        if _radiance_w4 is not None:
            _radiance_w4.begin_draft()
        try:
            dflash_model = get_model(
                vllm_config=draft_vllm_config, model_config=draft_model_config
            )
        finally:
            if _radiance_w4 is not None:
                _radiance_w4.end_draft()
"""


def main():
    apply(UTIL, OLD, NEW, "RADIANCE 4-bit drafter weights (patch_dflash_w4.py)",
          "mark the drafter's weight load for radiance_w4")


if __name__ == "__main__":
    main()
