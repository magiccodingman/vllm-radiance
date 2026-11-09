#!/usr/bin/env python3
"""Backport FP8 draft exclusion aliases to the stable v0.33 config mapper.

Selective adaptation of vLLM PR #61010 by Yufeng He, commit
c9cf9d5e77151bf47116ba908df748908b5112fd. ModelOpt mapping is unchanged.
Remove once the stable upstream mapper owns both exclusion lists and rooting.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/utils.py"

OLD = '''    exclusions = getattr(quant_config, "exclude_modules", None)
    if not isinstance(exclusions, list):
        return

    def offset_local_layer(match: re.Match[str]) -> str:
        layer_idx = int(match.group(1))
        if layer_idx >= num_hidden_layers:
            return match.group(0)
        return f"layers.{layer_idx + start_layer_id}"

    for exclusion in tuple(exclusions):
        # Accept layers.N or *layers.N, excluding nested names such as
        # lilicorr.layers.N, which are not decoder-layer indices.
        global_exclusion = re.sub(
            r"(?<![A-Za-z0-9_.])layers\\.(\\d+)", offset_local_layer, exclusion
        )
        if global_exclusion != exclusion and global_exclusion not in exclusions:
            exclusions.append(global_exclusion)
'''
NEW = '''    def offset_local_layer(match: re.Match[str]) -> str:
        layer_idx = int(match.group(1))
        if layer_idx >= num_hidden_layers:
            return match.group(0)
        return f"layers.{layer_idx + start_layer_id}"

    root = f"{prefix}." if prefix else ""
    # FP8 uses ignored_layers and exact names; other families use exclude_modules.
    # Preserve originals and already-global names, adding only local aliases.
    for attr in ("exclude_modules", "ignored_layers"):
        exclusions = getattr(quant_config, attr, None)
        if not isinstance(exclusions, list):
            continue
        for exclusion in tuple(exclusions):
            global_exclusion = re.sub(
                r"(?<![A-Za-z0-9_.])layers\\.(\\d+)", offset_local_layer, exclusion
            )
            if global_exclusion == exclusion:
                continue
            if global_exclusion not in exclusions:
                exclusions.append(global_exclusion)
            if root and "*" not in exclusion:
                rooted_exclusion = f"{root}{global_exclusion}"
                if rooted_exclusion not in exclusions:
                    exclusions.append(rooted_exclusion)
'''

if __name__ == "__main__":
    apply(F, OLD, NEW, 'for attr in ("exclude_modules", "ignored_layers"):', "DFlash FP8 draft exclusions")
