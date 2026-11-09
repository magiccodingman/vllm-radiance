#!/usr/bin/env python3
"""Keep both RDNA4 attention paths inside 64 KiB after AITER's config-API migration.

AITER 0.1.26 moved the selectors into a shared JSON configuration reader.
Apply the correctness clamp there, independently of the optional runtime tune.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "aiter/ops/triton/utils/unified_attention_utils.py"
OLD = "    return compute_segment_params(copy.deepcopy(config), params)\n"
NEW = '''    config = compute_segment_params(copy.deepcopy(config), params)
    # RADIANCE LDS fit: the configuration reader serves both attention paths.
    if ((arch or arch_info.get_arch()) in ("gfx1200", "gfx1201")
            and backend == "triton" and op in ("attn_2d", "attn_3d")):
        tile = config["TILE_SIZE"]
        stages = config.get("num_stages", 1)
        width = triton.next_power_of_2(max(params.head_size, params.head_size_v))
        element_bytes = params.kv_cache_dtype.itemsize
        while stages > 1 and tile * width * element_bytes * stages + 256 > 65536:
            stages -= 1
        while tile > 16 and tile * width * element_bytes * stages + 256 > 65536:
            tile //= 2
        if tile * width * element_bytes * stages + 256 > 65536:
            raise ValueError("RADIANCE: attention configuration exceeds RDNA4 LDS")
        config["TILE_SIZE"] = tile
        config["num_stages"] = stages
    return config
'''


def main():
    apply(F, OLD, NEW, "# RADIANCE LDS fit: the configuration reader", "RDNA4 shared LDS clamp")


if __name__ == "__main__":
    main()
