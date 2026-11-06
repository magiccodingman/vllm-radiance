#!/usr/bin/env python3
"""CPU regressions for the v0.33 draft mapper, LDS guard and plugin discovery.

Uses guarded source transforms in temporary fixtures. Installed-image and native
checks are separate gates; these tests never import Torch or enumerate GPUs.
"""
import ast
import copy
import importlib.metadata
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import install_radiance_hooks
import patch_dflash_quant_exclusions as aliases
import patch_unified_attention_lds as lds


def draft_mapper(body):
    scope = {"re": re}
    exec("def map_config(quant_config, prefix, start_layer_id, num_hidden_layers):\n"
         + body, scope)
    return scope["map_config"]


def main():
    # Negative control: stable upstream ignores FP8's ignored_layers entirely.
    fixture = ["layers.0.attention_conv.kernel_projection", "layers.64.already_global",
               "lilicorr.layers.0.aux", "*layers.1.*", "model.layers.2.proj"]
    old = SimpleNamespace(ignored_layers=fixture.copy())
    draft_mapper(aliases.OLD)(old, "model", 64, 5)
    assert old.ignored_layers == fixture
    fixed = SimpleNamespace(ignored_layers=fixture.copy(), exclude_modules=["layers.3.proj"])
    draft_mapper(aliases.NEW)(fixed, "model", 64, 5)
    assert "model.layers.64.attention_conv.kernel_projection" in fixed.ignored_layers
    assert "layers.65.*" not in fixed.ignored_layers  # preserve wildcard prefix
    assert "*layers.65.*" in fixed.ignored_layers
    assert "model.layers.64.already_global" not in fixed.ignored_layers
    assert "lilicorr.layers.64.aux" not in fixed.ignored_layers
    assert "model.layers.67.proj" in fixed.exclude_modules
    assert fixed.ignored_layers.count("layers.64.already_global") == 1
    # Existing rooted exclusions must not acquire a duplicate model prefix.
    assert not any(x.startswith("model.model.") for x in fixed.ignored_layers)

    # Execute the real replacement body with an intentionally oversized source
    # config. Check both paths, asymmetric heads, and non-RDNA isolation.
    source_config = {"TILE_SIZE": 128, "num_stages": 4, "BLOCK_M": 16}
    ns = {"copy": copy, "compute_segment_params": lambda c, p: c,
          "triton": SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
          "arch_info": SimpleNamespace(get_arch=lambda: "gfx1201")}
    exec("def config_for(params, arch, backend, op):\n    config = source_config\n" + lds.NEW,
         dict(ns, source_config=source_config), ns)
    get = ns["config_for"]
    for op in ("attn_2d", "attn_3d"):
        for arch in ("gfx1200", "gfx1201"):
            for q_width, v_width in ((128, 256), (256, 128), (512, 512)):
                p = SimpleNamespace(head_size=q_width, head_size_v=v_width,
                                    kv_cache_dtype=SimpleNamespace(itemsize=2))
                cfg = get(p, arch, "triton", op)
                width = 1 << (max(q_width, v_width) - 1).bit_length()
                assert cfg["TILE_SIZE"] * width * 2 * cfg["num_stages"] + 256 <= 65536
                assert source_config["TILE_SIZE"] == 128
    assert get(p, "gfx950", "triton", "attn_3d") == source_config
    assert get(p, "gfx1201", "gluon", "attn_3d") == source_config
    impossible = SimpleNamespace(head_size=4096, head_size_v=4096,
                                 kv_cache_dtype=SimpleNamespace(itemsize=2))
    try:
        get(impossible, "gfx1201", "triton", "attn_2d")
        raise AssertionError("oversized LDS was accepted")
    except ValueError:
        pass

    # Discover the real metadata entrypoint twice without changing installed
    # vLLM's loader or importing the GPU-dependent hook target.
    with tempfile.TemporaryDirectory() as directory:
        original = install_radiance_hooks.sysconfig.get_paths
        try:
            install_radiance_hooks.sysconfig.get_paths = lambda: {"purelib": directory}
            install_radiance_hooks.main()
            install_radiance_hooks.main()
        finally:
            install_radiance_hooks.sysconfig.get_paths = original
        distributions = list(importlib.metadata.distributions(path=[directory]))
        assert len(distributions) == 1
        entrypoints = [e for d in distributions for e in d.entry_points]
        assert [(e.group, e.name, e.value) for e in entrypoints] == [
            ("vllm.general_plugins", "radiance", "radiance_kernels:install_all")]
    # A newer upstream fused speculative layout must retain ownership before
    # any weight merge/release or optional Torch import can occur.
    gdn_tree = ast.parse((ROOT / "radiance_gdnmerge.py").read_text())
    merge = next(n for n in gdn_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_merge_one")
    scope = {}
    exec(compile(ast.Module(body=[merge], type_ignores=[]), "gdn-merge-guard", "exec"), scope)
    assert scope["_merge_one"](SimpleNamespace(enable_fused_gdn_spec_decode=True)) is False
    print("v0.33 draft exclusion, LDS and native plugin CPU regressions: PASS")


if __name__ == "__main__":
    main()
