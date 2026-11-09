#!/usr/bin/env python3
"""GPU-free wiring and layer-range checks for the KV restore overlay."""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def dflash_names(names: list[str], target_layers: int, draft_layers: int) -> set[str]:
    end_layer = target_layers + draft_layers
    selected: set[str] = set()
    for name in names:
        indices = [int(part) for part in name.split(".") if part.isdigit()]
        if len(indices) == 1 and target_layers <= indices[0] < end_layer:
            selected.add(name)
    return selected


def check_layer_range() -> None:
    names = [
        "model.language_model.layers.0.linear_attn",
        "model.language_model.layers.63.self_attn.attn",
        "model.layers.64.self_attn.attn",
        "model.layers.65.self_attn.attn",
        "model.layers.68.self_attn.attn",
        "model.layers.69.self_attn.attn",
        "model.layers.64.sub.1.attn",
    ]
    assert dflash_names(names, 64, 5) == {
        "model.layers.64.self_attn.attn",
        "model.layers.65.self_attn.attn",
        "model.layers.68.self_attn.attn",
    }


def check_mtp_namespace() -> None:
    names = {
        "model.language_model.layers.63.self_attn.attn",
        "mtp.layers.0.self_attn.attn",
        "draft.mtp.layers.1.self_attn.attn",
        "model.layers.0.linear_attn",
    }
    assert {
        name
        for name in names
        if "mtp" in name.split(".") and "layers" in name.split(".")
    } == {
        "mtp.layers.0.self_attn.attn",
        "draft.mtp.layers.1.self_attn.attn",
    }


def check_wiring() -> None:
    patch = (REPO / "patch_kv_offload_restore.py").read_text()
    ast.parse(patch)
    for sentinel in (
        "if spec.is_draft",
        "if attn_module.is_draft_layer:",
        "group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)",
        "spec_config.use_eagle_block_drop()",
        "Skipping CPU hit for request %s since some of its",
    ):
        assert sentinel in patch

    lifecycle = (REPO / "patch_kv_offload_lifecycle.py").read_text()
    ast.parse(lifecycle)
    for sentinel in (
        "4c58a0c398b056b135b98bd93c644945be7e3109",
        "barrier: Callable[[], None] | None = None",
        "def _all_workers_barrier() -> None:",
        "barrier=_all_workers_barrier",
        "Unlinked mmap file %s",
    ):
        assert sentinel in lifecycle

    for name in ("Dockerfile", "Dockerfile.dev", "Dockerfile.patch"):
        dockerfile = (REPO / name).read_text()
        assert "patch_kv_offload_lifecycle patch_kv_offload_restore" in dockerfile, name

    gate = REPO / "benchmarks/bin/run_kv_offload_restore_gate.py"
    ast.parse(gate.read_text())
    assert "reset_external" in gate.read_text()


def check_installed_annotation() -> None:
    """Check installed semantics when running in the candidate image."""
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        return  # Pure source CI; candidate-image qualification calls this too.
    from dataclasses import replace
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64,
                             dtype=__import__("torch").float16)
    # v0.33 marks layer roles at registration and annotates groups during global
    # planning. Names/positions (including MTP's local index) are not identities.
    names = ["model.layers.0.attn", "model.layers.64.attn",
             "draft.layers.1.attn", "mtp.layers.0.attn", "model.layers.999.attn"]
    specs = {name: replace(spec, is_draft=index in (2, 3))
             for index, name in enumerate(names)}

    class FixtureBuilder(KVCacheConfigBuilder):
        def get_kv_cache_groups(self, config, layer_specs):
            return [KVCacheGroupSpec([name], value)
                    for name, value in layer_specs.items()]

        def get_kv_cache_config_from_groups(self, *args, **kwargs):
            raise AssertionError("annotation must not allocate cache tensors")

    def flags(method, *, disabled=False, layer_specs=specs):
        # Exercise the real installed speculative policy without constructing a
        # model/device config. These policy methods read only these two fields.
        speculative = None
        if method is not None:
            speculative = object.__new__(SpeculativeConfig)
            speculative.method = method
            speculative.disable_eagle_block_drop = disabled
        config = NS(num_prefill_lookahead_tokens=0,
                    speculative_config=speculative)
        with patch("vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder",
                   return_value=FixtureBuilder()):
            # Real merge, registry, partition and annotation code; only the
            # allocation-independent grouping fixture is supplied by the seam.
            groups = get_global_kv_cache_groups(config, [layer_specs, layer_specs])
        assert [g.layer_names for g in groups] == [[n] for n in names]
        return [g.is_eagle_group for g in groups]

    for method in ("eagle", "eagle3", "mtp"):
        assert flags(method) == [False, False, True, True, False]
        assert not any(flags(method, disabled=True))
        # The retired positional fallback must not mark an arbitrary last layer.
        assert not any(flags(method, layer_specs=dict.fromkeys(names, spec)))
    # DFlash/DSpark KV does not depend on the following token. Upstream owns
    # their no-trailing-block-drop policy, even for correctly marked drafts.
    for method in (None, "dflash", "dspark", "draft_model"):
        assert not any(flags(method))


def main() -> None:
    check_layer_range()
    check_mtp_namespace()
    check_wiring()
    check_installed_annotation()
    print("KV offload restore regression checks: PASS")


if __name__ == "__main__":
    main()
