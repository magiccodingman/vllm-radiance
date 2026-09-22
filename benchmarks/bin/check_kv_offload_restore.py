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
        "def _dflash_draft_layer_names(",
        "def _mtp_draft_layer_names(",
        "draft_names.intersection(group.layer_names)",
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
    from types import SimpleNamespace as NS
    from vllm.v1.core.kv_cache_utils import _annotate_eagle_groups
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64,
                             dtype=__import__("torch").float16)
    names = ["model.layers.0.attn", "model.layers.64.attn", "model.layers.65.attn"]
    config = NS(parallel_config=None, model_config=NS(get_num_layers=lambda _: 64),
                speculative_config=NS(use_eagle_block_drop=lambda: True,
                    use_eagle=lambda: True, use_dflash=lambda: True,
                    draft_model_config=NS(get_num_layers=lambda _: 2)))
    groups = [NS(layer_names=[n], kv_cache_spec=spec, is_eagle_group=False) for n in names]
    _annotate_eagle_groups(config, dict.fromkeys(names, spec), groups)
    assert [g.is_eagle_group for g in groups] == [False, True, True]
    config.speculative_config.use_eagle_block_drop = lambda: False
    for g in groups:
        g.is_eagle_group = False
    _annotate_eagle_groups(config, dict.fromkeys(names, spec), groups)
    assert not any(g.is_eagle_group for g in groups)
    config.speculative_config.use_eagle_block_drop = lambda: True
    config.speculative_config.use_dflash = lambda: False
    _annotate_eagle_groups(config, dict.fromkeys(names, spec), groups,
                           use_deepseek_v4_fallback=True)
    assert [g.is_eagle_group for g in groups] == [False, False, True]


def main() -> None:
    check_layer_range()
    check_mtp_namespace()
    check_wiring()
    check_installed_annotation()
    print("KV offload restore regression checks: PASS")


if __name__ == "__main__":
    main()
