#!/usr/bin/env python3
"""Restore native CPU KV hits for Qwen hybrid DFlash deployments.

The pinned vLLM v0.28.0 scheduler treats every cache group as an EAGLE draft
group when no group is explicitly annotated. On hybrid GDN/Mamba models that
marks recurrent target groups as volatile; their widened lookup window cannot
be satisfied, so GPU prefix caching and the native CPU tier store data but
never serve it.

This overlay combines upstream vLLM #52047's general hybrid-path annotation
with explicit Qwen DFlash/DFlash2 and MTP rules. The DFlash loader appends draft
attention layers at ``target_num_layers`` while embedded MTP uses an
``mtp.layers`` namespace; only groups containing those layers receive the
volatile-tail treatment. The existing fallback remains unchanged for
unsupported EAGLE-family architectures, preserving its safety behavior instead
of silently disabling the tail guard.

It also applies the bounded scheduling correction proposed in upstream
#44295: a request sharing keys with an in-flight CPU load recomputes instead of
waiting behind a serialized load convoy. Duplicate CPU loads remain blocked.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])


def patch_cache_group_annotation() -> None:
    path = LIB / "vllm/v1/core/kv_cache_utils.py"
    apply(
        path,
        '''def _annotate_eagle_groups_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    # Detection uses the merged MLA spec's model_version.
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return
    # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
    # group contains it.
    # FIXME(yifan): avoid/generalize this hacky check.
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break
''',
        '''def _dflash_draft_layer_names(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> set[str]:
    """Identify Qwen DFlash layers appended after the target layer range."""
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_dflash():
        return set()
    draft_model_config = spec_config.draft_model_config
    if draft_model_config is None:
        return set()

    # This deliberately mirrors DFlashQwen3ForCausalLM.__init__, which passes
    # target_model.get_num_layers(...) as start_layer_id to the draft model.
    target_layers = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )
    draft_layers = draft_model_config.get_num_layers(vllm_config.parallel_config)
    end_layer = target_layers + draft_layers

    draft_names: set[str] = set()
    for name in kv_cache_spec:
        indices = [int(part) for part in name.split(".") if part.isdigit()]
        if len(indices) == 1 and target_layers <= indices[0] < end_layer:
            draft_names.add(name)
    return draft_names


def _mtp_draft_layer_names(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> set[str]:
    """Identify embedded Qwen MTP attention layers by module namespace."""
    spec_config = vllm_config.speculative_config
    if (
        spec_config is None
        or not spec_config.use_eagle()
        or spec_config.use_dflash()
    ):
        return set()
    return {
        name
        for name in kv_cache_spec
        if "mtp" in name.split(".") and "layers" in name.split(".")
    }


def _annotate_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    use_deepseek_v4_fallback: bool = False,
) -> None:
    """Flag only groups that contain volatile draft-attention state."""
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return

    # Qwen DFlash/DFlash2 has ordinary AttentionSpec objects, so the appended
    # layer range is the only scheduler-visible distinction from target KV.
    draft_layer_names = _dflash_draft_layer_names(vllm_config, kv_cache_spec)
    draft_layer_names.update(_mtp_draft_layer_names(vllm_config, kv_cache_spec))
    for group in kv_cache_groups:
        # Upstream #52047 marker for DSpark-style MLA draft groups.
        spec_marked = getattr(
            group.kv_cache_spec, "non_causal_multi_token_decode", False
        )
        if spec_marked or draft_layer_names.intersection(group.layer_names):
            group.is_eagle_group = True

    if not use_deepseek_v4_fallback:
        return
    # DeepSeek-V4's MTP block has no spec marker and is registered last.
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break
''',
        "def _dflash_draft_layer_names(",
        "kv-offload restore: identify draft cache groups",
    )
    apply(
        path,
        '''        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups
''',
        '''        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_deepseek_v4_fallback=True,
        )
        return kv_cache_groups
''',
        "use_deepseek_v4_fallback=True",
        "kv-offload restore: retain DeepSeek annotation",
    )
    apply(
        path,
        '''            groups.append(KVCacheGroupSpec([name], aligned))

    return groups
''',
        '''            groups.append(KVCacheGroupSpec([name], aligned))

    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)
    return groups
''',
        "_annotate_eagle_groups(vllm_config, kv_cache_spec, groups)",
        "kv-offload restore: annotate hybrid general path",
    )


def patch_shared_prefix_convoy() -> None:
    path = (
        LIB
        / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    apply(
        path,
        '''                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None
''',
        '''                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Preserve the no-duplicate-load guard without serializing
                    # every request sharing this prefix behind one transfer.
                    # This request recomputes while the first load warms APC.
                    logger.debug(
                        "Skipping CPU hit for request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return 0
''',
        "Skipping CPU hit for request %s since some of its",
        "kv-offload restore: avoid shared-prefix load convoy",
    )


patch_cache_group_annotation()
patch_shared_prefix_convoy()
