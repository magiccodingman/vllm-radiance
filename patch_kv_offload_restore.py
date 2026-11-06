#!/usr/bin/env python3
"""Verify upstream draft-cache ownership and retain the CPU-hit convoy correction.

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
    """v0.33 owns layer-role marking and DFlash's no-trailing-block policy."""
    contracts = {
        "vllm/v1/core/kv_cache_planning.py": (
            "spec_config.use_eagle_block_drop()",
            "name for name, spec in merged_kv_cache_specs.items() if spec.is_draft",
            "group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)",
        ),
        "vllm/v1/worker/gpu/attn_utils.py": (
            "if attn_module.is_draft_layer:", "spec = replace(spec, is_draft=True)",
        ),
        "vllm/v1/worker/gpu_model_runner.py": ("is_draft_layer",),
        "vllm/config/speculative.py": (
            "return self.use_eagle() and not (self.use_dflash() or self.use_dspark())",
        ),
    }
    for relative, required in contracts.items():
        source = (LIB / relative).read_text()
        for fragment in required:
            if fragment not in source:
                raise SystemExit(f"Upstream draft-cache identity contract drift: {relative}: {fragment}")
    print("OK upstream draft-cache identities and DFlash prefix safety")


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
