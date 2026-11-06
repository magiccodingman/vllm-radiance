# vLLM 0.33 / maintained RDNA4 upgrade

Status: **migration in progress; not qualified for production**.

## Scope and identities

- Base: main `1d2e24849c9d8441b9614acade5252c9937a2bbf`.
- Target: stable vLLM **0.33.0**, annotated tag `v0.33.0` resolving to
  `4ba787ec9959e566d3b587a796dff07673324da5` (the GitHub release URL retains an rc3 tag).
- No production deployment or release-branch merge is part of this MR.
- The owner authorizes focused CPU and **GPU 0 only** validation, using
  `/mnt/world7/AI/Models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd/`.
  Full performance campaigns and TP2 hardware qualification are deferred.

## Expected impact

Preserve MXFP4 W4A8, the qualified GDN decay correction, exact target-head policy,
DFlash integration, parser/structured-output regressions, native CPU KV offloading,
TP2 collectives and opt-in research boundaries. Audit V1 and V2 callsites separately.

The upstream ROCm 0.33 foundation is **not** the generic CUDA stack: its default
ROCm Dockerfile uses ROCm 10.0, AMD PyTorch 2.13, Triton 3.8 and AITER 0.1.26.
The source-built Radiance foundation must reconcile those pins, the runtime JIT
compiler, configuration layout, and final-image runtime probes together.

## Maintained kernel ownership

StillDeadcode retired the old `vllm-radiance` and `libr4d` repositories in favor of
`https://codeberg.org/StillDeadcode/radiance`. Its current libr4d has an engine/plugin
ABI, so retirement is not evidence that it is a drop-in replacement for our Python
extension. Audit current sources and port justified kernels behind the vLLM-facing
contract; record every retained adapter and any remaining deprecated-base debt.
Do not discard qualified local numerical fixes when moving source ownership.

GGZ14's GitHub and Codeberg lines are distinct audit sources. Current candidates:
startup-memory reduction (Codeberg #60), newer-vLLM plugin integration (#59),
optional host embeddings (#56), and GitHub's native MoE lane (#21). Lazy GDN and
new formats remain experiments until separately qualified.

## Initial overlay source audit

This is a disposable exact-source anchor audit, **not an installed-image or GPU
qualification**. Missing dependency source is listed explicitly. A pass only says
the existing patch applied; semantic ownership still needs review.

| Overlay | Initial anchor result |
|---|---|
| `patch_gfx1201` | applies; semantics pending |
| `patch_radiance_dispatch` | applies; semantics pending |
| `patch_skinny_gemm` | migration / equivalence review required |
| `patch_unified_attention_lds` | migration / equivalence review required |
| `patch_gdn_wmma` | applies; semantics pending |
| `patch_gdn_aiter_prefill` | migration / equivalence review required |
| `patch_preshuffle` | applies; semantics pending |
| `install_radiance_hooks` | applies; semantics pending |
| `patch_mtp_mm_mask` | applies; semantics pending |
| `patch_mtp_loopbreak` | applies; semantics pending |
| `patch_qwen3_toolparse` | applies; semantics pending |
| `patch_from_json_filter` | applies; semantics pending |
| `patch_dynamo_metrics` | migration / equivalence review required |
| `patch_conv1d_blockn` | applies; semantics pending |
| `patch_r4d` | migration / equivalence review required |
| `patch_dflash_fused_kv_fp8` | migration / equivalence review required |
| `patch_dflash_w4` | applies; semantics pending |
| `patch_dflash_selector_topk` | migration / equivalence review required |
| `patch_gdn_metadata` | migration / equivalence review required |
| `patch_gdn_shared_build` | migration / equivalence review required |
| `patch_topk_composite` | migration / equivalence review required |
| `patch_quark_mxfp4` | applies; semantics pending |
| `patch_nvfp4_mxfp4` | applies; semantics pending |
| `patch_quark_bf16_mtp` | applies; semantics pending |
| `patch_ar_maxbytes` | applies; semantics pending |
| `patch_ar_geometry` | applies; semantics pending |
| `patch_kv_group_size` | migration / equivalence review required |
| `patch_gdn_merge_inproj` | migration / equivalence review required |
| `patch_dynwidth` | applies; semantics pending |
| `patch_verify_head` | applies; semantics pending |
| `patch_kv_offload_registration` | migration / equivalence review required |
| `patch_kv_offload_lifecycle` | migration / equivalence review required |
| `patch_kv_offload_restore` | migration / equivalence review required |
| `patch_fp8_kv_sidecar` | applies; semantics pending |
| `patch_qwen_open_object_schema` | migration / equivalence review required |
| `patch_qwen_tool_grammar` | migration / equivalence review required |

## Validation boundary

Retain CPU/source/image failures as evidence. Run focused installed-image contracts,
GPU-0 native GDN/MXFP4 tests and real inference/tool gates when the candidate image
is available. No TP2, maximum capacity, or performance qualification may be inferred
from TP1. Existing production defaults are not promoted by this MR.
