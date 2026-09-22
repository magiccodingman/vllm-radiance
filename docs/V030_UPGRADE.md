# vLLM 0.30 resident platform migration

Status: **IN PROGRESS — NOT HARDWARE QUALIFIED**.

## Source and scope

This branch starts from main `285ac78e7f19bc1e1b5b09b25f865aea0c6d9754`.
MR !37 remains Draft/open/unmerged at research report
`b5208d32b8c024181a182b7e8fbdf5dff368bcf2`; its freeze note is
[note 2990](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/merge_requests/37#note_2990).
Its behavior tests and hardware history are reference evidence, not the upgrade base.

Target: vLLM 0.30.0, immutable commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
GGZ14 donor frozen at `31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`
from https://github.com/GGZ14/vllm-mxfp4 on 2026-09-22.

Production remains stopped under the owner's standing instruction. This work
uses separate candidate images and isolated resident-model validation. No merge,
tiered-v2 implementation, device-managed pool or publication benchmark is included.

## Release semantics and expected impact

The [v0.30 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
changes Runner V2 graph capture and speculative execution, online/partial
quantization, parser/serving dependencies, and Qwen QSA/PLE execution.
It removes deprecated environment variables, deprecates Mamba cache `all`, and
changes scale-out endpoint configuration. NVIDIA-specific compile removal is
not a ROCm policy. Runner V2 selection and graph mode require independent checks.

Migration areas: runner/sampling, GDN metadata and prefix state, FP8/MXFP4
dispatch, attention, TP collectives, speculative hooks, parser/XGrammar,
CPU KV transfer and graph capture. Every registered overlay must receive an
evidence-backed disposition in the existing maintenance registry.

## Stack comparison

The target's `docker/Dockerfile.rocm_base` pins the same AMD Torch
`6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`, Triton
`f0b55c07da61c71775bef6d1a15ebf846430ac75`, and torchvision 0.27.1
lineages as Radiance. Preserve these unless an actual compatibility issue requires
a change. Upstream uses ROCm 7.2.3 packaging and AITER `v0.1.21.post2`;
Radiance starts from qualified ROCm 7.14 and AITER 0.1.20. AITER API comparison
and native qualification are pending; no whole-stack replacement is assumed.

## Qualification ledger

Build/import, overlay dispositions, Runner V2 path evidence, resident FP8 and
Quark MXFP4 gates, NVFP4 conversion, parser/structured output, vision and bounded
performance sanity are pending. No historical test count is attributed to this
candidate. The existing full source/contract suite remains required; publication
BetterBench and long-context qualification are excluded by this mission.

## Overlay migration matrix

The machine-readable owner is `.radiance/patches.yaml`. Reasons describe the
retained behavior, not a new hardware claim. UPSTREAM_OWNED rows are not applied;
parser/XGrammar backport files were removed after installed behavior tests.
Other inactive historical scripts remain recoverable reference pending their
specific native/speculative gates. Every retained owner remains Radiance; every
upstream-owned implementation belongs to the immutable vLLM pin.

| Overlay | Disposition | Behavior / ownership decision |
| --- | --- | --- |
| `patch_ar_geometry` | RETAIN_UNCHANGED | Expose the measured TP2 compressed all-reduce geometry controls. |
| `patch_ar_maxbytes` | RETAIN_UNCHANGED | Retain exact-message size limits for the qualified Radiance TP2 collective path. |
| `patch_conv1d_blockn` | RETAIN_UNCHANGED | Retain block-N compatibility for the qualified GDN/conv1d path. |
| `patch_dflash2_v0271_backport` | OBSOLETE_DELETE | Historical v0.27.1 DFlash2 source-copy backport; retained only for archaeology because v0.28 owns DFlash2 natively. |
| `patch_dflash_base` | RETAIN_UNCHANGED | Install Radiance DFlash integration hooks on top of vLLM's native DFlash2 implementation. |
| `patch_dflash_fused_kv_fp8` | RETAIN_UNCHANGED | Preserve fused FP8 KV handling required by the qualified DFlash lane. |
| `patch_dflash_logits_cache_stride` | UPSTREAM_OWNED | Backport the post-v0.28 DFlash logits-cache stride/width correctness fix. |
| `patch_dflash_selector_topk` | RETAIN_UNCHANGED | Expose the DFlash selector top-k experimental control while preserving the checkpoint default. |
| `patch_dflash_w4` | RETAIN_UNCHANGED | Integrate Radiance W4 draft-linears for the opt-in fast DFlash path. |
| `patch_dynamo_metrics` | RETAIN_UNCHANGED | Expose/retain metrics required to diagnose compiled serving behavior. |
| `patch_dynwidth` | MECHANICAL_PORT | Apply per-request DFlash verification-width policy from observed acceptance. |
| `patch_fp8_kv_sidecar` | RETAIN_UNCHANGED | Support immutable checkpoint-bound FP8 KV calibration sidecars as an opt-in fidelity laboratory. |
| `patch_from_json_filter` | RETAIN_UNCHANGED | Retain JSON filtering compatibility used by the serving/tooling path. |
| `patch_gdn_aiter_prefill` | RETAIN_UNCHANGED | Provide the explicit AITER GDN prefill bridge retained as a measured opt-in path. |
| `patch_gdn_merge_inproj` | MECHANICAL_PORT | Merge compatible GDN input projections at load time to reduce launch overhead. |
| `patch_gdn_metadata` | MECHANICAL_PORT | Provide exact GPU GDN metadata used by speculative execution. |
| `patch_gdn_shared_build` | RETAIN_UNCHANGED | Share equivalent speculative GDN metadata construction across KV groups. |
| `patch_gdn_wmma` | RETAIN_UNCHANGED | Apply the gfx1201 WMMA triangular-solve path used by qualified GDN execution. |
| `patch_gfx1201` | RETAIN_UNCHANGED | Enable/guard gfx1201-specific vLLM behavior required by the qualified R9700 stack. |
| `patch_kv_group_size` | RETAIN_UNCHANGED | Select hybrid KV group size by measured capacity cost rather than smallest-bucket padding. |
| `patch_kv_offload_lifecycle` | MECHANICAL_PORT | Verify upstream mmap lifetime; retain only the independent rank-layout extension. |
| `patch_kv_offload_rank_sharded` | SEMANTIC_PORT | Add the opt-in rank-major CPU KV layout; invoked transitively by the lifecycle overlay. |
| `patch_kv_offload_registration` | SEMANTIC_PORT | Harden mmap host registration, HIP error draining, and coherent TP fallback. |
| `patch_kv_offload_restore` | SEMANTIC_PORT | Repair native CPU KV restore/group handling for qualified hybrid/speculative layouts. |
| `patch_mtp_loopbreak` | RETAIN_UNCHANGED | Preserve Radiance MTP loop termination/controller behavior. |
| `patch_mtp_mm_mask` | RETAIN_UNCHANGED | Preserve MTP matrix/mask correctness behavior used by the qualified fast-draft path. |
| `patch_parser_shared_engine` | UPSTREAM_OWNED | Backport the v0.28 shared-parser fix that preserves registered reasoning/tool adapters and structural tags. |
| `patch_preshuffle` | RETAIN_UNCHANGED | Install Radiance preshuffled FP8 weight handling and dispatcher integration. |
| `patch_quark_bf16_mtp` | RETAIN_UNCHANGED | Allow the qualified Quark checkpoint's BF16 MTP tensors only when explicitly selected. |
| `patch_quark_mxfp4` | MECHANICAL_PORT | Register and route native Quark MXFP4/W4A8 on gfx1201. |
| `patch_qwen3_toolparse` | RETAIN_UNCHANGED | Preserve Qwen3 tool/reasoning parser behavior required by Radiance tool-call qualification. |
| `patch_qwen_open_object_schema` | RETAIN_UNCHANGED | Preserve open nested-object schemas used by deferred-tool wrappers instead of flattening/restricting arguments. |
| `patch_r4d` | MECHANICAL_PORT | Integrate libr4d operators with the pinned vLLM model/runtime paths. |
| `patch_radiance_dispatch` | RETAIN_UNCHANGED | Install Radiance FP8 dispatch selection and guarded fallbacks. |
| `patch_rocm_cudagraph_current_stream` | UPSTREAM_OWNED | Backport vLLM's post-v0.28 ROCm graph capture fix to use the current stream. |
| `patch_skinny_gemm` | RETAIN_UNCHANGED | Route qualified skinny GEMM shapes through Radiance-tuned implementations. |
| `patch_topk_composite` | RETAIN_UNCHANGED | Install the bounded composite top-k path with exact fallback. |
| `patch_topk_triton_rows` | UPSTREAM_OWNED | Route small-row top-k work through the qualified Triton path. |
| `patch_unified_attention_lds` | RETAIN_UNCHANGED | Retain the qualified broader ROCm LDS fit/tuning guard for unified attention. |
| `patch_unpad` | RETAIN_UNCHANGED | Carry Radiance's guarded unpadding compatibility behavior. |
| `patch_verify_head` | MECHANICAL_PORT | Install the sampling-aware target verification head with exact fallback. |
| `patch_xgrammar_spec_reasoning` | UPSTREAM_OWNED | Backport reasoning-boundary validation for drafts generated before the grammar mask activates. |
| `patch_xgrammar_spec_termination` | UPSTREAM_OWNED | Backport speculative grammar termination handling so draft batches cannot overrun a terminated FSM. |
| `install_radiance_hooks` | RETAIN_UNCHANGED | Install Radiance runtime hooks into the pinned vLLM tree. |
| `patch_nvfp4_mxfp4` | SEMANTIC_PORT | Bounded NVFP4-only load-time conversion with per-partition scale provenance. |

## Integration details

- V2 post-load GDN merge follows the actual JIT-registry loader context.
- R4D GDN hook follows the actual per-owner metadata lookup, not a guessed ABI.
- Verification-head hook stays in the ordinary logits path; batch-sharded logits
  are upstream-owned and not claimed optimized by this hook.
- v0.30 structured-output validation runs before Radiance width capping.
- CPU KV uses upstream chunk terminology; rank layout rejects creator-only
  population rather than mixing incompatible ownership.
- AITER 0.1.20's preshuffled MXFP4 function is aliased to v0.30's renamed import.
  The explicitly requested W4A8 backend cannot silently become emulation.
- `RADIANCE_TOPK_TRITON_MIN_ROWS` is deprecated; nondefault values fail startup.
- NVFP4 adaptation and exact donor decisions are in
  [the conversion record](V030_GGZ14_MXFP4_NVFP4.md).

## Preliminary development evidence (not final image qualification)

Source audit applied all retained overlays to a clean v0.30 tree and parsed it.
Dev image `sha256:b1b88b18557991700a2132a2b7336663423a95dcf2d587c81babccd0c53258f8`
built/imported with pip consistency. Complete PR source plan passed after fixing
read-only bytecode placement and replacing obsolete string assertions with
installed semantic tests. NVFP4 CPU tests: 8 passed.

FP8 27B loaded with actual `vllm.v1.worker.gpu.model_runner.GPUModelRunner`,
`FULL_AND_PIECEWISE`, and retained AMD stack. Fresh/repeated/independent32-token
outputs and three exact-ID recomputations passed; nested JSON, tool call and
repeated red-image interpretation passed. Short unprofiled decode measured
30.106 / 36.968 / 37.025 tok/s (first request includes cold runtime effects).
This is NOT a matched performance comparison. The five-token prompt did not
exercise a complete hybrid prefix block; final coverage must include1600tokens.
An eight-token separate profiler observation confirmed real R4D attention/GDN,
exact small-message AR and Radiance FP8 preshuffle kernel activity on both ranks.
Profiler emitted duplicate-flow warnings; its times are not performance scores.

Artifacts: `/nvme/ediloca-1/scratch/v030-fp8-port01` (development candidate).
Production remained stopped. No historical MR !37 result is counted here.
