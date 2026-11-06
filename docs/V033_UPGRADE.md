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
GPU-0 native attention/GDN/MXFP4 and LFM3 convolution tests and real inference/tool gates when the candidate image
is available. No TP2, maximum capacity, or performance qualification may be inferred
from TP1. Existing production defaults are not promoted by this MR.

## LFM3 text addition

The owner also requested `Lfm3ForCausalLM` and supplied an FP8 compressed-tensors
checkpoint. The official post-release backport and its separate model/profile
qualification are recorded in [LFM3.md](LFM3.md). A pinned Transformers 5.16.1
CPU check caught and verified the repair for config label/RoPE normalization;
its initial failure is retained. No LFM3 runtime pass is claimed yet.

## Implemented migration (candidate)

- The native `vllm.general_plugins` entrypoint replaces modification of vLLM's
  plugin loader. An explicit `VLLM_PLUGINS` allowlist must include `radiance`.
- AITER's unified-attention selectors moved to a shared configuration reader.
  The mandatory 64-KiB LDS guard now covers both Triton attention paths there.
  Radiance settings use the new API; asymmetric Q/V heads retain upstream
  settings rather than inheriting measurements from a different shape.
- Quark MXFP4 tuning JSON files move to AITER's nested gfx1201 GEMM directory.
  FlyDSL moves to the AITER 0.1.26 requirement, **0.4.0**.
- GDN's V2 metadata reuse is upstream-owned; the redundant shared-build overlay
  is retired. `RADIANCE_GDN_SHARED_BUILD` is removed from image/Compose defaults;
  reuse is now upstream-owned rather than an optional local hook. Persistent index buffers and the explicit RDNA4 `aiter` prefill
  implementation remain guarded. The upstream CDNA FlyDSL architecture gate
  is not widened. Zero-draft state offsets retain upstream semantics.
- GDN projection merging leaves the new upstream fused speculative forward and
  its weight ownership untouched. Its compatible ROCm path remains eligible.
- Sampler top-k retains v0.33's positional `has_topp_only_rows` argument;
  Radiance's known bound is keyword-only. V1 and V2 load hooks are updated.
- CPU KV offload uses upstream mmap lifetime, runtime helpers, role detection
  and ROCm DMA support. Radiance retains coordinated whole-region registration,
  its rollback/required/auto policy, opt-in rank-major layout, and the prefix
  restore convoy fix. A failed unregister prevents unmapping pinned memory.
  This does **not** qualify the private TP2 layout on the new stack.
- R4D explicitly refuses quantized Q input, preserving its BF16-query ABI even
  when vLLM's Triton backend advertises quantized queries on gfx1201.
- The FP8 DFlash exclusion backport adapts Yufeng He's
  [vLLM PR 61010](https://github.com/vllm-project/vllm/pull/61010), commit
  `c9cf9d5e77151bf47116ba908df748908b5112fd`, to the stable mapper.
  Both `ignored_layers` and `exclude_modules` gain global-index/root aliases.
  ModelOpt's mapping is unchanged. This bookkeeping prevents unwanted draft
  projection quantization; it is not an end-to-end acceptance measurement.
- Draft-head startup memory adapts amittell's
  [GGZ14 Codeberg PR 60](https://codeberg.org/ggz14/radiance-vllm-mxfp4/pulls/60),
  commit `2ec647c12bdac200ba2cd06eb6498281e8dc3110`: bounded zero checks,
  smaller packing chunks and release of loader allocator reservations. INT2/4
  arithmetic and exact target-head policy remain unchanged.

### Compiler pin discrepancy

The v0.33 ROCm Docker recipes name AMD Torch 2.13, but their short source pin
`733fca1` did not resolve in ROCm/pytorch during this audit. This candidate uses
AMD's release/2.13 commit `0e8e47fd5bb5cd734b6194b92e941075e932c925`,
not an invented expansion of the unresolved short pin. It is a **new candidate
pair**, not a claim that upstream qualified this exact combination. The earlier
2.13 TP2 hang remains negative evidence; TP1 cannot clear that gate.

### Maintained kernel compatibility boundary

`tools/prepare_r4d.py` combines the retained v0.5.0 Python/private-extension ABI
base with attention and GDN-scan sources from maintained radiance commit
`451b136874084650992668bed9e8662deb58de88`. It copies the complete current
`R4DArgs` struct, zero-initializes new fields, explicitly preserves causality,
and passes fp32 state mode to the newer GDN scan. Engine/plugin entrypoints are
excluded at a checked boundary. The image retains upstream license, notice and
source hashes under `/opt/radiance-notices/libr4d-maintained/`.

The maintained attention includes the newer causal-row merge handling. The GDN
scan still forms nonpositive decay exponents, retaining the extreme-decay
correctness contract. New compressed-KV formats, standalone-engine integration,
and wholesale replacement of collectives are not enabled by this adaptation.
The remaining legacy bindings/private kernels are visible compatibility debt;
this report does not describe the entire retired library as current upstream.

Optional host embeddings, lazy GDN, and additional quantization formats remain
separate research candidates. No new experiment is silently made a default.

## Evidence so far

- All **36** active source overlays apply to disposable exact-source snapshots;
  repeated application is idempotent. This is source evidence only.
- Eight PR CPU checks pass, including negative controls for FP8 draft aliases,
  asymmetric-head LDS bounds, fail-closed oversized configurations and native
  plugin metadata discovery.
- Real Torch CPU INT2/INT4 packing agrees bit-for-bit with an independent
  reference across chunk boundaries; signed zero, NaN and shared packing reuse
  checks pass. Initially run in the previous image; repeat in the candidate.
- Maintained gfx1201 attention/GDN extension compiles with the new ROCm compiler
  and imports with the expected version and full maintained revision.
- Host-only symbol-interception probes of the compiled extension confirm GDN
  pointer/geometry/scale forwarding, the explicit fp32-state flag and stream,
  and attention's complete argument struct, causal flag and zeroed extension
  fields. These probes launch no GPU kernel and make no numerical claim.
- Twelve paged-attention cases pass on GPU 0 with the maintained extension
  under ROCm 10, using direct HIP calls and independent CPU-generated FP64
  references. Both BF16 and FP8 caches, shuffled/partial pages, guards and
  changed-input HIP graph replay pass; maximum relative error is 0.001772.
  The previous image also passes its independent oracle control. This is
  native-kernel evidence, not the still-pending new Torch/vLLM integration.
- Twenty-four maintained GDN extreme-decay cases pass on GPU 0 under ROCm 10,
  with FP64 output/state references and changed-query graphs for both 48/16 and
  24/8 geometries; maximum relative error is 0.003572.
- Sixteen newly compiled MXFP4/FP8 GEMM cases pass under ROCm 10, covering
  folded/per-block scaling, decode/prefill/TN4, K64-only widths, partial N and
  TP1-wide N=34816, guards, reset counters and changed-input graphs. Maximum
  relative error to exact stored-operand FP64 dots is 0.001683.
- Source, results, hashes and preflight limits are preserved in
  [native kernel evidence](../benchmarks/results/20261106-v033-native-kernels/README.md).
  These direct HIP checks do not qualify the pending Torch/vLLM/model integration.
- The full foundation source build is still running. No candidate-image serving,
  native GPU correctness, TP2, release, capacity or performance pass is claimed.

Preserved setup failures: an isolated attention probe initially lacked required
operand helpers; adding the actual upstream headers fixed compilation. A raw
ROCm-base import initially lacked `/opt/rocm/lib` in its library search path;
setting that path allowed import. Neither was a GPU correctness test.
