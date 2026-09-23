# vLLM 0.30 resident platform migration

Status: **V030_CORE_QUALIFIED_WITH_OPTIONAL_SPECULATIVE_LIMITATIONS**.

The subsequent [full publication pass](V030_PUBLICATION_20260922.md) is complete:
non-spec/MTP K4/DFlash2 K5/K7 weighted TPS is 53.9/135.6/172.3/186.7. All standard
performance requests succeeded. Non-spec and both DFlash lanes passed 30/30
sampled tool checks; MTP failed one (29/30). All speculative lanes remain only
1/8 strict-equivalent to non-spec. Thus this is **not an all-modes-qualified
merge sign-off**. The failed MTP response is preserved, not retried away.
NVFP4A16 conversion now fails closed; ten conversion regression cases pass.
The bounded initial qualification below remains historical evidence, not the
current publication table. Production promotion/deployment remains separate.

The source platform is vLLM 0.30.0; publication and deployment are separate actions. The deferred
gate is real NVFP4 checkpoint model quality/continuous-recompute qualification;
both core resident FP8/Quark lanes and the bounded compatibility checks passed.

## Source and scope

The platform upgrade started from main `285ac78e7f19bc1e1b5b09b25f865aea0c6d9754`.
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
tiered-v2 implementation or device-managed pool is included. The initial bounded
qualification below precedes the separately authorized publication benchmark pass.

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
Radiance retains qualified ROCm 7.14 and AITER 0.1.20. The only required AITER
adaptation is the guarded preshuffled-function import alias described below.
Native W4A8 and runtime AITER JIT passed; no whole-stack replacement was needed.

## Qualification ledger

Build/import, overlay dispositions, Runner V2 path evidence, resident FP8 and
Quark MXFP4 gates, native NVFP4 conversion fixtures, parser/structured output,
vision and bounded performance sanity PASS. Real NVFP4 model quality remains
deferred. No historical test count is attributed to this candidate. Publication
BetterBench was excluded from that initial pass; the separate publication pass
is now complete and linked above, including its failed MTP tool gate.
Long-context capacity qualification remains outside this task.

## Overlay migration matrix

The machine-readable owner is `.radiance/patches.yaml`. Reasons describe the
retained behavior, not a new hardware claim. UPSTREAM_OWNED rows are not applied;
parser/XGrammar backport files were removed after installed behavior tests.
Retired scripts remain recoverable in Git history; installed/native regressions
replace their old source-string assumptions. Every retained owner remains Radiance; every
upstream-owned implementation belongs to the immutable vLLM pin.

| Overlay | Disposition | Behavior / ownership decision |
| --- | --- | --- |
| `patch_ar_geometry` | RETAIN_UNCHANGED | Expose the measured TP2 compressed all-reduce geometry controls. |
| `patch_ar_maxbytes` | RETAIN_UNCHANGED | Retain exact-message size limits for the qualified Radiance TP2 collective path. |
| `patch_conv1d_blockn` | RETAIN_UNCHANGED | Retain block-N compatibility for the qualified GDN/conv1d path. |
| `patch_dflash2_v0271_backport` | OBSOLETE_DELETE | Historical v0.27.1 DFlash2 source-copy backport; retained only for archaeology because v0.28 owns DFlash2 natively. |
| `patch_dflash_base` | UPSTREAM_OWNED | Upstream owns inert rows, rejected-suffix/null-block guards and length clamping; source contracts retained. |
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
| `patch_unpad` | UPSTREAM_OWNED | Installed unpadding preserves sliced CPU sequence bounds; behavioral regression retained. |
| `patch_verify_head` | MECHANICAL_PORT | Install the sampling-aware target verification head with exact fallback. |
| `patch_xgrammar_spec_reasoning` | UPSTREAM_OWNED | Backport reasoning-boundary validation for drafts generated before the grammar mask activates. |
| `patch_xgrammar_spec_termination` | UPSTREAM_OWNED | Backport speculative grammar termination handling so draft batches cannot overrun a terminated FSM. |
| `install_radiance_hooks` | RETAIN_UNCHANGED | Install Radiance runtime hooks into the pinned vLLM tree. |
| `patch_nvfp4_mxfp4` | SEMANTIC_PORT | Bounded NVFP4-only load-time conversion with per-partition scale provenance. |

The final source audit reapplied every release overlay to the clean pinned vLLM
tree, repeated each application, and parsed all resulting Python: PASS, no
silently skipped vLLM anchors. Dependency copies (AITER/Transformers/Torch) come
from the pinned image and can already contain Radiance changes: a dependency
NOOP is **not** evidence that upstream owns that behavior. Those overlays were
not retired on that basis. V1 and V2 GDN post-load anchors are both required.

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
Dev image `sha256:71fb273706cda58dc81d95cf737d060d5af524694770f3d6775c279abd83b5eb`
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

The clean release-image build also passed import/pip checks. Its source changes
do not replace Torch/Triton/AITER; actual final versions are Torch2.12.0+rocm7.14,
Triton3.7.1 (pinned source `f0b55c07`), vision0.27.1+df56172, AITER0.1.20 and R4D0.5.0.
The first clean release image is
`sha256:8bcb12e362ea0f29d9052f0bb7210d1913310cd960d85ae2eafeb7f2c6fdef8f`;
its OCI config digest is `sha256:6caa7e46974f87569a8528674872c2c7fa2a5fed28be86e43e1a1f750172cbfc`.
These are different identity types and must not be interchanged.

Quark release-image resident gates passed at TP2/C1/8K, FP8KV, R4D and
FULL_AND_PIECEWISE. Native W4A8 is selected for every eligible observed layer;
all48 dense-model GDN input projection pairs merged per rank. Three short32-token
trajectories matched; warm decode52.209/52.293tok/s is a sanity observation, not
yet a matched speedup. The1600-token repeated prefix recorded1568 cache-hit tokens
with identical generated IDs. JSON/tool/recompute and two vision requests passed.
The separate eight-token kernel observation contains actual native
`radiance_mxfp4_fp8_gemm_decode` activity, not emulation. This dense27B does not
qualify routed MoE, QSA or giant-model PLE; those are not claimed.

## Final candidate identity and matched-control scope

Runtime source: `977c919ef2ab64ea96579b36ffef7d93d493db19`.
Final release build image: `sha256:4c37eb5a935d79a6e0092cf527cda7581d24bf761e8a8ac90820859fc2738d00`.
It was built through the release Dockerfile, not by replacing site-packages in
a running server. The preceding Quark-tested release image differs only in
retired no-op overlay files and the legacy V1 post-load anchor. Installed V2
runner, MXFP4, GDN and R4D attention source hashes match exactly between them.
The final FP8 qualification uses the final image directly.

The same-host old1.0.16 Quark control used the same TP2/C1/8K, model, flags and
three short32-token requests. Its actual runner was V1, as selected by that
release; the new release is V2. An initial V2-only inspection assertion rejected
the old runner before any generation; preserved as a harness-scope failure,
not a model failure. Timing-only controls now explicitly skip that V2 assertion;
new-candidate qualification still requires it. Old warm rates48.220/48.200tok/s
versus new52.209/52.293 show no material regression in this bounded sample.
All six requests have the same output-ID hash. Do not describe this as an
isolated kernel speedup: runner/version changes and short-run noise remain.

Final-image FP8 resident qualification also passed: actual V2, original R4D,
Radiance preshuffled FP8, FULL_AND_PIECEWISE, exact three32-token trajectories,
three independent exact-ID next-token checks, positive1568-token cache reuse,
strict nested JSON, nested tool call and two identical red-image answers. The
existing deferred/open-object wire suite passed8/8 (four schemas, streaming and
nonstreaming), including two parallel calls and preserved nested arguments.
Short rates were34.288/37.846/38.000tok/s; first request includes lazy runtime
effects. The separate eight-token observation executed FP8 preshuffle, R4D GDN,
attention and `r4d_ar_oneshot_2rank_exact_kernel` on both ranks. Unquantized
linears remain original ROCm ops where the checkpoint/owner requires them;
they are not mislabeled as FP8. Profile timing is not a throughput result.

Pre-request snapshots (both ranks equal within each lane): FP8 allocated
26,810,180,608B, reserved28,043,116,544B, raw free4,644,143,104B; Quark allocated
26,798,494,208B, reserved28,047,310,848B, raw free4,677,697,536B. These are scopes,
not additive memory categories or a capacity qualification. Both use85% budget,
TP2/C1/8192 context and2048 batch-token cap, FP8 main KV and prefix alignment.

Warnings retained: upstream erased-FX-node and Triton block-pointer deprecations,
Transformers video-processor documentation messages, and unavailable NVIDIA
DeepSelect. No AMD backend is replaced by DeepSelect. The generic Quark
"simulated dequantization" warning precedes the Radiance selector; actual owner
objects and native W4A8 kernel activity establish the executed path. Short
profiler duplicate-flow warnings limit timing attribution but not observed
kernel names. Source unittest runs have no warning summary; no historical
MR37 pytest count is claimed for this different main-derived test suite.

## Bounded non-speculative performance sanity

Same models/TP2/C1/8K/85%/2048-token batch cap, FP8KV and R4D. Each row uses
three32-token requests: first fresh, repeat, independent fresh salt. The repeat
five-token prompt does not itself form a cache block; the separate1600-token
test proves prefix reuse. Steady rate excludes TTFT and uses31 visible output
intervals; short SSE timing and fresh-cache JIT effects are limitations.

| Lane/image | First TPS | Warm repeat TPS | Warm fresh TPS | TTFT first/repeat/fresh (s) |
| --- | ---: | ---: | ---: | --- |
| FP8, old1.0.16 | 26.996 | 35.321 | 35.356 | 0.572 / 0.064 / 0.064 |
| FP8, final0.30 | 34.288 | 37.846 | 38.000 | 1.091 / 0.057 / 0.055 |
| Quark, old1.0.16 | 35.546 | 48.220 | 48.200 | 0.564 / 0.036 / 0.035 |
| Quark,0.30 release | 46.209 | 52.209 | 52.293 | 1.122 / 0.029 / 0.028 |

No material decode regression is observed; this is not a statistically powered
speedup claim or proof that cold TTFT improved. All12 output-ID hashes equal
`20b490d92586d7ce3787238a95c8b320c5c8b07f4b9d6d2304580bbac23fb33f`.
No forced EOS, retokenized output IDs, repeated-until-good sample selection,
BetterBench, concurrency sweep or long-context capacity campaign was used.

## Source/native gate accounting

- Full PR plan:5/5 registered automatic gates, including all three KV-offload
  source/host lifetime/layout contracts and compileall.
- Eight additional installed/source scripts: NVFP4 (8 unit cases), XGrammar
  termination/reasoning, open-object grammar, FP8KV calibration, TunableOp,
  installed unpadding/DFlash ownership, tokenizer/parser and composite top-k
  (21 cases), all PASS. These are13 script-level gates total, not a pytest total.
- Registry validation:45 historical/current overlays,23 tests,5 upstreams.
- Clean pinned-vLLM source application and repeat:35 release overlays twice,
  70 successful operations, no syntax errors. Transitive patches remain registered.
- Both GPUs: installed NVFP4 loader/native W4A8/rollback and TP1wide-shape
  fixtures PASS; upstream DFlash stride/narrow guard, top-k rows1/2/8 and
  changed-input fixed-address graph replay PASS.
- Native conversion is fixture qualification, not a real NVFP4 model quality
  claim. Dense resident models do not qualify routed MoE/QSA/giant PLE.
- CPU KV integration is source/host-contract qualified here; destructive
  registration and long-context offload pressure are intentionally NOT RUN.

## Speculative compatibility and cleanup

One existing local FP8 MTP head was loaded with `method=mtp`, two speculative
tokens, original R4D and fresh cache. Both ranks reported actual V2
`vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator`. One16-token
completion matched the non-speculative exact-ID prefix. Metrics recorded7
drafts,14 proposed tokens and9 accepted tokens. This proves bounded integration
compatibility, not full speculative equivalence or performance qualification.
DFlash stride/null-block/config/source regressions passed; a new DFlash model
campaign was not run. Existing speculative modes remain opt-in.

The optional read-only worker extension was mounted from the qualification
checkout; its final addition only reports the original speculator object and
does not replace methods, change graph boundaries or perform model arithmetic.
The image's production runtime remains the source977c919ef identity above.

All lab containers were stopped after completed requests, with120-second maximum
graceful-stop bounds. No running experiment remains. Production1.0.16 is **stopped**,
image `sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc`,
original `unless-stopped` policy preserved. It was never replaced by a candidate.
MR !37 remains Draft/open/unmerged; its historical freeze note is retained.
No tiered-v2/device-pool work has begun. The next phase is the separate
[continuation plan](V030_FLASH_NEXT_CONTINUATION_PLAN.md), only after review/merge.

Compact machine-readable evidence is in
[`benchmarks/results/20260922-v030-platform.json`](../benchmarks/results/20260922-v030-platform.json).
Raw bounded request/native/kernel/source records are attached to MR !42; full
build logs and caches remain in the local `/nvme/ediloca-1/scratch/v030-*` runs.

The228,631-byte archive is split only to fit the connector's upload limit:
[part00](https://gitlab.sayou.io/-/project/9/uploads/58793777297329d0a45507d073533725/v030-platform-evidence.tar.gz.part00),
[part01](https://gitlab.sayou.io/-/project/9/uploads/950a5c0e8d7ab455147451836e8cd46f/v030-platform-evidence.tar.gz.part01),
[part02](https://gitlab.sayou.io/-/project/9/uploads/a01780e94213dc8a908e8854a6eb2bc6/v030-platform-evidence.tar.gz.part02),
[part03](https://gitlab.sayou.io/-/project/9/uploads/324aa20e2ad8dc11d696d8cdfbe70c4c/v030-platform-evidence.tar.gz.part03).
Concatenate in numeric order, then extract as gzip tar. Combined SHA256:
`39e4830192cb58d200834b922663e935c5fbd38616565cb10e4f4def7f9dfc6d`.
It contains no model weights or checkpoint archive. Original failed development
attempts remain locally preserved; the packet includes the old-runner inspection
failure and the corrected timing-only scope rather than concealing that change.
