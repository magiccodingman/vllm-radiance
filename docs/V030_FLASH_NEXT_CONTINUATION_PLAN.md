# After the resident v0.30 platform MR merges

This is a plan, not work started in this MR. Branch a NEW Draft MR from upgraded
main only after Lance reviews/merges the platform upgrade. MR !37 stays a frozen
v0.28 evidence corpus; do not transplant its registration/ABI glue wholesale.

1. Establish v0.30's actual ROCm Qwen owners and resident behavior first.
2. Port regressions before architecture: F0/F1/F2 exact histories, N9/18-step,
   saved projection operands, GDN initialized/zero continuation and state,
   HC projection and FP32 sparse-router weight exactness, prefix/cache state,
   structured output, resource/source/staging/publication/terminal lifetimes.
   Delete old workarounds when upstream passes their saved failure; otherwise
   port only the proved semantic correction.
3. Reuse still-needed verified immutable backing, sealed planner budgets,
   pageable DDR5/NVMe authority, registered staging, expert-wave prefill,
   native routed W4A8/I320, reader leases and fail-closed publication,
   R4 placement evidence and host mixed-service reference.
4. Evaluate device-managed residency as a second backend, not a replacement
   assumed correct. A stable GPU-addressable warm host source is distinct from
   pageable/NVMe authoritative storage. Budget pinned coverage explicitly:
   VRAM misses with a warm addressable source may use device publication;
   pageable/NVMe misses need bounded host admission before stable publication.
   Do not pin the entire bank to mimic a CUDA/Blackwell implementation.
5. Hold checkpoint, representation, TP/context, expert capacity, warm-source
   coverage, prefix state, graph mode and corpus constant across backends.
   Keep non-speculative decode as the controlling measurement.

## v0.30 upstream families to inspect on the actual AMD path

Separate QSA prefill/decode indexers, FP8 QSA indexer cache, sparse-GQA padded
index skipping, fused PLE and PLE residual/QSA gate fusion, Engram/PLE offload,
and modern runner/MTP integration are candidate families, not ROCm performance
claims. Audit each installed dispatch and platform gate. NVIDIA-only
FlashInfer/DeepGemm/DeepSelect implementations do not become AMD kernels by
changing an allowlist. Generic metadata/model code may be reused; qualify
Triton/ROCm implementations at the exact geometry before promotion.

Pinned-source findings (not giant-model hardware qualification):

| Family | v0.30 AMD implementation | Disposition for next MR |
| --- | --- | --- |
| QSA prefill/decode split | `models/qwen4_exp/amd/indexer_qsa.py` calls unified `qsa_select_paged_tokens`; NVIDIA calls separate decode/prefill selectors | Missing equivalent split on AMD; profile before port |
| FP8 indexer cache | AMD constructs raw and compressed caches in BF16; NVIDIA has explicit `indexer_kv_dtype == "fp8"` branch | Not automatically available on AMD; new numerical/cache gate required |
| QSA pre-indexer fusion | NVIDIA has `_supports_fused_pre_indexer`; AMD uses `ReplicatedLinear`, portable norm and RoPE helpers | Retain portable AMD path as reference |
| Sparse padded-index handling | AMD Triton `ops/qsa.py` masks valid indices/rows | Correctness masking exists; do not claim NVIDIA's optimization or speedup |
| PLE | AMD `ple_layer.py` has portable grouped norm, ngram embedding, original short-conv metadata and shared `common/ple.py` embedding | Audit actual residual/gate and offload owners separately; no NVIDIA fusion imported here |
| Runner/MTP | Generic config and QSA metadata exist; AMD `skip_topk` preserves MTP step0 indices | Port exact history/state tests before reusing this optimization |

The resident27B gates exercise Qwen3.5-style dense/GDN owners, not Qwen4Exp QSA
or the giant Flash-Next PLE table. Do not label those unexecuted paths qualified.

Ordinary resident vision belongs to the platform foundation. Large-model native
multimodal ownership is later, with coarse vision residency separate from expert
granularity; pageable vision remains another phase. MTP/speculation follows a
strong correct base decoder and must not hide poor non-speculative performance.

Retain all !37 positive/negative experiments and exact source/image identities.
Do not restart broad numerical archaeology when a saved minimal fixture answers
the question. No tiered-v2 or device-pool implementation is included here.
