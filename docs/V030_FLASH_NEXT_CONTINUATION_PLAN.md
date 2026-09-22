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

Ordinary resident vision belongs to the platform foundation. Large-model native
multimodal ownership is later, with coarse vision residency separate from expert
granularity; pageable vision remains another phase. MTP/speculation follows a
strong correct base decoder and must not hide poor non-speculative performance.

Retain all !37 positive/negative experiments and exact source/image identities.
Do not restart broad numerical archaeology when a saved minimal fixture answers
the question. No tiered-v2 or device-pool implementation is included here.
