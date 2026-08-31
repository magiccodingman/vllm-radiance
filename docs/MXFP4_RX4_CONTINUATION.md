# RX4 MXFP4 decode continuation

## Scope and provenance

This continuation starts from merged fork `main` at
`10a55730efc281c7548ed7d7a30fa6fa82428715` and selectively ports Brian's RX4
work from
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4).
The audited upstream point is main commit
`b9d7ecf8bed3d4be5e99ad7582ad0d64bb27fc99`; its final authored head is
`5950b383dfe9026df638e8f6bbd69bae968ea71b`. Original authorship is retained in
the four imported Git commits.

The sequence matters:

1. `db9dba680f77` adds traced activation quantization, the FP8 residual stream,
   fused TP2 all-reduce epilogues, shared speculative GDN metadata, and the
   composite small-k sampler.
2. `3d159921edc8` restores new-file hunks that were accidentally omitted when
   the libr4d patch was regenerated. A source build without this correction is
   not an RX4 build.
3. `eafcac951d4f` adds an epilogue prefetch experiment that regressed real
   prefill. It is retained in history for provenance.
4. `5950b383dfe9` reverts that regression, selects 512 threads for large
   prefill epilogues, and adds an inert DFlash selector-width control.

The final tree therefore contains the corrected state, not the briefly shipped
prefetch regression. The upstream launcher is not copied: this fork keeps its
portable Compose, immutable benchmark laboratory, pinned vLLM/libr4d builds,
and deployment-specific environment overlays.

## Integrated operators

- `RADIANCE_NORMQUANT_FUSION=1` atomically hoists MXFP4 activation
  quantization into the traced graph, selects the plain-Torch FP8 quant math,
  and merges vLLM's norm/activation fusion passes into the existing compiler
  JSON.
- `RADIANCE_FP8_STREAM=1` lets adjacent decoder layers exchange an FP8
  activation plus scale while a fused libr4d kernel performs TP2 all-reduce,
  residual add, RMSNorm, and quantization. Unsupported layers retain the stock
  contract.
- `RADIANCE_GDN_SHARED_BUILD=1` builds identical speculative GDN metadata once
  per step and updates only each KV group's block-table-derived state indices.
- `RADIANCE_TOPK_COMPOSITE=1` replaces a serial whole-vocabulary scan with
  multi-block `torch.topk` and a bounded mask when every active top-k is at or
  below `RADIANCE_TOPK_COMPOSITE_KCAP` (64 by default).
- `RADIANCE_DFLASH_SELECTOR_TOPK` can raise DFlash's selector truncation for a
  labeled experiment. Unset or zero preserves the checkpoint and remains the
  default.

The old AR-overlap experiment remains excluded. RX4 composes with this fork's
already-qualified WHT6 large-message all-reduce, GDN merge/fused update,
dynamic verify width, exact rerank/verify head, R4D attention, and safe
M<=64 decode boundary.

## Safety policy

The generic native-FP8 profile is unchanged. Traced quant and FP8 stream are
off by default because they require the Quark MXFP4 W4A8 TP2 geometry. Enabling
FP8 stream without all prerequisites exits with status 64 instead of silently
running an unintended graph.

Radiance environment flags are not included in vLLM's persistent AOT cache
key. The entrypoint therefore namespaces graph caches by non-default GDN
unmerge, traced quant, FP8 stream, and fast-draft state. The established RX3
GDN-merged default deliberately keeps its old cache lineage; changing that
lineage caused a reproducible sampled-output regression even with every RX4
execution feature disabled. The entrypoint also rejects a separate explicit
`--compilation-config` when the RX4 aggregate profile is active; callers must
provide `RADIANCE_COMPILATION_CONFIG` so the required pass settings can be
merged into the same JSON object.

## GPU-free validation

Before production was touched, the following passed against the exact merged
RX3 production image and pinned sources:

- all three new vLLM patch scripts applied cleanly to vLLM 0.28.0;
- the corrected `r4d_radiance_extras.patch` applied to libr4d commit
  `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`;
- the overlay image compiled `r4d.so` and `radiance_mxfp4_fp8.so` for gfx1201;
- `pip check`, native imports, exact version checks, and required RX4 symbol
  checks passed;
- the entrypoint derived both traced-quant flags, merged PIECEWISE plus both
  compiler passes, and selected an isolated
  `radiance-gdnm-nqft-fp8s-fast-draft` cache namespace;
- an invalid FP8-stream profile failed closed with status 64;
- the composite top-k/top-p implementation matched vLLM's full-sort reference
  on 21 CPU cases, including top-k boundary ties;
- the GPU-free speculative XGrammar termination test and shared
  reasoning/tool-parser structural test passed;
- public and benchmark Compose files parsed successfully.

## Live qualification

The local final overlay image is
`vllm-radiance:mxfp4-rx4-final@sha256:e45786fe206ee5b7c9336423118669790560b9187578c0e0552461ee704955b9`.
It was built on the immutable production/RX3 digest
`sha256:79b0a7386b9ba910e2eec0179b6e2d8335167961fa9d70f7510180939b2f10cd`.
The 100/100 live gate used digest
`sha256:df26b78b95ba13157ccd52a560b1a2e5e3c2eea2e5b9b0a525ea84dca9e978e5`;
the final digest differs only in human-facing wording for an invalid
FP8-stream dependency error, after which package/import/Compose checks were
repeated.
All runs below are under
`benchmarks/results/20260831T1130Z_mxfp4-rx4/`; failed and diagnostic runs are
preserved rather than overwritten.

The matched BetterBench control kept traced quant and FP8 stream off. The full
candidate enabled both. Both used the AMD Quark MXFP4 target, matched
tcclaviger DFlash2-FP8 drafter, TP2, K7, FP8 KV, PIECEWISE graphs, cold-prefix
measurement, and identical seeds/corpus.

| Metric | RX4-dark matched control | Full RX4 | Full vs control |
|---|---:|---:|---:|
| weighted single-stream decode | 174.0 t/s | 177.7 t/s | +2.1% |
| ITL 1% low | 133.0 t/s | 105.1 t/s | -21.0% |
| c1 aggregate | 155.3 t/s | 155.0 t/s | -0.2% |
| c2 aggregate | 274.4 t/s | 268.8 t/s | -2.0% |
| c4 aggregate | 399.4 t/s | 410.4 t/s | +2.8% |
| c8 aggregate | 493.3 t/s | 490.4 t/s | -0.6% |
| 2K prefill median | 3980.1 t/s | 3865.0 t/s | -2.9% |
| 4K prefill median | 4547.1 t/s | 4471.0 t/s | -1.7% |
| 7K prefill median | 4313.5 t/s | 4257.1 t/s | -1.3% |

Publication-grade run IDs:

- control: `rx4-control-betterbench-standard`;
- full RX4: `rx4-full-v2-betterbench-standard`.

Single-stream category medians were:

| Category | Control t/s | Full RX4 t/s |
|---|---:|---:|
| chat | 115.9 | 120.7 |
| code | 173.6 | 171.3 |
| file edit | 198.0 | 192.3 |
| JSON | 222.1 | 236.2 |
| math | 224.2 | 229.5 |
| prose | 124.0 | 121.7 |
| reasoning | 146.7 | 154.4 |
| summarization | 204.5 | 218.7 |

The small weighted-average gain is not operationally persuasive: it came with
worse c1/c2/c8, worse prefill at every measured depth, and a much worse ITL
tail. Relative to the prior qualified RX3 publication run
`20260830T0424Z_mxfp4-rx3/final-dflash-betterbench-standard-min5`, full RX4 was
approximately +1.6% at c1, -0.3% at c2, -5.1% at c4, and -14.0% at c8. Most
of the high-concurrency difference also appears in the matched RX4-dark
control, so it must not be attributed solely to the new kernels.

## Correctness investigation

The first full startup exposed an integration defect omitted by the upstream
port: callers used `torch.ops.radiance.mxfp4_linear_pq`, but the custom-op
registration was absent. The registration and fake implementation are now
present, and every Docker build checks the symbol before it can succeed.

After that fix, all eight fixed greedy prompts diverged from the matched
control, first at token positions `32, 1, 22, 39, 123, 44, 0, 1`. The profile
therefore fails strict equivalence. The result is consistent with the RX4 FP8
residual stream adding a lossy representation boundary, but it remains a real
qualification failure regardless of whether the text is coherent.

The sampled multi-tool gate found another useful distinction:

| Run | Result |
|---|---:|
| full RX4 | 27/30 |
| RX4-dark candidate in a new AOT namespace | 29/30 |
| exact immutable RX3 production control | 100/100 |
| RX4 overlays using the established RX3 cache lineage | 30/30 |
| final RX4-dark image with corrected namespace logic | 100/100 |

The failed samples repeated valid `<tool_call>` blocks until the 1024-token
limit, leaving no parsed `tool_calls` object. A bounded bisect substituted the
exact production `r4d.so`, MXFP4 extension, and legacy Python overlays, and
also removed the three RX4 vLLM source patches. None removed the deterministic
one-in-thirty miss. Preserving the qualified graph/Inductor cache lineage did.
The default namespace logic was corrected accordingly; genuinely different
RX4 graphs remain isolated.

Relevant immutable diagnostics include
`rx3-production-digest-toolgate-control-100`,
`rx4-core-old-r4d-v2-toolgate`, `rx4-core-old-mxfp4-toolgate`,
`rx4-core-nopatches-toolgate`,
`rx4-core-old-python-production-cache-v2-toolgate`, and
`rx4-core-production-cache-toolgate`. The final milestone is
`rx4-final-default-toolgate-100`.

Three preserved directories are infrastructure failures, not measurements:
`rx4-full-betterbench-standard` exposed the missing pre-quantized op before
startup; `rx4-core-old-r4d-toolgate` accidentally retained the diagnostic
container's `sleep` entrypoint; and
`rx4-core-old-python-production-cache-toolgate` leaked a container-only cache
path into the host benchmark client. Their corrected successors carry `v2`.

## Decision

The RX4 implementation is retained as an explicit experimental lane, but
`RADIANCE_NORMQUANT_FUSION=0` and `RADIANCE_FP8_STREAM=0` remain the shipped
defaults. It is not a production recommendation: the throughput signal is
small and mixed, its 1%-low latency is worse, and strict greedy/tool
qualification fails. The qualified deployment remains the RX3 execution path
with the new source available dark for continued kernel work.
