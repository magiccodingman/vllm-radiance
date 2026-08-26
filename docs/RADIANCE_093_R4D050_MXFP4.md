# Radiance 0.9.3, libr4d 0.5.0, and MXFP4 continuation

## Outcome

This continuation ports DeadCode Radiance 0.9.3 and libr4d 0.5.0 onto this
fork's stable vLLM 0.27.1 and AMD compiler stack, incorporates the latest
gfx1201 MXFP4 work, and qualifies the combined target-matched DFlash2 path on
two Radeon AI PRO R9700 GPUs.

The fastest general candidate is MXFP4/W4A8 target + selective-FP8 DFlash2 K7
+ `RADIANCE_FAST_DRAFT=1`. It reached **145.4 weighted BetterBench TPS** and
**132.4 / 234.6 / 343.3 / 416.9 aggregate TPS at c1/c2/c4/c8**. K5 reached
136.2 weighted and 123.0 / 216.4 / 343.0 / 435.7 TPS, making it the bounded
C8 option. Matched non-spec was 43.6 weighted and
43.2 / 83.1 / 145.7 / 241.3 TPS. Matched fast MTP K4 was 102.3 weighted and
97.8 / 173.1 / 284.5 / 372.1 TPS.

DFlash remains experimental. K7 completed the meaningful eight-prompt fixture
coherently and passed the required multi-tool schema test 30/30, but matched
non-spec exactly on only 1/8 fixed greedy prompts. No result in this report is
described as lossless.

## Immutable provenance

| Component | Exact source |
|---|---|
| This work | `agent/radiance-093-r4d050-mxfp4` based on `ce5b850` |
| DeadCode Radiance | `e1c99aab7050c6933af02a09888e104210752518` (0.9.3 line) |
| libr4d | v0.5.0 / `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7` |
| MXFP4 reference | `dba9defefb2de7f914fe9cb45ffdf49989c923d6` |
| vLLM | v0.27.1 / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| PyTorch | AMD 2.12 / `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5` |
| Triton | AMD 3.7.1 / `f0b55c07da61c71775bef6d1a15ebf846430ac75` |
| AITER | 0.1.20 / `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618` |
| ROCm | 7.14 |

Publication benchmarks used image tag
`vllm-radiance:0.9.3-dev.vllm0.27.1-r4d0.5.0-mxfp4.dflash2` at local digest
`sha256:a4f782cbee2f76a1e8156a051b95534a7d0c4247659211f3db7bff3bea707279`.
The final local tag digest is
`sha256:4d739aee7ef3e8a1db8466122aabe03fa03008215df6b3a220946a10d3a3c8d0`;
its runtime OCI manifest is
`sha256:cdbefb91b15b075b7556cf815c75c0f3ecacba728514acd3d77e3d23774d8ad5`,
the same runtime manifest used by the publication runs. The changing outer
index is BuildKit provenance/attestation metadata, not a runtime filesystem
change.

The target was `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`: weight SHA256
`be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272`, config
SHA256 `04c9b07a3a9260cbc8a2ea5b5e5f84ced8274cf412deb9895e91204383ed20e3`.
The target-matched drafter was `tcclaviger/Qwen3.8-27B-DFlash2-FP8`: weight
SHA256 `7dbb99a8d0120f502e66b256aa7c0866d933ceeee4a02463d9db591811e8404e`,
config SHA256 `5b5668a00b26aaebd88c7e3d961f7d1cdef025867fee158dfccb84f29fd8caec`.

## What changed

- Replaced the prior router-only integration with libr4d v0.5.0's complete
  operator registry and added its BF16, W4A16, W4A8, and DFlash kernels.
- Ported the guarded DFlash base, W4 linear, fused scaled-FP8 context-K/V,
  exact GDN metadata, skinny GEMM, and small-row Triton sampling changes.
- Generalized the INT2-g128 exact-rerank draft head for both MTP and DFlash.
- Added load-time W4 quantization for eligible DFlash draft linears behind
  `RADIANCE_FAST_DRAFT=1`.
- Extended the fused gfx1201 MXFP4 small-M kernel from M<=48 to M<=64 and
  enlarged its persistent scratch allocation accordingly.
- Stopped physically transposing activation scales, excluded W4 layers from
  the FP8 preshuffle hook, and installs the W4 runtime dispatcher.
- Isolated fast-draft vLLM and TorchInductor graph caches. A stale
  ordinary-draft graph otherwise retained the old rank-2 weight assumption and
  failed against packed W4 weights at startup.
- Made the DFlash FP8 migration patch recognize the prior fork's fused-K/V
  block, so `Dockerfile.patch` works against the current published image as
  well as a clean full build.
- Extended benchmark manifests and the live-server gate with configurable
  tool parser and `BENCH_TOOL_SCHEMA_ATTEMPTS` controls.

The normal Radiance FP8 dispatcher, R4D attention/GDN paths, TP2 custom
all-reduce, six-bit compressed payload path, FP8 KV contract, stable vLLM pin,
and AMD compiler pins remain intact.

## Benchmark contract

BetterBench v0.2.2 commit
`575cc3925bac922d6ad4a39e62502673799979d9` used corpus v1, 10 measured
passes/category, temperature zero, nonce-disjoint cold prompts, and 24 requests
at each of c1/c2/c4/c8. The server used TP2, FP8 KV, 8K model length, eight
admitted sequences, 4,096 batched tokens, 85% GPU allocation, R4D target
attention, Triton draft attention, `PIECEWISE` graphs, no prefix caching, no
dynamic draft, and `disable_padded_drafter_batch:true`. Cold prefill was tested
at nominal 2K/4K/7K depths. Warmup traffic remained disjoint from measurement.

All run directories are immutable beneath:

`benchmarks/runs/20260826T023001Z_radiance093-r4d050-mxfp4-continuation`

## BetterBench results

### Per-category single-stream median TPS

| Category | Non-spec | Fast MTP K4 | Fast DFlash K5 | Fast DFlash K7 |
|---|---:|---:|---:|---:|
| Chat | 43.7 | 86.7 | **97.4** | 95.6 |
| Code | 43.5 | 99.1 | 141.4 | **142.3** |
| File edit | 43.6 | 120.5 | 143.4 | **176.2** |
| JSON | 43.6 | 130.0 | 170.1 | **191.1** |
| Math | 43.5 | 116.8 | 167.4 | **193.1** |
| Prose | 43.6 | 78.7 | **106.0** | 94.1 |
| Reasoning | 43.6 | 85.6 | 113.3 | **121.3** |
| Summarization | 43.6 | 119.3 | 156.8 | **171.1** |
| **Weighted** | **43.6** | **102.3** | **136.2** | **145.4** |

K7 is +233% over matched non-spec and +42.1% over the matched current fast-MTP
weighted result. K7 significantly improved JSON relative to the
previous K7 publication; several other category shifts were positive but did
not clear the comparison tool's noise threshold.

The offline current-MTP-to-K7 comparison classified all eight category gains
as significant at its 95% threshold. It is an unpaired cross-file comparison,
so the raw medians remain the primary result; the complete output is preserved
as `comparisons/mtp-k4-vs-dflash-k7-betterbench.txt` in the run root.

### Concurrency and latency

| Mode | c1 | c2 | c4 | c8 | TTFT p50 | ITL 1% low |
|---|---:|---:|---:|---:|---:|---:|
| Non-spec | 43.2 | 83.1 | 145.7 | 241.3 | 72 ms | 42.0 TPS |
| Fast MTP K4 | 97.8 | 173.1 | 284.5 | 372.1 | 82 ms | 81.0 TPS |
| Fast DFlash K5 | 123.0 | 216.4 | 343.0 | **435.7** | 74 ms | 108.4 TPS |
| Fast DFlash K7 | **132.4** | **234.6** | **343.3** | 416.9 | 73 ms | 114.7 TPS |

K7 wins the balanced c1-c4 and category workload. K5 is 4.5% faster at c8.
Different prompts have materially different acceptance, so short synthetic
replicates can be noisy; the 10-pass category corpus is the publication result.
Against MTP, K7 is +35.4% / +35.5% / +20.7% / +12.0% at c1/c2/c4/c8.

### Prefill, telemetry, and headroom

| Mode | 2K prefill | 4K prefill | 7K prefill | Minimum headroom/GPU | Peak board power |
|---|---:|---:|---:|---:|---:|
| Non-spec | 3,644.7 | 4,136.5 | 4,003.1 | about 4.9 GiB | recorded in run telemetry |
| Fast MTP K4 | 3,395.3 | 3,846.7 | 3,761.0 | 6.05 GiB | 296 W |
| Fast DFlash K5 | 3,636.7 | 4,124.1 | 3,947.3 | about 5.2 GiB | 275 W |
| Fast DFlash K7 | 3,648.5 | 4,122.5 | 3,948.5 | about 5.3 GiB | 336 W |

Fast drafting does not materially degrade target prefill. The K7 publication
peaked at 69/67 C on the two cards. The 8K context qualification completed
8,128-input+64-output C1 and 7,936-input+64-output C2 cases with no OOM or
server errors. A sustained 512-output sweep reached
154.40 / 196.65 / 476.24 / 457.58 TPS at c1/c2/c4/c8 and retained about
5.3 GiB per GPU.

## Isolated controls

- M<=64 versus forced M<=48 at C8: 511.17 versus 463.07 quick-gate TPS,
  a direct **+10.4%** gain.
- Fast selective-FP8 drafter versus ordinary selective-FP8 draft, quick gate:
  +10.3% C1, +10.2% C4, and +8.0% C8.
- Runtime-quantizing the BF16 `z-lab` drafter worked, but reached only
  153.94 / 321.08 / 432.62 TPS at c1/c4/c8 and acceptance fell to 48.47% at
  C8. The target-matched selective-FP8 drafter is the recommendation.
- Matched non-spec remained essentially unchanged: 43.6 weighted versus 43.7
  in the previous publication.
- The publication MTP K4 lane retained 59.1% aggregate draft acceptance
  (53,927 accepted of 91,259 drafted tokens), with per-position acceptance of
  78.7% / 57.4% / 41.8% / 31.8%. It completed every request and retained at
  least 6.05 GiB physical VRAM per GPU.

## Correctness and negative results

The fixed meaningful fixture remained coherent in every candidate, but the
strict gate failed:

| Candidate | Exact matches versus matched non-spec |
|---|---:|
| ordinary DFlash K7 | 2/8 |
| ordinary DFlash K5 | 2/8 |
| fast DFlash K5 | 2/8 |
| fast DFlash K7 | 1/8 |
| fast MTP K4 | 2/8 |

Several first divergences had target logit margins of 0-0.125, consistent with
shape-sensitive numerical choices, but at least one observed margin was 0.375.
That is characterization, not proof of losslessness. The existing strict gate
was not weakened.

The qwen3_coder required-union tool test passed **30/30** on the final K7
configuration with all required fields present. This separately confirms that
the earlier moving-vLLM structured-output regression has not returned.

The first fast-DFlash attempt failed during graph capture with a wrong-rank
weight assertion. Investigation showed that the shared persistent compile
cache contained an ordinary selective-FP8 draft graph. The entrypoint cache
namespace fix resolved it; the failed run remains preserved as
`mxfp4-r4d050-dflash-k7-fastdraft-fp8-quick`.

An exact-image smoke also tried suffixing the raw Triton cache. Both TP workers
then raced over the fresh shared cache and emitted many recoverable missing-
HSACO reload warnings. It still completed at 145.9 TPS with zero failed
requests, but the extra suffix was removed; vLLM/Inductor graph isolation is
the part required for packed-weight safety.

The first overlay migration build also failed closed because the old fork's
fused-K/V block no longer matched the new source anchor. The migration-aware
patch was corrected and a second overlay build against the published image
applied every patch, compiled all 19 libr4d kernels, and passed imports. No
failed experiment was deleted or relabeled.

## Immutable run IDs

- `mxfp4-r4d050-nonspec-matched-betterbench-standard`
- `mxfp4-r4d050-dflash-k5-fastdraft-fp8-betterbench-standard`
- `mxfp4-r4d050-dflash-k7-fastdraft-fp8-betterbench-standard`
- `mxfp4-r4d050-mtp-k4-fastdraft-betterbench-standard`
- `mxfp4-r4d050-dflash-k7-m48-c8-ab`
- `mxfp4-r4d050-dflash-k7-fastdraft-fp8-cachefix-quick`
- `mxfp4-r4d050-dflash-k7-fastdraft-bf16-quick`
- `mxfp4-r4d050-dflash-k7-fastdraft-context-qualification`
- `mxfp4-r4d050-dflash-k7-fastdraft-sustained-qualification`
- `mxfp4-r4d050-dflash-k7-fastdraft-tool-schema-gate`
- `mxfp4-r4d050-dflash-k7-fastdraft-final-image-smoke`
- `mxfp4-r4d050-mtp-k4-fastdraft-quick`

Every directory contains the resolved command, environment, image inspection,
model metadata, raw responses, logs, telemetry, and reports appropriate to its
gate.

## Deployment recommendation

Keep non-spec as the generic correctness-qualified Compose default. Enable the
MXFP4/W4A8 path only for a compatible Quark checkpoint. For an experimental
throughput deployment with the exact AMD target and tcclaviger drafter, use K7,
draft TP2, `TRITON_ATTN` for the drafter, R4D for the target, `PIECEWISE`,
`disable_padded_drafter_batch:true`, mandatory FP8 KV, and
`RADIANCE_FAST_DRAFT=1`. Prefer K5 only when sustained C8 aggregate throughput
matters more than c1-c4 and category-weighted performance.

Production prefix caching and GDN `align` remain recommended for shared agent,
system, or RAG prefixes. They were disabled only in the cold BetterBench
contract. Long-context capacity numbers from the previous 128K/C4 production
qualification remain the conservative guide; this 8K performance continuation
did not attempt to increase the maximum resident context.
