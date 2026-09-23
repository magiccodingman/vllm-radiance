# vLLM 0.30 publication measurements — 2026-09-22

Status: **PUBLICATION_SET_COMPLETE_WITH_MTP_TOOL_QUALIFICATION_FAILURE**.
Core non-speculative serving is the recommendation. MTP is not tool-qualified;
DFlash remains opt-in/experimental. This is not an all-modes-qualified merge
sign-off. MR !42 remains Draft/open/unmerged for Lance's review. No production
image was published or deployed; production remains stopped by standing instruction.

## Identities and reproducibility

- Runtime build source: `487a62e829f16d35d86a0e5b6790050ed62a4c24`.
- Image: `vllm-radiance:v030-publication-r2`, inspected ID
  `sha256:6e4f456ab8617ebad12de4da50d297b96828365cab02bfd38a85d5d6d4b536db`.
- Radiance: `1.1.0-rc1.vllm0.30.0`; vLLM `0.30.0`,
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- ROCm 7.14 / AMD Torch 2.12 (`6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`),
  Triton 3.7.1 (`f0b55c07da61c71775bef6d1a15ebf846430ac75`), AITER 0.1.20
  (`fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`), R4D 0.5.0
  (`e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`), Transformers 5.15.0,
  XGrammar 0.2.3. Protected build base: ROCm digest
  `439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7`.
- Target: `/nvme/lexar-2/ai/models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd`,
  revision `156be69f9cac862a41d8b32e773ea2d2754341e8`;
  config SHA256 `04c9b07a3a9260cbc8a2ea5b5e5f84ced8274cf412deb9895e91204383ed20e3`.
- DFlash target-matched drafter:
  `/nvme/lexar-2/ai/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger`, revision
  `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`;
  config SHA256 `5b5668a00b26aaebd88c7e3d961f7d1cdef025867fee158dfccb84f29fd8caec`.
- Native W4A8 binary SHA256:
  `1370e6f3d722eca18b014f13c1d2f4fee371818603f8ececc5c1ce00181c5ae6`;
  R4D binary `daa7a3bf79d2a1e0a7909a6ed9ddac2f0f3ac74b878569f4eabc2ccae839aecc`.
  Both match the earlier qualified v0.30 binaries.

Every lane's `manifest.json` retains resolved Compose, container command/environment,
image identity, checkpoint metadata and host details. `runtime-owners.json` records
both ranks' actual class/source hashes, target/draft owners and graph mode. All
four lanes used `vllm.v1.worker.gpu.model_runner.GPUModelRunner`, PIECEWISE, R4D
target attention and **304 eligible native W4A8 target layers per rank**. No generic
MXFP4 emulation was relabeled W4A8. Target MXFP4 weights use dynamic FP8 activation
quantization; this is W4A8, not bit-identical checkpoint W4A4 execution.

MTP used V2 `MTPSpeculator` / `Qwen3_5MTP`, target-contained BF16 MTP tensors and
an INT2-g128 draft head with exact top-64 reranking. DFlash used V2
`DFlash2Speculator` / `DFlash2Qwen3ForCausalLM`, TRITON_ATTN draft attention,
the selective-FP8 checkpoint with eligible draft linears converted to W4, and
the same INT2 reranked-head policy. Logs prove actual conversion, not just flags.

## Established methodology, unchanged

BetterBench v0.2.2 commit `575cc3925bac922d6ad4a39e62502673799979d9`, corpus v1,
repository `benchmarks/betterbench/standard.json`. Two warmups and ten measured
passes per category; greedy seed 20260823, cold nonce prompts. TP2, FP8 KV,
8,192 context, eight sequences, 4,096 batch tokens, 85% GPU allocation, prefix
off/Mamba cache none, no async scheduling. WPERM/decode-NT on; tiled-A, GDN
norm-quant, norm-quant fusion and FP8 residual stream off. Dynamic draft/width
off. Fixed maximum K4/K5/K7; no depth or kernel tuning. Fresh per-mode compile caches.

Each lane completed **80/80** measured category requests, **96/96** concurrency
requests (24 at each level), and **12/12** prefill measurements. Standard warmups
are additional and excluded from reported performance statistics. BetterBench's
own report functions calculate the weighted score and ITL tails. No alternative
weighting, dropped category or retokenized model-output IDs were introduced.
All eight fixed correctness responses/lane completed with coherent reported counts.

Commands are reproducible through
`bash benchmarks/bin/run_v030_publication.sh MODE RUN_ROOT IMAGE`, where MODE is
`non-spec`, `mtp-k4`, `dflash-k5` or `dflash-k7`. This delegates to the established
runner and refuses to overwrite a lane. Identity RPCs are opt-in, localhost-only,
outside timed traffic, with no model hooks or tensor tracing.

## Current mode comparison

| Mode | Weighted TPS | ITL 1%-low TPS | TTFT p50 ms | c1 aggregate | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-spec | 53.9 | 51.7 | 64 | 53.3 | 101.8 | 181.4 | 304.1 |
| Fast MTP K4 — tool qualification FAIL | 135.6 | 109.7 | 67 | 128.7 | 227.3 | 360.2 | 463.9 |
| Fast DFlash2 K5 — experimental | 172.3 | 139.0 | 65 | 154.2 | 281.3 | 414.6 | 610.4 |
| Fast DFlash2 K7 — experimental | 186.7 | 143.5 | 64 | 173.5 | 293.7 | 446.6 | 510.2 |

Per-request median decode TPS is distinct from aggregate throughput:

| Mode | c1 per request | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| Non-spec | 53.7 | 51.9 | 49.6 | 45.1 |
| MTP K4 | 140.6 | 130.0 | 109.0 | 85.3 |
| DFlash2 K5 | 182.4 | 166.2 | 128.7 | 97.2 |
| DFlash2 K7 | 204.5 | 187.8 | 143.9 | 95.2 |

K7 leads weighted/c1–c4 performance; K5 leads c8. This is an observed workload
tradeoff, not permission to tune depths. Non-spec category TPS is nearly flat
(53.7–54.0), with chat's lower ITL tail (45.7). K7 is much faster on JSON (249.1)
and math (238.9) than prose (130.5), chat (134.3) or reasoning (144.7); its weighted
score does not hide those slower categories. K7's code ITL low is 116.1 despite
190.5 median decode TPS. Every mode's complete category/ITL/TTFT/IQR table is linked below.

## Prefill

Standard nominal 2K/4K/7K fixtures produced median **1,556 / 3,023.5 / 5,226 actual
prompt tokens**. Do not label these exact 2,000/4,000/7,000-token inputs. Throughput
is actual prompt tokens divided by TTFT; four measured requests/depth after one warmup.

| Mode | nominal 2K TPS | nominal 4K TPS | nominal 7K TPS |
|---|---:|---:|---:|
| Non-spec | 4,056.4 | 4,480.0 | 4,367.2 |
| MTP K4 — tool qualification failed | 4,239.0 | 4,531.3 | 4,348.2 |
| DFlash2 K5 | 4,040.9 | 4,458.4 | 4,283.7 |
| DFlash2 K7 | 4,044.1 | 4,456.5 | 4,280.8 |

## Acceptance and actual depth

| Lane | Scope | Proposed | Accepted | Acceptance | Actual depth evidence |
|---|---|---:|---:|---:|---|
| MTP K4 | 52 logged server intervals, including correctness/tools | 100,840 | 56,258 | 55.79% | Both V2 runners report 4; four-position acceptance vectors in logs; dynamic controller OFF |
| DFlash2 K5 | Benchmark-bound counters, including standard warmups | 109,200 | 56,702 | 51.92% | 21,840 drafts; exactly 5.0 proposals/draft |
| DFlash2 K7 | Benchmark-bound counters, including standard warmups | 138,922 | 58,711 | 42.26% | 19,846 drafts; exactly 7.0 proposals/draft |

DFlash accepted-plus-bonus means are 3.596 and 3.958 tokens/draft. Full per-position
counters are retained. MTP's original wrapper exited on the failed tool gate before
exporting final Prometheus counters; its log-derived scope is **not** benchmark-only
and is not an interchangeable acceptance comparison. The harness now preserves
counters on gate failure and brackets BetterBench separately. No MTP rerun was
performed merely to obtain a nicer result or fill this instrumentation gap.

## Correctness and merge-readiness boundary

| Gate | Non-spec | MTP K4 | DFlash2 K5 | DFlash2 K7 |
|---|---|---|---|---|
| Publication request/accounting health | PASS | PASS | PASS | PASS |
| Required multi-tool schema, temperature 1 | 30/30 PASS | **29/30 FAIL** | 30/30 PASS | 30/30 PASS |
| Strict fixed output against non-spec | Reference | **1/8 FAIL** | **1/8 FAIL** | **1/8 FAIL** |

The strict comparator checks exact generated text and reported output lengths,
with identical prompts/seed; no fabricated token IDs or tolerance. K5 vs K7 is
also only 6/8. Speculative modes remain experimental and no production-equivalence
promotion is claimed. Performance comparisons can involve different generated
trajectories and are not isolated per-token kernel-speed comparisons.

MTP tool attempt 006 generated an unfinished `click` JSON object followed by
repeated whitespace, exhausted 1,024 output tokens, and returned `finish_reason=length`
without usable tool calls. Hermes' JSON parse error is a consequence of incomplete
generation, not evidence that a parser should invent missing arguments. HTTP success
alone did not pass the gate. The source/evidence does not uniquely justify an ordinary
parser or configuration correction; no numerical route, whitespace constraint,
token cap, parser, schema or acceptance assertion was changed to conceal it.
This **blocks an all-modes/tool-qualified sign-off**. Non-spec remains recommended;
K7 is only an experimental single-stream performance option, K5 the observed c8 leader.

Release build, pip/import/native selection and **13 source/installed script-level
gates PASS** (five PR gates plus eight installed checks; ten NVFP4 cases and 21
top-k cases are included, not additional script counts). Registry: 45 overlays,
23 registered tests, five upstreams. These are not MR !37's pytest totals.
CPU-only inspection emits expected no-driver notices; retained runtime warnings
include FP8 scale defaults, optional DeepSelect/cuteDSL availability, Transformers
processor documentation and speculative batch-cap warnings. No warning-free claim.

Earlier qualified v0.30 FP8 resident, prefix/exact-ID recompute, nested JSON,
open-object tools and ordinary vision evidence is retained in [V030_UPGRADE](V030_UPGRADE.md).
No expensive FP8/vision rerun was triggered by prose, dormant inspection packaging,
or the default-off NVFP4 guard. This publication profile deliberately has prefix off;
it does not replace the separate prefix-on qualification or qualify long-context capacity.

NVFP4A16 now fails closed before backend selection/construction when conversion is
enabled. Supported quantized-input conversion, default-off behavior, preserved
FP8/BF16/lm_head policy and rejected extra conversions have regression coverage.
Real NVFP4 model quality remains **DEFERRED/NOT RUN**, not qualified by these Quark runs.

## Historical context, not an isolated vLLM speedup claim

Earlier non-spec/MTP/K5/K7 weighted results were 43.6/102.3/136.2/145.4 TPS;
the later safe-RX5 K7 publication was 183.1 weighted, 137.6 ITL low and
163.0/286.1/462.0/523.5 aggregate TPS. The current K7 result is 186.7/143.5 and
173.5/293.7/446.6/510.2: improvements are not uniform across concurrency.
These historical sources differ in Radiance source/profile and vLLM speculative
implementation; no causal “v0.30 +X%” claim is made. All old reports remain intact.

## Immutable artifacts and cleanup

Root: [20260922-v030-publication](../benchmarks/results/20260922-v030-publication/).

- [Machine-readable summary](../benchmarks/results/20260922-v030-publication/summary.json)
- [Non-spec complete category report](../benchmarks/results/20260922-v030-publication/non-spec/betterbench/report.md)
- [MTP complete category report — failed tool qualification](../benchmarks/results/20260922-v030-publication/mtp-k4/betterbench/report.md)
- [DFlash2 K5 complete category report](../benchmarks/results/20260922-v030-publication/dflash-k5/betterbench/report.md)
- [DFlash2 K7 complete category report](../benchmarks/results/20260922-v030-publication/dflash-k7/betterbench/report.md)
- [Telemetry summary](../benchmarks/results/20260922-v030-publication/telemetry-summary.csv)
- [Portable artifact hashes](../benchmarks/results/20260922-v030-publication/ARTIFACT_SHA256SUMS)
- [Validation and failed-attempt evidence](../benchmarks/results/20260922-v030-publication/validation/)

Each lane retains manifest, exact command, source/class identities, raw BetterBench
JSON/HTML/Markdown, one-second telemetry, fixed outputs, all tool responses and logs.
MTP retains `status.txt=failed`; a completed throughput table does not overwrite it.
Its final Prometheus export is unavailable, explicitly noted above. The earlier
bounded [platform-sanity JSON](../benchmarks/results/20260922-v030-platform.json) is unchanged.

An initial no-model-loaded startup failed because the optional identity probe was
not packaged. Commit `487a62e82` installs the dormant module; the failed log and both
build identities remain under validation. No failed benchmark configuration was
repeated to select better statistics. No giant model, BetterBench variant sweep,
capacity campaign, tiered-v2/device pool, model download or production deployment occurred.
All benchmark containers stopped. Production 1.0.16 remains stopped, with its original
image/configuration and `unless-stopped` policy unchanged. No automatic restoration.
