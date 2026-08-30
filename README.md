# vllm-radiance

[![Docker Hub](https://img.shields.io/docker/v/magiccodingman/vllm-radiance?sort=semver&label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/magiccodingman/vllm-radiance)
[![Docker Pulls](https://img.shields.io/docker/pulls/magiccodingman/vllm-radiance?logo=docker)](https://hub.docker.com/r/magiccodingman/vllm-radiance)

A vLLM inference-server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It combines a pinned
vLLM v0.28.0 ROCm stack with [libr4d](https://codeberg.org/StillDeadcode/libr4d)'s hand-written RDNA4
attention, gated-delta-net, vision, all-reduce, MXFP4, and DFlash kernels while retaining Radiance's tuned
FP8 GEMM and speculative-decoding paths.

> **Status: experimental.** The primary qualified environment is two R9700s (TP2), native FP8 or AMD
> Quark MXFP4 target weights, and mandatory FP8 KV. Other models, quantization recipes, GPU counts, and
> hardware may work but have not received the same qualification. Speculative modes remain opt-in because
> their strict cross-mode output-equivalence gate has not passed.

This fork tracks and credits DeadCode's
[vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) and libr4d work, with additional compiler
pins, native v0.28 DFlash2 plus focused post-release correctness backports, native gfx1201 MXFP4/W4A8
support, reproducible benchmarks, and deployment qualification. Published images are at
[`magiccodingman/vllm-radiance`](https://hub.docker.com/r/magiccodingman/vllm-radiance).

## Quick start

The portable Compose file contains no machine-local paths. Copy the environment template and point it at
your model directory:

```bash
git clone https://gitlab.sayou.io/lance-wright/vllm-radiance.git
cd vllm-radiance
cp .env.example .env
# Edit MODELS, MODEL_PATH, and SERVED_MODEL_NAME in .env.
mkdir -p vllm-cache
docker compose up -d
docker compose logs -f
```

The reusable baseline is native FP8 weights, FP8 KV, TP2, 16K maximum context, 85% GPU allocation, an
eight-request admission ceiling, and automatic prefix caching with hybrid-GDN state alignment. It listens
on `0.0.0.0:8000`, retains language and vision support, enables Qwen tool/reasoning parsers, loads the
checkpoint-native chat template and generation defaults, and allows clients to override request-level
sampling and reasoning effort.

`MAX_NUM_SEQS` is an admission ceiling—not a promise that every admitted request can simultaneously reach
`MAX_MODEL_LEN`. Select both from the measured capacity tables below.

Common operations:

```bash
docker compose up -d
docker compose ps
docker compose logs -f vllm
curl -fsS http://localhost:8000/health
docker compose down
```

Host paths, GPU IDs, private image tags, and local overrides belong in the gitignored `.env` or an ignored
`docker-compose.dev.yml`, never in the public Compose file. See `.env.example` and
`docker-compose.dev.example.yml` in the
[source repository](https://gitlab.sayou.io/lance-wright/vllm-radiance).

## Target formats

### Native FP8

The default Compose profile expects a native-FP8 checkpoint:

```dotenv
WEIGHT_QUANTIZATION=fp8
GPU_UTIL=0.85
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=8
```

Radiance keeps its preshuffled block-FP8 dispatcher, fused RMSNorm/FP8 quantization, split-K fixes, R4D
attention/GDN, and custom TP2 all-reduce. Replacing the FP8 dispatcher with the generic upstream AITER
linear path was 9–11% slower in matched controls.

### AMD Quark MXFP4 with native W4A8

For [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4), point
`MODEL_PATH` at the checkpoint and use:

```dotenv
WEIGHT_QUANTIZATION=auto
RADIANCE_MXFP4=1
RADIANCE_MXFP4_W4A8=1
RADIANCE_MXFP4_W4A8_MIN_M=0
RADIANCE_MXFP4_DECODE_MAX_M=64
RADIANCE_MXFP4_TN4_MIN_M=2048
```

`auto` lets vLLM consume the checkpoint's Quark metadata. On gfx1201, W4A8 retains packed OCP group-32
MXFP4 weights and dynamically quantizes activations to FP8 E4M3 so the kernels use RDNA4's native FP8 WMMA
path. Keep `RADIANCE_MXFP4_W4A8_MIN_M=0`: the generic AITER W4A4 fallback is numerically incorrect for one
of the qualified Qwen GDN projections. The decode-shaped kernel covers `M<=64`; larger batches use the
prefill kernel.

The checkpoint's embedded MTP tensors are BF16. Add `RADIANCE_QUARK_BF16_MTP=1` only when selecting its MTP
profile. Non-spec and DFlash do not need that override.

Implementation, numerical controls, provenance, and immutable runs are documented in
[MXFP4/W4A8 on dual R9700](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_W4A8_R9700.md).

## Serving modes

Choose exactly one mode. `RADIANCE_SPECULATIVE_CONFIG` contains either MTP or DFlash; the modes are not
cumulative.

| Mode | Separate drafter | Required profile |
|---|---|---|
| Qualified non-spec | No | Leave speculative variables unset |
| Fast MTP | No; head is stored in the target | MTP JSON plus `RADIANCE_FAST_DRAFT=1` |
| Experimental DFlash2 | Yes | V2 runner, `PIECEWISE` graphs, draft TP2, matched context, fast draft |

### Fast MTP

For a Qwen checkpoint with an in-checkpoint MTP head:

```dotenv
RADIANCE_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D","disable_padded_drafter_batch":true}'
RADIANCE_FAST_DRAFT=1
```

K8 is a ceiling. Radiance's dynamic controller may select a shallower depth based on confidence and active
batch size. Fast draft uses an INT2-g128 LM-head copy with exact top-64 reranking; target verification remains
in place.

### DFlash2

For the native-FP8 ARA target and its selective-FP8 drafter:

```dotenv
MAX_MODEL_LEN=8192
VLLM_USE_V2_MODEL_RUNNER=1
RADIANCE_COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE"}'
RADIANCE_FAST_DRAFT=1
RADIANCE_SPECULATIVE_CONFIG='{"method":"dflash","model":"/models/Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","max_model_len":8192,"disable_padded_drafter_batch":true}'
```

For AMD's Quark MXFP4 target, use the target-matched
[`tcclaviger/Qwen3.8-27B-DFlash2-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8)
drafter. `RADIANCE_FAST_DRAFT=1` runtime-quantizes eligible draft linears to W4 and uses the INT2 exact-rerank
head. The target retains R4D attention while the drafter uses Triton attention. The current image also
merges GDN input projections, uses libr4d's fused speculative GDN update, increases exact-rerank width to
64, and narrows DFlash verification per request only at c5 and above when observed acceptance says the
full K7 target verification is wasteful. Every optimization is independently reversible through the
controls documented in `benchmarks/README.md`.

Prefix caching and `MAMBA_CACHE_MODE=align` remain the deployment defaults. Disable them only for a cold,
nonce-disjoint benchmark or a deliberate maximum-capacity experiment. Recreate the container after changing
modes:

```bash
docker compose down
docker compose up -d
```

## Measured capacity on two 32 GiB R9700s

These values include FP8 KV, TP2, no CPU/KV offload, and deliberate VRAM headroom. They are model- and
profile-specific; larger models and different drafters must be requalified.

### Native FP8 target plus selective-FP8 DFlash drafter

Measured at 85% GPU allocation with prefix caching disabled for the capacity laboratory:

| Maximum context | Conservative `MAX_NUM_SEQS` | Highest completed burst |
|---:|---:|---:|
| 8K | 8 | 8 |
| 16K | 7 | 8 |
| 32K | 5 | 6 |
| 64K | 3 | 3 |
| 128K | 2 | 2 |
| 256K | 1 | 1 |

Every submission completed; minimum observed physical headroom was 4.41 GiB per GPU.

### Quark MXFP4/W4A8 target plus matched DFlash drafter

Measured at 90% GPU allocation. The target payload is 18.44 GiB versus 28.75 GiB for the native-FP8
regression target, a 10.31 GiB (35.9%) reduction.

| Maximum context | Conservative production C | Highest completed burst |
|---:|---:|---:|
| 32K | 8 | 11 |
| 64K | 6 | 7 |
| 128K | 4 | 4 |
| 256K | 2 | 2 |

The recommended long-context deployment is **128K/C4**, 90% allocation, prefix caching enabled,
`MAMBA_CACHE_MODE=align`, and DFlash K7. The capacity qualification itself used K5 (the draft depth does not
change the reserved model/KV capacity): it exposed 576,001 GPU KV tokens (4.39 full 128K requests), completed
four simultaneous full-context requests without OOM or preemption, and retained 5.17 GiB minimum physical
headroom per GPU. A repeated 32K prefix reduced TTFT from 9.04 seconds cold to 0.70–0.71 seconds warm. Prefer
K5 only for a workload that remains dominated by steady c8 traffic.

Full methodology and run IDs are in the
[Compose capacity report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/COMPOSE_CAPACITY.md).

## Measured performance

BetterBench v0.2.2 used its v1 corpus, ten measured passes per category, greedy decoding, cold nonce-prefixed
prompts, and c1/c2/c4/c8 on two R9700s. The current Radiance 0.9.3/libr4d 0.5.0 Quark lane measured:

| Mode | Weighted single-stream TPS | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| Non-spec | 43.6 | 43.2 | 83.1 | 145.7 | 241.3 |
| Fast MTP K4 | 102.3 | 97.8 | 173.1 | 284.5 | 372.1 |
| Fast DFlash K5 | 136.2 | 123.0 | 216.4 | 343.0 | **435.7** |
| Fast DFlash K7 | **145.4** | **132.4** | **234.6** | **343.3** | 416.9 |

K7 is the general DFlash recommendation and wins weighted decode plus c1–c4. K5 is 4.5% faster for a
steady C8-heavy deployment. The K7 candidate passed 30/30 required multi-tool schema requests, but strict
speculative/non-spec greedy equivalence passed only 1/8 fixed prompts; DFlash therefore remains experimental
and opt-in.

The stable-v0.28 continuation reran this exact K7 lane against a fresh
merged-main v0.27.1 control. v0.28 measured **152.5 weighted TPS** and
135.0 / 222.3 / 348.8 / 474.0 TPS at c1/c2/c4/c8: +4.8% weighted and +13.2%
at c8, with a disclosed -5.2% c2 regression and essentially flat c4/prefill.
It passed 100/100 sampled required multi-tool calls after the focused upstream
parser fix. Exact provenance, acceptance, telemetry, negative results, and run
IDs are in the [v0.28 qualification report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/V028_UPGRADE.md).

The subsequent RX3 continuation integrates Brian's latest MXFP4/DFlash/GDN
work from `ggz14/radiance-vllm-mxfp4`. The final exact-default standard run
measured **171.0 weighted TPS**:

| | weighted | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| Merged v0.28 baseline | 152.5 | 135.0 | 222.3 | 348.8 | 474.0 |
| RX3 continuation | 171.0 | 152.5 | 269.5 | 432.6 | 570.4 |
| Gain | **+12.1%** | **+13.0%** | **+21.2%** | **+24.0%** | **+20.3%** |

The final single-stream category medians were chat 131.1, code 176.4,
file-edit 204.5, JSON 233.2, math 227.4, prose 110.1, reasoning 125.8, and
summarization 196.2 TPS. Six category gains are statistically significant in
BetterBench's offline comparison; prose and reasoning remain noise-limited.
Prefill improved 9.7–10.5% across the 2K/4K/7K depths. Three standard runs
produced the same eight fixed greedy outputs across independent server starts
and passed 90/90 sampled required multi-tool calls with zero XGrammar FSM errors. DFlash
still does not pass strict equivalence against non-spec, so it remains clearly
experimental despite the speedup. Full provenance is in the
[RX3 continuation report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_RX3_CONTINUATION.md).

Per-category TPS, acceptance, TTFT/TPOT, prefill, telemetry, confidence intervals, negative results, and
immutable run IDs are in the
[Radiance 0.9.3 qualification report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/RADIANCE_093_R4D050_MXFP4.md).

## What is included

- **libr4d 0.5.0:** RDNA4 attention, GDN prefill/decode/spec-state handling, vision flash attention, exact and
  rotated-six-bit TP2 all-reduce, BF16/DFlash GEMMs, and DFlash-specific kernels.
- **Radiance FP8 paths:** preshuffled block-FP8 GEMMs, split-K alignment fixes, fused RMSNorm/quantization,
  and guarded fallbacks.
- **Native Quark MXFP4/W4A8:** packed OCP group-32 weights with dynamic FP8 activation quantization and
  separate small-M decode and prefill kernels.
- **Fast speculative drafting:** dynamic MTP depth, verbatim n-gram tails, INT2 exact-rerank heads, and W4
  DFlash draft linears.
- **Hybrid-safe prefix caching:** automatic prefix caching with GDN convolution/recurrent-state restoration
  through `--mamba-cache-mode=align`.
- **Spec-safe structured output:** upstream XGrammar termination and reasoning-boundary fixes prevent
  speculative draft batches from overrunning or desynchronizing the tool-call grammar.
- **Topology qualification:** a background startup sweep reports GPU enumeration, P2P access, NUMA distance,
  and peer-copy bandwidth.

Unsupported geometries fall back per operator. AITER, FLA, Triton, and RCCL controls remain available for
matched experiments.

## Build

The published image is built entirely from pinned source commits:

| Component | Version/pin |
|---|---|
| vLLM | 0.28.0, `2cf0a6915ce544dc493a0990f2ea38d81601128a`, plus reviewed DFlash/XGrammar/parser/ROCm-graph fixes |
| AMD PyTorch | 2.12 branch, `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5` |
| AMD Triton | 3.7.1, `f0b55c07da61c71775bef6d1a15ebf846430ac75` |
| AITER | 0.1.20, `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618` |
| libr4d | 0.5.0, `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7` |
| ROCm userspace | 7.14 |

```bash
docker build \
  -t vllm-radiance:$(cat VERSION) \
  --build-arg RADIANCE_VERSION=$(cat VERSION) \
  .
```

The multi-stage build compiles the stack for `gfx1201`, prunes unrelated ROCm device code, builds libr4d with
the image's `hipcc`, and copies only the runtime into the release stage. A compiler and headers remain in the
release image because AITER JIT-compiles kernels on first use. The pruned image measured 3.66 GiB compressed,
down from 9.35 GiB before pruning. A full build takes hours; `Dockerfile.patch` provides a guarded overlay for
ordinary Radiance/libr4d iteration without rebuilding PyTorch and the compiler stack.

Do not independently bump PyTorch, Triton, torchvision, or vLLM. The qualified versions are a compiler stack,
and an earlier mismatched combination caused sustained TP hangs.

## Documentation

- [Upgrade and reproducibility history](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/UPGRADE_PROGRESS.md)
- [Stable vLLM v0.28 upgrade and qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/V028_UPGRADE.md)
- [Radiance 0.9.3 / libr4d 0.5.0 qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/RADIANCE_093_R4D050_MXFP4.md)
- [MXFP4/W4A8 implementation and validation](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_W4A8_R9700.md)
- [Compose capacity and prefix-cache qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/COMPOSE_CAPACITY.md)
- [DFlash2 optimization and correctness investigation](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/DFLASH2_OPTIMIZATION.md)
- [XGrammar speculative-decoding backport](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/XGRAMMAR_SPECULATIVE_BACKPORT.md)
- [BetterBench methodology and earlier mode comparison](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/LIBR4D_BETTERBENCH.md)
- [Benchmark laboratory usage](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/benchmarks/README.md)
- [Complete runtime knob reference](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/DOCKERHUB.md)

The source repository is the canonical location for detailed qualification evidence. This landing page is
intentionally concise so the same content can be published as the Docker Hub repository overview.

## Upstream and attribution

This fork exists on top of two unusually strong RDNA4 efforts:

- [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) and
  [StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) provide the core Radiance runtime and
  hand-written gfx1201 kernels.
- [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4), authored by Brian, is the
  source of the native Quark MXFP4/W4A8 work and the RX3 optimization series adapted here. Its original
  authorship is preserved in the Git history.

The continuation pins the exact audited ggz14 upstream commit in its qualification report. Changes are
ported selectively because this fork carries a different vLLM/libr4d base and additional DFlash and
correctness patches; attractive results from incompatible or failed experiments are not silently copied.
