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

Two additional opt-in laboratories preserve the qualified default: immutable,
checkpoint-bound FP8 attention/KV calibration sidecars for fidelity work, and
an offline collect → tune → verified-serve PyTorch TunableOp workflow for
residual BLAS GEMMs. Neither is enabled until its exact model/profile passes
the normal correctness and benchmark gates. See
[FP8-KV calibration and persisted TunableOp](docs/FP8_KV_TUNABLEOP.md).

**Target verify head:** when target-head acceleration is enabled, this revision
uses **global-256 candidate selection by default** on supported TP1 requests.
It removes the former eight-candidates-per-tile restriction while keeping the
drafter unchanged. Unsupported requests use the full BF16 target head. This is
an approximate acceleration, with measured recall and numerical differences
reported [below](#target-verify-head-global-256). It does not enable speculative
decoding by itself or extend the existing TP2 qualification to this new path.

Start with [Quick start](#quick-start), choose a [target format](#target-formats)
and one [serving mode](#serving-modes), then select the measured
[capacity](#measured-capacity-on-two-32-gib-r9700s) for that profile.
[Repository layout](#repository-layout-and-execution-order),
[Build](#build), and [Verification](#verification) describe how source changes
reach the server and how to check them.

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

The quick start runs the configured image. To use the global-256 implementation
from this checkout, [build this revision](#build) and point `IMAGE` at
that image. Editing the host source or Compose default alone does not install
the implementation into a previously published image.

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
RADIANCE_MXFP4_WPERM=1
RADIANCE_MXFP4_DECODE_NT=1
```

`auto` lets vLLM consume the checkpoint's Quark metadata. On gfx1201, W4A8 retains packed OCP group-32
MXFP4 weights and dynamically quantizes activations to FP8 E4M3 so the kernels use RDNA4's native FP8 WMMA
path. Keep `RADIANCE_MXFP4_W4A8_MIN_M=0`: the generic AITER W4A4 fallback is numerically incorrect for one
of the qualified Qwen GDN projections. The decode-shaped kernel covers `M<=64`; larger batches use the
prefill kernel.

The final two switches are the qualified RX5-safe decode subset. They store
weights in the kernel's fragment order and use non-temporal decode loads. The
more aggressive RX5 A-tiled, norm-quant, and FP8-stream paths remain available
only as disabled experiments; do not infer that `RX5` as a whole is qualified.

The checkpoint's embedded MTP tensors are BF16. Add `RADIANCE_QUARK_BF16_MTP=1` only when selecting its MTP
profile. Non-spec and DFlash do not need that override.

Implementation, numerical controls, provenance, and immutable runs are documented in
[MXFP4/W4A8 on dual R9700](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_W4A8_R9700.md).

The later ggz14 RX4 traced-quant and FP8 residual-stream kernels are included
but remain off by default. On this dual-R9700 qualification they produced only
a mixed +2.1% weighted single-stream signal, regressed ITL 1%-low by 21%, did
not improve c1/c2/c8 or prefill, and failed strict greedy/tool-call gates. Do
not enable `RADIANCE_NORMQUANT_FUSION` or `RADIANCE_FP8_STREAM` in production;
see the [RX4 continuation report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_RX4_CONTINUATION.md).
The subsequent safe-kernel selection, FP8-KV calibration work, and complete
RX5 negative results are recorded in the
[RX5 continuation report](docs/MXFP4_RX5_FP8KV_CONTINUATION.md).

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
batch size. Fast draft uses an INT2-g128 LM-head copy with BF16-weight reranking
of 64 candidates; target verification remains in place. The drafter's shortlist
and the [target-head shortlist](#target-verify-head-global-256) are separate.

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
drafter. `RADIANCE_FAST_DRAFT=1` runtime-quantizes eligible draft linears to W4 and uses the INT2 BF16-rerank
head. The target retains R4D attention while the drafter uses Triton attention. The current image also
merges GDN input projections, uses libr4d's fused speculative GDN update, sets drafter rerank width to
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

## Target verify head: global-256

The target head scores possible next tokens. Its shortlist determines which
tokens the sampler can choose, so candidate selection affects the answer.
The former fast head kept only eight candidates from each 64-token vocabulary
tile, then globally reranked 80 in the measured profile. With `top_k=20`, more
than eight required tokens can occupy one tile and disappear before reranking.

The default target method now uses the complete INT2 score vector:

```text
hidden states → INT2 scores for the entire vocabulary
              → global top-256 candidates
              → rescore using original BF16 weights
              → existing sampler
```

There is no per-tile quota in this path. The drafter keeps its existing
block-8 selection and rerank configuration; it shares the packed weights but
does not pay for the target's deeper candidate set.

### Configuration and fallback

With `RADIANCE_FAST_DRAFT=1` and `RADIANCE_VERIFY_HEAD=1`, no additional opt-in
is needed for global-256. Python and Compose both default
`RADIANCE_VERIFY_HEAD_GLOBAL_TOPK` to `256`.

| Setting | Target-head behaviour |
|---|---|
| `RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=256` | Default global-256 candidate selection |
| `RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=128` | Smaller global shortlist for comparison |
| `RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=0` | Legacy block shortlist; sampled `top_k > min(RERANK // 4, 8)` uses the full head |
| `RADIANCE_VERIFY_HEAD=0` | Full reference target head for every request |

Global selection supports **TP1, BF16 inputs/weights, supported layouts and at
most 32 target rows**. Sampled requests require positive `top_k` no larger
than one quarter of the candidate depth, so `top_k=20` uses global-256. That
margin is empirical. Greedy requests do not use the top-k limit.

The full head handles TP2, unsupported shapes/dtypes, embedding bias, grammar
masks, logprobs, sampled min-p, unsupported sampled top-k, penalties, logit
bias/allowed-token masking, bad words, thinking-budget interventions and
unknown sampler layouts. These checks select the fallback before sampling;
they do **not** detect approximation misses in an otherwise eligible request.

### One R9700: 60K+ generated tokens per method

Eleven intact private Pi coding request boundaries contain **57,008–65,527
input tokens**. Each method generated at least **60,000 measured output tokens**
across 115 natural completions; no tools were executed. This used one R9700,
TP1, fixed D7 speculation, ROCm 7.14, PyTorch 2.12.0, Triton 3.7.1 and a
248,320 × 5,120 BF16 head. These results are separate from the historical
dual-GPU BetterBench results later in this README.

| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained | Measured tok/s | Estimated tok/s |
|---|---:|---:|---:|---:|---:|
| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) | 63.9 | 63.9 |
| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) | 67.2 | 67.2 |
| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) | 67.5 | 67.2 |
| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) | 66.8 | 67.1 |

Head time is the median GPU time for an **eight-row target verification
invocation**, with 1,265 timed invocations per method. Top-1 compares the final
argmax token ID. Complete top-20 retention requires every reference token at or
above the twentieth score to survive, including ties; it does not imply
identical logits, rankings or probabilities.

An independent reference pass generated **61,561 output tokens**. All **15,191
consecutive head invocations / 119,988 prediction rows** were replayed, including
prefill and rejected speculative rows. Every full-head digest reproduced exactly.
Global-256 reduced complete-top-20 misses from **21,536 to 202**, adding **0.042 ms**
to median head time versus block-8. It retained the reference winner in every
row, but its reranking arithmetic changed two final argmax results and 6,295
retained logits, with maximum absolute difference 0.25.
**Global-256 remains approximate; it is not certified lossless.**

Measured rates pool time after first output at temperature 1, top-p 0.95 and
top-k 20, excluding warmup and prefill. Global-256 was **4.5% faster than full
BF16** in these runs. Estimates hold output and acceptance fixed and replace
only head cost. Repeat full-head requests reproduced their earlier output on
54/111 prompt/seed pairs, so end-to-end differences cannot be attributed solely
to shortlist selection. The accuracy replay uses identical hidden vectors.
These sample counts are not universal correctness probabilities.

See [the methodology and limitations](docs/VERIFY_HEAD_GLOBAL_TOPK.md),
[long-run aggregate measurements](benchmarks/results/20260916-verify-head-global-topk-long/summary.json)
and [validation evidence](benchmarks/results/20260916-verify-head-global-topk-long/validation.json).
The [earlier short pilot](benchmarks/results/20260916-verify-head-global-topk/summary.json)
is retained separately. Private conversation text and captured tensors are not published.

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

### Optional CPU KV offload

Set `RADIANCE_KV_OFFLOADING_SIZE` (GiB) only when the required long-context
envelope exceeds the GPU KV tier. Radiance coordinates mmap host registration
across TP workers and defaults `RADIANCE_KV_OFFLOAD_PIN_POLICY=auto`: every rank
uses pinned DMA only when all ranks register successfully; otherwise the failed
HIP error is drained and all ranks coherently use slower pageable DMA. Use
`required` when silently losing pinned-transfer performance is unacceptable, or
`disabled` as a diagnostic control.

`RADIANCE_KV_OFFLOAD_REGISTER_CHUNK_GIB=0` is the shipped default and preserves
one whole-region registration. Positive chunk sizes are experimental until
qualified on the deployment host. The dual-R9700 32/36 GiB investigation and
reproducible maintenance probe are documented in
[ROCm KV-offload registration hardening](docs/ROCM_KV_OFFLOAD_REGISTRATION.md).

## Measured performance

The following TP2 measurements predate the global-256 target-head change.
They describe their recorded revisions and profiles; they do not qualify the
new TP1 path or establish the speed of this revision on TP2.

BetterBench v0.2.2 used its v1 corpus, ten measured passes per category, greedy decoding, cold nonce-prefixed
prompts, and c1/c2/c4/c8 on two R9700s. The current recommended MXFP4 kernel
profile adds `RADIANCE_MXFP4_WPERM=1` and `RADIANCE_MXFP4_DECODE_NT=1` while
keeping full RX5 (`A_TILED`, `GDN_NORM_QUANT`, `NORMQUANT_FUSION`, and
`FP8_STREAM`) disabled. The measured serving lane used the matched DFlash K7
drafter, TP2, FP8 KV, and PIECEWISE graphs:

| Weighted single-stream | ITL 1%-low | TTFT p50 | c1 | c2 | c4 | c8 |
|---:|---:|---:|---:|---:|---:|---:|
| **183.1 TPS** | **137.6 TPS** | **64 ms** | **163.0** | **286.1** | **462.0** | **523.5** |

Single-stream category medians:

| Category | Decode TPS | ITL 1%-low TPS | TTFT p50 |
|---|---:|---:|---:|
| Chat | 140.4 | 118.6 | 66.0 ms |
| Code | 188.8 | 116.5 | 63.1 ms |
| File edit | 218.8 | 152.6 | 67.5 ms |
| JSON | 249.4 | 183.4 | 63.5 ms |
| Math | 243.7 | 196.4 | 62.8 ms |
| Prose | 117.9 | 108.2 | 63.1 ms |
| Reasoning | 134.8 | 108.6 | 63.1 ms |
| Summarization | 210.1 | 186.5 | 68.3 ms |

Cold prefill measured **4,031.8 / 4,444.7 / 4,268.6 TPS** at the 2K/4K/7K target depths. Every concurrency
arm completed 24/24 requests. These are the standard 8K/C8 laboratory results at 85% GPU allocation with
prefix caching and CPU offload disabled; the 128K/C4 production profile above intentionally has a different
capacity/latency contract. Exact category TTFT, ITL, prefill, run metadata, and immutable raw results are in
the [current safe-subset BetterBench report](benchmarks/results/20260908T1905Z_safe-rx5-final/safe-wperm-nt-betterbench-standard/betterbench/report.md)
and [RX5 continuation report](docs/MXFP4_RX5_FP8KV_CONTINUATION.md).

DFlash remains experimental and opt-in because strict speculative/non-spec greedy equivalence has not passed,
even though the stable-default lane passed its meaningful-output and sampled tool-call qualification. The
full RX4/RX5 traced-quant, tiled-prefill, GDN norm-quant, and FP8
residual-stream profile is not represented by the table above and remains off.
The full RX5+DFlash interaction passed only 93/100 tool calls; constraining it
to one tool call improved that to 98/100 but did not qualify it.

For historical mode-to-mode context, the earlier Radiance 0.9.3/libr4d 0.5.0 matched publication measured:

| Mode | Weighted single-stream TPS | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| Non-spec | 43.6 | 43.2 | 83.1 | 145.7 | 241.3 |
| Fast MTP K4 | 102.3 | 97.8 | 173.1 | 284.5 | 372.1 |
| Fast DFlash K5 | 136.2 | 123.0 | 216.4 | 343.0 | **435.7** |
| Fast DFlash K7 | **145.4** | **132.4** | **234.6** | **343.3** | 416.9 |

This older table is retained because non-spec, MTP, K5, and K7 have not all been rerun on the current RX4-dark
image. Do not treat it as the current K7 performance ceiling. Its per-category TPS, acceptance, TTFT/TPOT,
prefill, telemetry, confidence intervals, negative results, and immutable run IDs are in the
[Radiance 0.9.3 qualification report](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/RADIANCE_093_R4D050_MXFP4.md).

## What is included

- **libr4d 0.5.0:** RDNA4 attention, GDN prefill/decode/spec-state handling, vision flash attention, exact and
  rotated-six-bit TP2 all-reduce, BF16/DFlash GEMMs, and DFlash-specific kernels.
- **Radiance FP8 paths:** preshuffled block-FP8 GEMMs, split-K alignment fixes, fused RMSNorm/quantization,
  and guarded fallbacks.
- **Native Quark MXFP4/W4A8:** packed OCP group-32 weights with dynamic FP8 activation quantization and
  separate small-M decode and prefill kernels.
- **Fast speculative drafting:** dynamic MTP depth, verbatim n-gram tails, INT2 BF16-rerank heads, and W4
  DFlash draft linears.
- **Global target candidate selection:** default global-256 for eligible TP1
  requests, an unchanged block-8 drafter, explicit legacy controls and full-head
  fallback. Measured candidate recall and score fidelity are reported separately.
- **Hybrid-safe prefix caching:** automatic prefix caching with GDN convolution/recurrent-state restoration
  through `--mamba-cache-mode=align`.
- **Spec-safe structured output:** upstream XGrammar termination and reasoning-boundary fixes prevent
  speculative draft batches from overrunning or desynchronizing the tool-call grammar; Qwen structural-tag
  normalization also preserves open nested objects used by generic deferred-tool wrappers.
- **Topology qualification:** a background startup sweep reports GPU enumeration, P2P access, NUMA distance,
  and peer-copy bandwidth.

Unsupported geometries fall back per operator. AITER, FLA, Triton, and RCCL controls remain available for
matched experiments.

## Repository layout and execution order

```text
vllm-radiance/
├── Dockerfile, Dockerfile.patch       # Full pinned build and compatible overlay
├── docker-compose.yml, .env.example  # Server arguments, paths and runtime controls
├── patch_*.py, _patchlib.py          # Guarded source patches applied during build
├── install_radiance_hooks.py         # Install the vLLM plugin-loader hook
├── radiance_*.py, radiance_*.hip     # Runtime dispatchers and native kernels
├── radiance_entrypoint.sh            # Image startup and server execution
├── fp8-configs/, mxfp4-configs/      # Tuned GEMM configurations
├── moe-configs/                      # Tuned MoE configurations
├── tests/                           # CPU dispatch and native verify-head checks
├── benchmarks/
│   ├── bin/                         # Benchmark and correctness drivers
│   ├── fixtures/, betterbench/      # Inputs and benchmark profiles
│   └── results/                     # Recorded measurements and provenance
└── docs/                            # Implementation and qualification reports
```

The normal sequence is **build → configure → start → verify → benchmark**.
The Dockerfiles define source-patch order; individual patch scripts operate
on the pinned image's installed packages and are not host setup commands.

| Entry point | Inputs | Operation and outputs |
|---|---|---|
| `Dockerfile` | Pinned dependency commits, patches, runtime sources, kernel configs | Builds the compiler/runtime stack and a tagged serving image |
| `Dockerfile.patch` | A compatible built image and updated source checkout | Applies guarded overlays, rebuilds small HIP extensions and produces a replacement image |
| `patch_*.py`, `_patchlib.py` | Installed vLLM/AITER source and expected anchors | Patch supported source layouts; build output records applied/no-op/failed patches |
| `install_radiance_hooks.py` | vLLM plugin loader | Installs `radiance_kernels.install_all()` before model loading |
| `radiance_entrypoint.sh`, `radiance_preamble.py` | Environment, devices and server arguments | Run startup checks/reporting and launch `vllm serve` |
| `patch_verify_head.py` | V2 model-runner sampling call site | Adds the per-step target-head eligibility hook before logits computation |
| `radiance_verifyhead.py` | Target hidden states, BF16 weights and sampler metadata | Selects global candidates or the full head and returns target logits |
| `radiance_drafthead.py` | Drafter hidden states and shared packed weights | Produces draft-head scores with its existing block shortlist |
| Other `radiance_*.py` modules | Model tensors, state and per-feature environment flags | Dispatch attention, GDN, GEMM, sampling, KV/offload and telemetry operations |
| `tests/test_verify_head_*.py` | Public synthetic tensors and request metadata | Assert dispatch, capacity, default selection, fallback and native kernel behaviour |
| `benchmarks/bin/` | Model/profile, prompt fixtures and endpoint or image | Produces run manifests, responses, timings and correctness reports; see the [lab guide](benchmarks/README.md) for each driver |

The target-head runtime pipeline is
`patch_verify_head.py → before_compute_logits → sampling eligibility → global-256 or full head → sampler`.
The native test suite exercises the public hook as well as the numeric path;
CPU tests cover default and fallback decisions without allocating GPU memory.

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

After building, select the resulting image explicitly before starting Compose:

```bash
IMAGE="vllm-radiance:$(cat VERSION)" docker compose up -d
```

Keep that image selection in `.env` for subsequent starts. Runtime feature
changes require container recreation so the worker imports the new defaults.

## Verification

For CPU dispatch tests, use an environment with NumPy and pytest:

```bash
python -m pytest -q tests
```

For native operator and hook tests, run from this checkout inside the pinned
ROCm/PyTorch/Triton environment with NumPy, pytest and access to a test GPU:

```bash
RADIANCE_TEST_NATIVE=1 python -m pytest -q tests
```

These tests use public synthetic tensors and do not need a model checkpoint.
The native implementation passed **135 tests with no skips** before the
default-selection change. The current default change passes **117 CPU tests**
and explicitly skips 19 native tests. Its numerical kernels and fallback
predicates are unchanged; native tests were not repeated for this
configuration change. Revision identities and source hashes are preserved in
the [validation record](benchmarks/results/20260916-verify-head-global-topk/validation.json).

The checks cover mixed/reordered batches, unset global-256 selection, explicit
legacy selection, unsupported sampling transformations, clustered top-20
tokens, target row counts 1/2/3/8/16/32, unchanged drafter output and switching
between global and full-head execution. They establish these tested
properties, not universal model equivalence. Use the
[benchmark laboratory](benchmarks/README.md) for model-level throughput,
tool-call and long-context qualification after operator checks pass.

## Documentation

- [Global-256 target head: configuration, measurements and limitations](docs/VERIFY_HEAD_GLOBAL_TOPK.md)
- [Upgrade and reproducibility history](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/UPGRADE_PROGRESS.md)
- [Stable vLLM v0.28 upgrade and qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/V028_UPGRADE.md)
- [Radiance 0.9.3 / libr4d 0.5.0 qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/RADIANCE_093_R4D050_MXFP4.md)
- [MXFP4/W4A8 implementation and validation](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_W4A8_R9700.md)
- [RX4 MXFP4 continuation and qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/MXFP4_RX4_CONTINUATION.md)
- [Compose capacity and prefix-cache qualification](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/COMPOSE_CAPACITY.md)
- [DFlash2 optimization and correctness investigation](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/DFLASH2_OPTIMIZATION.md)
- [XGrammar speculative-decoding backport](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/XGRAMMAR_SPECULATIVE_BACKPORT.md)
- [Qwen open nested-object tool-call fix](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/QWEN_OPEN_OBJECT_TOOL_FIX.md)
- [BetterBench methodology and earlier mode comparison](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/docs/LIBR4D_BETTERBENCH.md)
- [Benchmark laboratory usage](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/benchmarks/README.md)
- [Complete runtime knob reference](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/blob/main/DOCKERHUB.md)

The source repository holds the implementation and detailed qualification
evidence. Recorded benchmark results retain their original workload and
revision scope; changing a default does not retroactively qualify a new build.

## Upstream and attribution

This fork exists on top of two unusually strong RDNA4 efforts:

- [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) and
  [StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) provide the core Radiance runtime and
  hand-written gfx1201 kernels.
- [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4), authored by Brian, is the
  source of the native Quark MXFP4/W4A8 work and the RX3/RX4 optimization series adapted here. Its original
  authorship is preserved in the Git history.

The continuation pins the exact audited ggz14 upstream commit in its qualification report. Changes are
ported selectively because this fork carries a different vLLM/libr4d base and additional DFlash and
correctness patches; attractive results from incompatible or failed experiments are not silently copied.
