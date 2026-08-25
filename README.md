# vllm-radiance

[![Docker Hub](https://img.shields.io/docker/v/magiccodingman/vllm-radiance?sort=semver&label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/magiccodingman/vllm-radiance)
[![Docker Pulls](https://img.shields.io/docker/pulls/magiccodingman/vllm-radiance?logo=docker)](https://hub.docker.com/r/magiccodingman/vllm-radiance)

A vLLM inference server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It combines a pinned
vLLM v0.27.1 ROCm stack with [libr4d](https://codeberg.org/StillDeadcode/libr4d)'s hand-written
RDNA4 attention, gated-delta-net, vision, all-reduce, and router kernels, while retaining Radiance's tuned
FP8 GEMM and speculative-decoding paths.

> **Status: experimental.** The fork pins the exact vLLM v0.27.1 tag commit and a qualified AMD
> ROCm compiler stack; see [docs/UPGRADE_PROGRESS.md](docs/UPGRADE_PROGRESS.md) and
> [docs/LIBR4D_BETTERBENCH.md](docs/LIBR4D_BETTERBENCH.md). The Docker image is published at
> [magiccodingman/vllm-radiance](https://hub.docker.com/r/magiccodingman/vllm-radiance).
> The current regression models are **Qwen3.8-27B FP8** and AMD's Quark **Qwen3.8-27B MXFP4**.
> Earlier Radiance work was measured on
> **Qwen3.6-27B-FP8**, **Qwen3.6-35B-A3B-FP8** (fine-grained MoE, 256 experts / top-8), and
> **Gemma-4-31B-it-FP8** (block-fp8, sliding + global attention, vision), all with fp8 (or bf16/`auto`) KV
> cache on two R9700 GPUs (tensor parallel). Other models, non-FP8 weights, single or
> 3+ GPUs, and non-R9700 hardware are untested. Expect rough edges and breaking changes. Not production
> hardened. Use at your own risk.

This is the `magiccodingman/vllm-radiance` fork. It tracks and credits DeadCode's upstream
[vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) and libr4d releases, but deliberately
diverges by carrying a selective DFlash2 backport plus the AMD PyTorch 2.12 and Triton 3.7.1 compiler pair.
The current line uses stable vLLM v0.27.1 plus the exact reviewed DFlash2 PR #52816 head rather than a moving post-release tree,
after the latter was shown to intermittently violate required JSON-schema fields during tool calls.
The GitLab pipeline publishes this fork as `magiccodingman/vllm-radiance`; the immutable upstream
`stilldeadcode/vllm-radiance` images are external comparison baselines, not products of this repository.

## Build

Everything the build needs is in this directory (a flat Docker build context). The version string lives in
one place, the `VERSION` file, which the tag and the build-arg both read:

```bash
docker build -t vllm-radiance:$(cat VERSION) --build-arg RADIANCE_VERSION=$(cat VERSION) .
```

That single command builds everything from source, in four stages. **builder** compiles PyTorch, Triton,
torchvision, AITER, and vLLM for `PYTORCH_ROCM_ARCH=gfx1201` against the official `rocm/dev-ubuntu-24.04`
base (digest-pinned) and leaves the wheels in `/wheels` (it also builds `rocm-bandwidth-test` for the startup
sweep). **rocmprune** (`prune_rocm.sh`) cuts the 19 GB ROCm tree down to this one GPU architecture.
**assemble** installs the wheels, applies the guarded RDNA4 patches, and builds the exact post-v0.4.0 libr4d
commit containing the MXFP4 GDN exponent guards
with the image's own `hipcc`. **final** is the release image: a clean `ubuntu:24.04` that receives only the pruned ROCm tree, the
venv, and the entrypoint, so neither the build toolchain nor the wheels ever reach the published image. No
prebuilt component wheels, no rotating wheel indexes, and no checked-in binaries go into the image. It is a
long build (a full PyTorch compile); expect it to run for hours on a many-core box.

For ordinary guarded patch, Radiance kernel, hook, config, or entrypoint iteration, layer
`Dockerfile.patch` over an already-built immutable stack instead of recompiling
the compiler stack:

```bash
docker build -f Dockerfile.patch \
  --build-arg BASE_IMAGE=vllm-radiance:0.8.0-dev.vllm0.27.1-r4d0.4.0-mxfp4.dflash2.fusedsplitk \
  --build-arg RADIANCE_VERSION=0.8.0-dev.patch \
  -t vllm-radiance:0.8.0-dev.patch .
```

The overlay reruns every source-drift guard and import check and rebuilds the
small Radiance/libr4d HIP extensions with the base image's pinned compiler. It
does not replace the torch/Triton/AITER/vLLM wheels. Use the full `Dockerfile`
when the compiler stack or one of those dependency wheels changes.

That structure is what keeps the download reasonable: the stock ROCm base is 7.4 GiB compressed on its own,
most of it device code for GPUs this image cannot run on. Pruning it to gfx1201 and shipping an allowlist
takes the image from 9.35 GiB compressed to **3.66 GiB**. Note the prune must happen in a stage the release
stage copies *from* -- deleting files in a layer stacked on the base reclaims nothing.

The release stage still ships a working **compiler** (hipcc, g++, and the C++/Python headers). That is
not slack to trim: AITER JIT-compiles its kernels on first use, inside the running container, so an
image without those headers boots and then dies on the first AITER module build. The build asserts it
by compiling and importing a pybind11 HIP module.

The component and source pins are the `ARG`s at the top of `Dockerfile`: exact commits for vLLM, AMD
PyTorch, AMD Triton, AITER, and libr4d plus asserted package versions. Builds fail if the checkout,
reported package version, or libr4d tag differs. To use this fork without building, pull
`magiccodingman/vllm-radiance:latest` after the corresponding release pipeline completes.

**Do not bump torch / triton / torchvision on their own.** They are not independent choices: vLLM pins the
torch version it is tested against, torch pins its triton, and torchvision ships a matching release. The
build runs vLLM's own `use_existing_torch.py`, which *strips* those pins — but that exists so pip does not
re-download torch, not as licence to install a newer one. Builds 0.5.0 through 0.5.4 compiled against a
newer trio and hung a GPU under sustained tensor-parallel load; restoring the pinned versions fixed it with
no code change. If you override these with `--build-arg`, move them together and soak-test under real load.

## Run

`docker-compose.yml` is the canonical way to serve. Start from the portable
environment template:

```bash
cp .env.example .env
# Edit MODELS, MODEL_PATH, and SERVED_MODEL_NAME in the private .env.
mkdir -p vllm-cache
docker compose up -d          # start; follow with: docker compose logs -f
docker compose down           # stop
```

The public Compose contains no machine-local paths or group IDs. Its portable
fallbacks are `./models`, `./vllm-cache`, `/models/model`, and the published
`magiccodingman/vllm-radiance:latest` image. The tuned starting envelope uses
native FP8 weights, mandatory FP8 KV, TP2, a 16K maximum length, an eight-request
admission ceiling, 85% GPU allocation, and automatic prefix caching with hybrid
GDN state alignment. Speculative decoding remains disabled for the
correctness-qualified baseline. Runtime and capacity settings, including
speculative decoding, can be overridden in `.env` without editing the public
file.

### Optional Quark MXFP4 / native W4A8 target

For [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4),
point `MODEL_PATH` at the checkpoint and use this `.env` profile:

```dotenv
WEIGHT_QUANTIZATION=auto
RADIANCE_MXFP4=1
RADIANCE_MXFP4_W4A8=1
RADIANCE_MXFP4_W4A8_MIN_M=0
RADIANCE_MXFP4_DECODE_MAX_M=48
RADIANCE_MXFP4_TN4_MIN_M=2048
# MTP only: this checkpoint stores its embedded MTP tensors in BF16.
RADIANCE_QUARK_BF16_MTP=1
```

`auto` is required: it lets vLLM consume the checkpoint's `quant_method: quark`
metadata instead of forcing the normal FP8 loader. The two feature switches are
opt-in and inert for ordinary FP8 checkpoints. On gfx1201, W4A8 keeps the packed
MXFP4 weights but dynamically quantizes activations to FP8 E4M3 so the hand-written
kernel reaches RDNA4's native FP8 WMMA path. This is more precise than the declared
W4A4 activation format, but it is a numerical mode change rather than bit-identical
emulation.

Keep `RADIANCE_MXFP4_W4A8_MIN_M=0`: the generic AITER W4A4 fallback is numerically
wrong for the Qwen GDN `N=5120,K=3072` projection. The decode-shaped kernel handles
`M<=48`; the prefill kernel handles larger batches and selects its wider N tile from
`M>=2048`. Mandatory FP8 KV and the normal R4D attention/GDN/all-reduce paths remain
active. The AMD checkpoint also contains a one-layer BF16 MTP head. Its Quark
metadata does not exclude those unpacked tensors from the global FP4 recipe, so
the normal MTP profile additionally requires `RADIANCE_QUARK_BF16_MTP=1`.
DFlash2 still requires a compatible separate drafter and its own qualification.

The implementation, provenance, safety constraints, and exact benchmark run IDs
are recorded in [docs/MXFP4_W4A8_R9700.md](docs/MXFP4_W4A8_R9700.md).

For a measured long-context MXFP4 deployment on two 32 GiB R9700s, this profile
completed four simultaneous 128K requests while retaining automatic prefix
caching:

```dotenv
MODEL_PATH=/models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd
GPU_UTIL=0.90
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=4
WEIGHT_QUANTIZATION=auto
RADIANCE_MXFP4=1
RADIANCE_MXFP4_W4A8=1
RADIANCE_MXFP4_W4A8_MIN_M=0
RADIANCE_MXFP4_DECODE_MAX_M=48
RADIANCE_MXFP4_TN4_MIN_M=2048
PREFIX_CACHING_FLAG=--enable-prefix-caching
MAMBA_CACHE_MODE=align
VLLM_USE_V2_MODEL_RUNNER=1
RADIANCE_COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE"}'
RADIANCE_SPECULATIVE_CONFIG='{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger","num_speculative_tokens":5,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","max_model_len":131072}'
```

The engine exposed 576,001 GPU KV tokens (4.39 fully populated 128K
requests). The full-context C4 pressure run completed without OOM or
preemption and retained 5.17 GiB minimum physical headroom per GPU. Treat the
paths above as examples inside the `/models` mount; host paths belong only in a
private `.env` or developer overlay.

The default server preserves the checkpoint's full language-and-vision
capability; deployments that intentionally want text-only execution may add
vLLM's `--language-model-only` flag in a private override. Compose loads the
checkpoint's recommended sampling defaults with `GENERATION_CONFIG=auto` and
defaults Qwen3.8 reasoning to `xhigh`. Clients remain authoritative: explicit
sampling parameters and request-level `chat_template_kwargs.reasoning_effort`
override those defaults.

Compose also uses the checkpoint-native chat template. This is the safest
generic default for a multi-model image and is the configuration used by the
tool-schema regression gate. A deployment-specific template such as
`qwen3.8-enhanced.jinja` remains available, but must be added explicitly with
`--chat-template` in a private Compose override.

### Choose and switch serving modes

The modes are mutually exclusive: `RADIANCE_SPECULATIVE_CONFIG` must contain
either one speculative method or be unset. They are not cumulative.

| Mode | `RADIANCE_SPECULATIVE_CONFIG` | Separate drafter | Profile-specific overrides |
|---|---|---|---|
| Qualified non-spec | unset | no | none |
| Fast MTP | `"method":"mtp"` | no; head is in the target | `RADIANCE_FAST_DRAFT=1` |
| Experimental DFlash2 | `"method":"dflash"` | yes | V2 runner, `PIECEWISE`, matched target/drafter context, draft TP2 |

To switch from MTP to DFlash2, replace—not append—the speculative JSON, remove
`RADIANCE_FAST_DRAFT=1`, and add the DFlash2-only variables in the DFlash2 block
below. To switch from DFlash2 to MTP, delete
`VLLM_USE_V2_MODEL_RUNNER`, `RADIANCE_COMPILATION_CONFIG`,
`PREFIX_CACHING_FLAG`, and `MAMBA_CACHE_MODE`, restore the desired
`MAX_MODEL_LEN` (16K is the portable baseline), replace the speculative JSON
with the MTP value, and set `RADIANCE_FAST_DRAFT=1`. To return to non-spec,
remove both speculative and fast-draft variables plus all DFlash2-only
overrides.

Recreate the container after every mode change so the new environment and
server arguments are applied:

```bash
docker compose down
docker compose up -d
docker compose logs -f
```

### Optional fast MTP

Qwen checkpoints with an in-checkpoint MTP head can enable the measured R4D
fast-MTP path entirely from the private `.env`:

```dotenv
RADIANCE_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D","disable_padded_drafter_batch":true}'
RADIANCE_FAST_DRAFT=1
```

`RADIANCE_SPECULATIVE_CONFIG` is passed through as one vLLM
`--speculative-config` argument by the image entrypoint. An explicit command-line
flag takes precedence. K8 is a ceiling: the enabled dynamic-draft controller
selects a shallower depth when confidence or active batch size warrants it. The
R4D backend is the qualified Qwen head-256 route; Gemma-4's separate head-512
assistant must continue to use `ROCM_AITER_UNIFIED_ATTN`, as documented below.
Fast MTP remains opt-in because the current strict cross-mode output gate has not
qualified; the BetterBench result is a performance measurement, not a default-use
correctness claim.

### Optional DFlash2/V2 drafter

The measured high-throughput DFlash2 K7 profile is also selectable entirely
from `.env`. Place the selective-FP8 drafter beneath the mounted `MODELS`
directory (Hugging Face repository
`magiccodingman/Qwen3.8-27B-heretic-ara-DFlash2-fp8`), adjust its in-container
path if necessary, and use the deployment-oriented cache settings below:

```dotenv
MAX_MODEL_LEN=8192
PREFIX_CACHING_FLAG=--enable-prefix-caching
MAMBA_CACHE_MODE=align
VLLM_USE_V2_MODEL_RUNNER=1
RADIANCE_COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE"}'
RADIANCE_SPECULATIVE_CONFIG='{"method":"dflash","model":"/models/Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","max_model_len":8192}'
```

For the AMD Quark MXFP4 target, use the target-matched
`tcclaviger/Qwen3.8-27B-DFlash2-FP8` drafter instead of the ARA drafter and set
its mounted path in the same JSON. This checkpoint uses block-FP8 QKV weights;
the image's DFlash2 backport applies their scales when constructing the fused
context-K/V projection while leaving normal draft execution quantized.

This selects the V2-compatible runner, the measured `PIECEWISE` graph mode,
TP2 draft sharding, and `TRITON_ATTN` for the drafter while retaining R4D for
the target. Prefix caching remains enabled for production because repeated
system/RAG/agent histories benefit enormously. The published BetterBench lanes
explicitly disabled it only to enforce cold, nonce-disjoint prompts; set
`PREFIX_CACHING_FLAG=--no-enable-prefix-caching` and `MAMBA_CACHE_MODE=none`
only when reproducing that benchmark contract or deliberately trading repeated-
prefix latency for the largest possible context envelope. On the original ARA
FP8 target, K7 was the measured c1/c2/c4/c8 winner. On the AMD Quark MXFP4
target below, K7 wins weighted single stream and c1-c4, while K5 is decisively
faster at c8. DFlash2 is experimental and failed the strict cross-mode gate, so
either profile is intentionally an explicit alternative rather than the
default.

For source development, copy `docker-compose.dev.example.yml` to
`docker-compose.dev.yml` and layer it explicitly:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

The developer overlay and `.env` are both gitignored and excluded from Docker
build contexts. The release pipeline fails closed if a developer overlay is
ever force-added, preventing local paths and image tags from reaching a release.

### Context and concurrency capacity

`MAX_NUM_SEQS` is an admission ceiling, not a guarantee that every admitted
request can remain fully resident at `MAX_MODEL_LEN`. Prefix caching uses the
same evictable GPU block pool as active KV; on Qwen's hybrid GDN, `align` mode
also retains restorable convolution/recurrent state and reconciles page sizes.
It can therefore reduce the absolute maximum context slightly while removing
most repeated-prefill work. `PIECEWISE` graphs also consume a bounded memory
reservation. The tables below keep those tradeoffs explicit instead of mixing
unlike profiles.

#### Native-FP8 target: cold-capacity profile

These conservative pairs were measured on two 32 GiB R9700s with the 28.75 GiB
native-FP8 Qwen3.8-27B target, FP8 KV, the 2.34 GiB selective-FP8 DFlash2 K5
drafter, TP2 `PIECEWISE` graphs, 85% GPU allocation, no CPU/KV offload, and
prefix caching disabled for the capacity laboratory:

| Maximum context per request | Suggested `MAX_NUM_SEQS` | Validated simultaneous submissions |
|---:|---:|---:|
| 8K | 8 | 8 |
| 16K | 7 | 8 |
| 32K | 5 | 6 |
| 64K | 3 | 3 |
| 128K | 2 | 2 |
| 256K | 1 | 1 |

All submissions completed. The larger 16K/32K bursts use normal scheduler
queueing; the suggested values stay below the engine-reported fully resident KV
capacity. Minimum observed physical headroom was 4.41 GiB per GPU. These values
are deliberately model/profile-specific—larger models, BF16 weights/KV,
different graph modes, or a different drafter need their own capacity check.
DFlash2 itself remains experimental because strict greedy equivalence has not
qualified; the table proves memory/serving capacity, not losslessness. See
[docs/COMPOSE_CAPACITY.md](docs/COMPOSE_CAPACITY.md) for exact runs and method.

#### Quark MXFP4/W4A8 target: measured capacity

AMD's MXFP4 target safetensors are 18.44 GiB versus 28.75 GiB for the native-FP8
regression target—a 10.31 GiB / 35.9% smaller target payload. With the 1.97 GiB
target-matched DFlash2 drafter, FP8 KV, TP2 `PIECEWISE` graphs, 90% GPU
allocation, and no offload, the prefix-disabled high-capacity sweep completed:

| Maximum context per request | Conservative production C | Highest completed burst |
|---:|---:|---:|
| 32K | 8 | 11 |
| 64K | 6 | 7 |
| 128K | 4 | 4 |
| 256K | 2 | 2 |

The conservative column deliberately stays below the observed boundary at
32K/64K. Every request completed without OOM; minimum physical headroom was
5.06 GiB per GPU. The 256K/C2 profile is valid, but setting
`MAX_NUM_SEQS=4` beside a 256K ceiling does **not** create room for four full
256K requests—it merely allows four shorter requests to be admitted and may
queue or preempt as they grow.

For repeated agent/RAG/chat workloads, the recommended profile is prefix caching
enabled, `MAMBA_CACHE_MODE=align`, `MAX_MODEL_LEN=131072`, and
`MAX_NUM_SEQS=4`. That exact MXFP4+DFlash K5 configuration reported 576,001 KV
tokens / 4.39x 128K capacity and completed a disjoint four-request 128K pressure
run with zero failures, OOMs, or preemptions and 5.17 GiB minimum physical
headroom. A 32K shared-prefix probe reduced TTFT from 9.04 seconds cold to
0.70-0.71 seconds warm (12.8x), reused 93.7% of the prompt, and produced
byte-identical sequential cold/warm outputs. Subsequent live agent turns reused
96.2% of newly queried prefix tokens.

The public Compose already defaults to `--enable-prefix-caching` plus
`--mamba-cache-mode=align`. Disable those only for a cold-cache benchmark or a
deliberate maximum-context experiment, then requalify the chosen envelope.

To serve the fine-grained-MoE **Qwen3.6-35B-A3B-FP8**, point it
at that model and keep the batch-token budget at **≥ 2240** (align mode
reconciles the GDN state to attention block size 2240; the Compose default is 4096). Its tuned
MoE config and the `RADIANCE_MOE_ROUTER` gate GEMM are baked in and turn on automatically.

To serve **Gemma-4-31B-it-FP8** (block-fp8, e.g. `RedHatAI/gemma-4-31B-it-FP8-block`), just point the compose
at it: the quantization is auto-detected from `config.json` (compressed-tensors, 128x128 blocks), the tuned
GEMM configs load by shape, and the long-context prefill attention path is tuned for its head-512 global
layers. Drop the Qwen-specific `--mamba-cache-mode` and chat template / tool-reasoning parsers (it is not a
GDN hybrid and uses its own template). Its vision tower works as-is. Note it is a *big-KV* model (60 layers,
50 sliding + 10 global), so give it a smaller `--max-model-len` than the Qwen models at the same
`--gpu-memory-utilization`.

Gemma-4-31B also supports **MTP speculative decoding** for a large decode speedup, using Google's official
drafter `google/gemma-4-31B-it-assistant` (vLLM loads it as an MTP model). It is lossless (the target
verifies every drafted token) and the dynamic draft controller applies to it. Its one requirement on this
card: the drafter has a head-512 layer, so pass `"attention_backend":"ROCM_AITER_UNIFIED_ATTN"` in the
speculative config (the usual `flash_attn` caps at head 256). For example:
`--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' --no-async-scheduling`.

All tunables are `${VAR:-default}` in the compose file; override via the shell,
a private `.env`, or an ignored developer overlay without editing it. The full
knob list (kernel toggles, draft controller, AITER routing, …) is in
[DOCKERHUB.md](DOCKERHUB.md).

## Measured performance

The headline FP8-versus-MXFP4 operating points are summarized here. These are
two separately published target checkpoints with their matching drafters, not a
bit-identical quantization A/B, so the table describes deployable capability
rather than attributing every delta to weight format alone:

| Target | Target payload | Weighted non-spec | Weighted DFlash K5 | DFlash K5 c1 / c4 / c8 | Long-context capacity checkpoint |
|---|---:|---:|---:|---:|---|
| Native FP8 Qwen3.8-27B | 28.75 GiB | 35.8 TPS | 99.4 TPS | 93.3 / 293.6 / 451.6 TPS | 128K C2, APC off, 85% GPU |
| AMD Quark MXFP4/W4A8 Qwen3.8-27B | 18.44 GiB | 43.7 TPS | 129.8 TPS | 114.8 / 341.0 / 506.7 TPS | 128K C4, APC+align on, 90% GPU |

The MXFP4 production checkpoint also measured 355.8 output TPS at C4 on a
disjoint 512-input/512-output serving control while retaining the 128K/C4
envelope. Full-context output TPS is intentionally much lower because it
includes four enormous cold prefills; use the BetterBench decode numbers above
for generation throughput and the capacity section for residency.

BetterBench v0.2.2 (`575cc3925bac922d6ad4a39e62502673799979d9`) used its v1 corpus,
10 measured passes per category, greedy decoding, cold nonce-prefixed prompts,
and c1/c2/c4/c8 on two R9700s. All fork lanes used the same native-FP8 target,
FP8 KV, TP2, 8K envelope, 85% allocation, and 4,096-token scheduler budget:

| Mode | Weighted single-stream decode | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| R4D non-spec | 35.8 | 35.5 | 67.0 | 118.4 | 187.6 |
| R4D MTP K8 + INT2 exact-rerank head | 93.7 | 82.8 | 149.0 | 242.7 | 316.3 |
| R4D DFlash2 K5 | 99.4 | 93.3 | 169.6 | 293.6 | 451.6 |
| R4D DFlash2 K7 | **112.6** | **102.1** | **189.0** | **305.1** | **496.1** |

The c1-c8 columns are aggregate output tok/s from 24 requests per level. DFlash2
K7 is the performance winner and retained at least 6.47 GiB physical VRAM
headroom per card, but it is **not the production default**: strict fixed-prompt
greedy equivalence against matched non-spec passed only 3/8 prompts. Repeated
K5, K7, and MTP runs were deterministic and produced the same fixed outputs,
which characterizes a shared speculative/runner numerical path difference but
does not turn a failed strict gate into a pass. Exact run IDs and the external
DeadCode 0.7.4 comparison are in
[docs/LIBR4D_BETTERBENCH.md](docs/LIBR4D_BETTERBENCH.md).

### Quark MXFP4/W4A8 on dual R9700

The same BetterBench standard contract was rerun with
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, the native gfx1201 W4A8 kernel, mandatory
FP8 KV, and the target-matched `tcclaviger/Qwen3.8-27B-DFlash2-FP8` drafter.
The rows below are category medians, not a single aggregate repeated eight
times:

| Category | Non-spec | MTP K4 fast | DFlash K5 | DFlash K7 |
|---|---:|---:|---:|---:|
| Chat | 43.9 | 84.3 | 97.9 | 97.3 |
| Code | 43.7 | 98.4 | 130.0 | **144.0** |
| File edit | 43.7 | 119.0 | 137.4 | **168.7** |
| JSON | 43.9 | 130.8 | 158.4 | **180.3** |
| Math | 43.5 | 116.2 | 157.8 | **176.9** |
| Prose | 43.6 | 80.7 | **107.8** | 101.0 |
| Reasoning | 43.5 | 85.3 | **110.1** | 107.6 |
| Summarization | 43.6 | 117.6 | 152.0 | **160.4** |
| **Weighted** | **43.7** | **101.9** | **129.8** | **139.8** |

| Mode | c1 | c2 | c4 | c8 | Draft acceptance | Minimum headroom/GPU |
|---|---:|---:|---:|---:|---:|---:|
| Non-spec | 43.5 | 83.3 | 144.6 | 243.0 | — | 4.91 GiB |
| MTP K4 fast | 97.5 | 173.6 | 277.9 | 365.3 | 59.25% | 5.54 GiB |
| DFlash K5 | 114.8 | 219.3 | 341.0 | **506.7** | 54.28% | 6.55 GiB |
| DFlash K7 | **125.9** | **225.6** | **357.3** | 360.1 | 43.59% | 6.48 GiB |

K7 is the measured latency/c1-c4 profile; K5 is the measured c8 throughput
profile. K5 significantly beat MTP in every category. K7 significantly beat K5
for code, file edit, JSON, math, and summarization, while chat/prose/reasoning
were noise-equivalent. All three speculative modes completed the fixed fixture
but matched non-spec exactly on only 2/8 prompts, so these are experimental
performance profiles—not lossless/default qualification. Kernel provenance,
confidence intervals, prefill, telemetry, checksums, negative results, and exact
run IDs are in [docs/MXFP4_W4A8_R9700.md](docs/MXFP4_W4A8_R9700.md).

## What's inside

Everything below is baked into the image; the tuned paths are env-gated and on by default. See
**[DOCKERHUB.md](DOCKERHUB.md)** for the per-knob reference: every flag, its default, and what it does.

- **gfx1201 correctness and routing** (always on): deterministic GPU enumeration plus current vLLM's
  native RDNA4 AITER and sampler routing (without exposing CDNA-only CK/MFMA/ASM kernels),
  MTP drafter unpad + multimodal draft-mask alignment, tool-parser + `from_json` chat-template filter, and
  an attention LDS fit that shrinks the staged K/V tile into the R9700's 64 KiB shared memory for any head
  size and KV dtype (AITER sizes it for a larger LDS; without this, 2-byte KV at head 256 and fp8 KV at
  head 512 both abort at CUDA-graph capture).
- **libr4d 0.4.0 kernels**: R4D attention for prefill/decode, all-R4D gated-delta-net prefill/decode/spec
  state handling, a head-dim-72 vision flash kernel, exact TP2 all-reduce, rotated six-bit compressed
  all-reduce, and the router GEMM. The startup report proves every expected kernel resolves; measured AITER,
  FLA, and Triton paths remain available as per-operator fallbacks and controls.
- **Radiance FP8 target paths**: preshuffled block-FP8 GEMM selection, split-K alignment fixes, fused
  RMSNorm+quant, and guarded fallback dispatch. These remain because matched controls showed the generic
  upstream AITER linear route was 9-11% slower.
- **Quark MXFP4/W4A8 target path** (opt-in): native loading of packed OCP group-32 MXFP4 weights,
  gfx1201-specific AITER tiles, and hand-written FP8-WMMA GEMMs with separate prefill and small-M
  decode tilings. The decode kernel fuses its split-K reduction into the last arriving block, and
  the scale-fold table uses hardware-validated E4M3 subnormals so rare exponent deltas round instead
  of flushing weight groups to zero. It preserves the FP8/R4D paths above for their original
  checkpoints and operators.
- **Fine-grained MoE support** (e.g. Qwen3.6-35B-A3B): RDNA4-tuned fused-MoE Triton configs (always on;
  removes the stock config's `M>=96` cliff for a lower prefill TTFT, lossless), plus a custom bf16 MoE-gate
  GEMM (`RADIANCE_MOE_ROUTER`) for the `n` in `[6,16]` band that rocBLAS serves poorly. Both inert on
  models they do not apply to.
- **Dynamic MTP drafting**: a per-request confidence gate plus verbatim n-gram tail, with an opt-in
  `RADIANCE_FAST_DRAFT=1` INT2-g128 copy of the MTP head followed by exact top-32 reranking. Target
  verification remains in place, but this fork does not label the full speculative runner byte-equivalent
  while its strict cross-mode gate is failing.
- **Experimental DFlash2** selectively backported onto stable vLLM v0.27.1: Qwen3.8 local-convolution and
  candidate-selector support, scaled context-K/V materialization for block-FP8 draft QKV weights, TP2
  draft sharding, and piecewise graph execution. It stays explicit-only pending strict output qualification.
- **Prefix caching that works on the GDN hybrid** (enabled in the compose): hybrid models leave automatic
  prefix caching off by default, so it is turned on explicitly with `--enable-prefix-caching
  --mamba-cache-mode=align`. Align mode snapshots and restores the linear-attention (GDN) recurrent state at
  block boundaries (verified bit-identical to full recompute, including under MTP), giving a large TTFT drop
  on shared prefixes (system prompts, RAG, agentic context): 3.6x in the original qualification and 12.8x
  for a 32K shared prefix in the MXFP4+DFlash production profile.
- **Startup topology + bandwidth sweep** (`RADIANCE_RUN_BWTEST`, on by default): device list, P2P access
  matrix, NUMA distances, and peak uni/bidirectional copy bandwidth per agent pair, from a
  `rocm-bandwidth-test` compiled into the image. Backgrounded and about a second, so it never delays the
  serve. Set `0` to skip.
- **Optional NUMA pinning** (`--numa-bind`, off by default) for multi-NUMA-node hosts.

## Layout

Flat build context: runtime `radiance_*.py` modules, guarded `patch_*.py` transforms, FP8/MoE configs, chat
template, Dockerfiles, and portable Compose live at the repository root. The image fetches libr4d from its
exact public tag and commit and compiles `r4d.so`; no generated binary is checked in. `prune_rocm.sh`
self-checks the slim ROCm runtime, while the Makefile is a development helper for checking out and rebuilding
the exact libr4d pin against an existing image toolchain.
