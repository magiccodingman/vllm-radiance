# LFM3 text support

Status: **backport integrated; candidate-image and GPU qualification pending**.
This profile does not promote or replace the Qwen production deployment.

## Source ownership

Stable vLLM 0.33.0 predates LiquidAI's official LFM3 text implementation.
Radiance backports [vLLM PR 65826](https://github.com/vllm-project/vllm/pull/65826),
commit `c96e5f60bbdd7b523d56f6e7cb9b39e0a7013287` (merged 2026-11-06).
The original upstream patch, authorship, license headers and tests are retained
under `upstream/vllm-lfm3/`, with SHA-256 provenance. The build installs its six
runtime files through `patch_lfm3_backport.py` with exact context/checksum guards.
Every target is validated before writing any file. Reapplication is supported.

This includes native `Lfm3ForCausalLM`, the bundled `Lfm3Config`, model/config
registration and width-7 causal convolution in prefill and decode. The bundled
config reconciles the checkpoint's partial RoPE with the existing proportional
implementation. It does not require the checkpoint's Python model code.

Remove the backport when a pinned stable vLLM owns these changes upstream.
This is text support; VL, Audio and Omni architectures are outside this import.

## Model profile

The supplied `LiquidAI/LFM3-1.7B-FP8` checkpoint declares **compressed-tensors**
FP8 block weights and dynamic grouped activation quantization. Its exclusions
keep convolution and attention-gate projections unquantized. Use automatic
checkpoint quantization detection; forcing `--quantization=fp8` would select a
different loader. BF16 in the profile describes compute dtype, not replacement
of the checkpoint's serialized FP8 weights.

LFM3 has head dimension 128, GQA 4 and sliding-window attention. Radiance's R4D
backend is specialized for head dimension 256/GQA 6 and does not support sliding
windows. The optional profile therefore uses `TRITON_ATTN`, compute-dtype KV,
non-speculative TP1, bounded context and prefix caching off initially. This
keeps its qualification separate from Qwen/GDN/R4D and leaves those defaults
unchanged. Context capacity, FP8 KV, prefix-state reuse, TP2, LoRA and speculative
modes require their own evidence before promotion.

Set these in a **separate** environment file and use your existing ignored host
mount overlay; do not overwrite a production `.env`:

```dotenv
IMAGE=vllm-radiance:1.2.0-rc1.vllm0.33.0
MODEL_PATH=/models/LFM3-1.7B-FP8-LiquidAI
SERVED_MODEL_NAME=lfm3
TP=1
HIP_VISIBLE_DEVICES=0
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
GPU_UTIL=0.65
```

Review the resulting Compose configuration before launch. This command operates
on the selected Compose project; use a distinct project/container name and port
for development when another server is running:

```bash
docker compose --env-file .env.lfm3 \
  -f docker-compose.yml -f docker-compose.lfm3.yml \
  -f docker-compose.dev.yml config
```

`docker-compose.lfm3.yml` replaces the Qwen command and explicitly resets
quantization/speculative/MXFP4 controls. The common host overlay supplies mounts
and GPU access. Its Qwen-specific command overrides, if any, must be removed
for an LFM3 test. The profile uses the checkpoint's own chat template, the Qwen3
reasoning parser and Qwen3 Coder tool parser. Tool/streaming compatibility is a
live gate, not inferred from registration.

## Qualification

Run in the exact candidate image, with the repository mounted and pytest
available in the disposable test environment:

```bash
python benchmarks/bin/check_lfm3_backport_cpu.py
python benchmarks/bin/check_lfm3_image.py --model "$MODEL_PATH"
python -m pytest --noconftest upstream/vllm-lfm3/test_lfm3.py -q
# Physical GPU gate: expose only the authorized GPU to the test container.
python -m pytest --noconftest upstream/vllm-lfm3/test_causal_conv1d.py -q
```

The unchanged upstream convolution suite covers width 7 as well as existing
widths, prefill/decode state, chunk boundaries, padded/gathered requests, and
unsupported-width rejection. The CPU RoPE reference covers the unusual
partial-frequency layout, which a registry-only change would miss.

Follow with real FP8 checkpoint startup, short and >256-token prompts, batching,
chat, tool calls/tool results and streaming. Preserve source/version identity,
logs and responses in immutable scratch output. A clean source application or
CPU integrity check is not a claim of ROCm/model correctness.
