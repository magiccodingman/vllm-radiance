# vllm-radiance

A vLLM inference server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It bundles a working
ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on
this card, plus RDNA4-tuned GEMM / attention / all-reduce paths and a dynamic MTP draft controller, so you
don't have to build the stack yourself.

> **Status: super early dev (v0.2.7). Experimental.** Everything here was built and measured on one exact
> setup: Qwen3.6-27B-FP8, fp8 (or bf16/`auto`) KV cache, two R9700 GPUs (tensor parallel). Other models,
> non-FP8 weights, single or 3+ GPUs, and non-R9700 hardware are untested. Expect rough edges and breaking
> changes. Not production hardened. Use at your own risk.

This repository is the **source** for the image published as `stilldeadcode/vllm-radiance` on Docker Hub.
See **[DOCKERHUB.md](DOCKERHUB.md)** for the full description, the complete environment-variable / knob
reference, tested configuration, and stack versions.

## Build

Everything the build needs is in this directory (a flat Docker build context):

```bash
docker build -t vllm-radiance:0.2.7 .
```

The build pins a working ROCm/PyTorch/Triton/vLLM combination, builds AITER from source for gfx1201, applies
the RDNA4 correctness patches, and bakes in the tuned kernels. `radiance_ar_ext.so`, `radiance_ar_quant_ext.so`,
`rocm-bandwidth-test`, and `radiance_p2p_probe` are prebuilt binaries; `radiance_ar_ext.hip` /
`radiance_ar_quant_ext.hip` are the all-reduce extension sources, kept here for transparency.

> ### ⚠️ Wheel pins rotate: bump them before building
> The vLLM and ROCm-SDK wheels come from prebuilt indexes (`wheels.vllm.ai/rocm`, `rocm.nightlies.amd.com`)
> that keep only their newest builds, so the two dated pins in the `Dockerfile` **stop resolving within days**
> and the build fails at the wheel step with *"no version of vllm==… your requirements are unsatisfiable"*.
> Before building, set both `ARG`s to a version **currently listed** on the index:
>
> ```bash
> # latest vLLM (ROCm): set ARG VLLM_VERSION
> curl -sL https://wheels.vllm.ai/rocm/vllm/ | grep -oiE 'vllm-[0-9][^"<> ]*' | sort -u | tail
> # latest ROCm SDK nightly: set ARG ROCM_SDK_VERSION (format 7.x.ya2026MMDD)
> curl -sL https://rocm.nightlies.amd.com/whl-multi-arch/rocm-sdk-libraries/ | grep -oE '7\.[0-9.]+a[0-9]+' | sort -u | tail
> ```
>
> then either edit the `ARG` defaults or pass `--build-arg VLLM_VERSION=… --build-arg ROCM_SDK_VERSION=…`.
> The checked-in defaults were current on 2026-07-17 and *will* age out. If you just want a known-good image
> without chasing nightlies, pull the published one instead: `docker pull stilldeadcode/vllm-radiance`.

## Run

`docker-compose.yml` is the canonical way to serve. Point it at your model directory and your GPU group GIDs,
then:

```bash
# put your model at ./models/Qwen/Qwen3.6-27B-FP8  (or set MODELS=/your/model/dir)
docker compose up -d          # start; follow with: docker compose logs -f
docker compose down           # stop
```

All tunables are `${VAR:-default}` in the compose file; override via the shell or a `.env` file without
editing it. The full knob list (kernel toggles, draft controller, AITER routing, …) is in
[DOCKERHUB.md](DOCKERHUB.md).

## What's inside

- **Correctness patches** to make vLLM run on gfx1201 (GPU enumeration, AITER enablement, native sampler
  fallback, MTP drafter unpad + multimodal draft-mask alignment, tool-parser + `from_json` filter). Always on.
- **Tuned kernels** (env-gated, on by default): preshuffled FP8 blockscale GEMM, RDNA4 unified-attention
  tiling (fp8 + bf16/`auto` KV), fused RMSNorm+quant, a fp16 matrix-core (WMMA) path for the gated-delta-net
  gram of hybrid linear-attention models (`RADIANCE_GDN_WMMA`, big prefill + cold-start win vs the stock fp32
  scalar path on RDNA4), and a P2P one-shot all-reduce for TP=2 (with an optional fp8-quantized payload for
  large prefill messages, `RADIANCE_AR_QUANT`).
- **Multimodal ViT attention** (`RADIANCE_VIT_FLASH`, on by default): a native head_dim-72 flash-attention
  kernel for the vision encoder — the vendor flash kernels (CK / AITER) have no gfx1201 device code, so this
  is ~1.5-2x faster than the SDPA fallback while preserving per-image / windowed attention. Only active when
  serving a vision model.
- **Dynamic MTP drafting** (`RADIANCE_DYNAMIC_DRAFT`): a per-request, per-slot confidence gate for speculative
  draft depth (draft while cumulative confidence holds, take free-win verbatim n-grams, else verify), with a
  batch-size schedule that caps serial forwards at concurrency. Tune with `RADIANCE_DRAFT_SCHEDULE` /
  `RADIANCE_DRAFT_TAU`. Lossless (changes only how many tokens are drafted, never what the model verifies).
- **Optional NUMA pinning** (`RADIANCE_NUMA_BIND` / `--numa-bind`, off by default): on multi-socket or
  multi-NUMA-node hosts, pin the server and its TP workers to the NUMA node(s) local to the GPUs (`auto`
  detects from the visible GPUs; or give explicit nodes / `interleave` / `preferred=`). A no-op on single-node
  hosts; needs `--cap-add SYS_NICE` under Docker's default seccomp. See [DOCKERHUB.md](DOCKERHUB.md).

## Layout

Flat build context: the runtime Python modules (`radiance_*.py`), the `patch_*.py` fixes, the `fp8-configs/`
GEMM configs, the prebuilt binaries, the chat template, `Dockerfile`, and `docker-compose.yml` all live at the
repo root so `docker build .` works directly.
