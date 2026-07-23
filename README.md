# vllm-radiance

A vLLM inference server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It bundles a working
ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on
this card, plus RDNA4-tuned GEMM / attention / all-reduce paths and a dynamic MTP draft controller, so you
don't have to build the stack yourself.

> **Status: super early dev (v0.2.8). Experimental.** Everything here was built and measured on one exact
> setup: Qwen3.6-27B-FP8, fp8 (or bf16/`auto`) KV cache, two R9700 GPUs (tensor parallel). Other models,
> non-FP8 weights, single or 3+ GPUs, and non-R9700 hardware are untested. Expect rough edges and breaking
> changes. Not production hardened. Use at your own risk.

This repository is the **source** for the image published as `stilldeadcode/vllm-radiance` on Docker Hub.
See **[DOCKERHUB.md](DOCKERHUB.md)** for the full description, the complete environment-variable / knob
reference, tested configuration, and stack versions.

## Build

Everything the build needs is in this directory (a flat Docker build context). The version string lives in
one place, the `VERSION` file, which the tag and the build-arg both read:

```bash
docker build -t vllm-radiance:$(cat VERSION) --build-arg RADIANCE_VERSION=$(cat VERSION) .
```

The build pins a working ROCm/PyTorch/Triton/vLLM combination, builds AITER from source for gfx1201, applies
the RDNA4 correctness patches, and bakes in the tuned kernels. `radiance_ar_ext.so`, `radiance_ar_quant_ext.so`,
and `rocm-bandwidth-test` are prebuilt binaries; `radiance_ar_ext.hip` / `radiance_ar_quant_ext.hip` are the
all-reduce extension sources, kept here for transparency.

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

Everything below is baked into the image; the tuned paths are env-gated and on by default. See
**[DOCKERHUB.md](DOCKERHUB.md)** for the per-knob reference — every flag, its default, and what it does.

- **gfx1201 correctness patches** (always on): GPU enumeration, AITER enablement, native sampler fallback,
  MTP drafter unpad + multimodal draft-mask alignment, tool-parser + `from_json` chat-template filter.
- **RDNA4-tuned kernels**: preshuffled FP8 blockscale GEMM, unified-attention tiling (fp8 + bf16/`auto` KV),
  fused RMSNorm+quant, an fp16 matrix-core (WMMA) gated-delta-net path, a TP=2 P2P one-shot all-reduce
  (optional fp8 payload), and a native head_dim-72 ViT flash kernel for multimodal vision encoders.
- **Fine-grained MoE support** (e.g. Qwen3.6-35B-A3B): RDNA4-tuned fused-MoE Triton configs (always on;
  removes the stock config's `M>=96` cliff for a lower prefill TTFT, lossless), plus a custom bf16 MoE-gate
  GEMM (`RADIANCE_MOE_ROUTER`) for the `n` in `[6,16]` band that rocBLAS serves poorly. Both inert on
  models they do not apply to.
- **Lossless dynamic MTP drafting**: a per-request confidence gate plus verbatim n-gram tail that varies
  draft depth without changing what the model verifies.
- **Prefix caching that works on the GDN hybrid** (enabled in the compose): hybrid models leave automatic
  prefix caching off by default, so it is turned on explicitly with `--enable-prefix-caching
  --mamba-cache-mode=align`. Align mode snapshots and restores the linear-attention (GDN) recurrent state at
  block boundaries — verified bit-identical to full recompute, including under MTP — giving a large TTFT drop
  on shared prefixes (system prompts, RAG, agentic context).
- **Optional NUMA pinning** (`--numa-bind`, off by default) for multi-NUMA-node hosts.

## Layout

Flat build context: the runtime Python modules (`radiance_*.py`), the `patch_*.py` fixes, the `fp8-configs/`
and `moe-configs/` GEMM configs, the `router_gemm.hip` kernel source, the prebuilt binaries, the chat template,
`Dockerfile`, and `docker-compose.yml` all live at the repo root so `docker build .` works directly.
