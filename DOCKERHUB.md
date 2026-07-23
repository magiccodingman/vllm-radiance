# vllm-radiance

vLLM inference server for the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4). Bundles a working ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on this card, so you don't have to build the stack yourself.

> **Status: super early dev (v0.3.0). Experimental.**
> This is a very early build. The performance numbers here come from two exact configurations: Qwen3.6-27B-FP8 and Qwen3.6-35B-A3B-FP8 (fine-grained MoE), both with fp8 KV cache on two R9700 GPUs (tensor parallel); bf16 / `auto` KV also works (see below). Other models, non-FP8 weights, single or 3+ GPUs, and non-R9700 hardware are untested. Expect rough edges, breaking changes between versions, and things that just don't work yet. Not production hardened. Use at your own risk.

## Tested so far

Two setups have been run and measured:

| | |
|---|---|
| Models | **Qwen3.6-27B-FP8** (dense) and **Qwen3.6-35B-A3B-FP8** (fine-grained MoE: 256 experts, top-8) |
| KV cache | FP8 (bf16 / `auto` also supported) |
| GPUs | 2x R9700, tensor parallel (TP=2) |

Untested (may or may not work): any other model, non-FP8 weights, single GPU, more than two GPUs, non-R9700 hardware. Treat the defaults below as a starting point for these two setups, not a general recommendation.

**Fine-grained MoE (Qwen3.6-35B-A3B-FP8).** Supported and tuned. Two MoE paths are baked in and activate automatically for it: RDNA4-tuned fused-MoE Triton configs (removes the stock config's `M>=96` cliff, lower prefill TTFT, lossless) and a custom bf16 MoE-gate GEMM (`RADIANCE_MOE_ROUTER`, on by default) for the skinny `n` in `[6,16]` band that rocBLAS serves poorly. One serving requirement: with `--mamba-cache-mode=align` this model's attention block size is 2240, and align asserts `block_size <= max_num_batched_tokens`, so pass **`--max-num-batched-tokens >= 2240`** (2560 is a clean default; the compose ships 2048, which is fine for the 27B but must be raised for the 35B).

**bf16 / `auto` KV cache is supported** (0.1.5). Earlier builds crashed at startup with a Triton shared-memory (`OutOfResources`) error on any 2-byte KV cache (including the `--kv-cache-dtype auto` default), because the attention decode kernel's tile didn't fit the R9700's 64 KiB LDS at head_size 256. That kernel is now fixed and tuned for RDNA4, so fp8, bf16, and `auto` KV caches all work. fp8 packs more KV per GB of VRAM; bf16 / `auto` keeps full KV precision.

## Why this exists

vLLM's ROCm builds target datacenter cards (MI300 / CDNA). RDNA4 workstation cards like the R9700 (gfx1201) don't work out of the box: AITER isn't enabled for the arch, GPU enumeration fails, several kernels need patching, and the vendor attention and GEMM paths aren't tuned for RDNA4. This image pins a working combination, builds AITER from source for gfx1201, applies the fixes, and adds tuned kernels.

## Stack

| Component | Version |
|---|---|
| vLLM | 0.25.1 (ROCm) |
| PyTorch | 2.11 (ROCm 7.2) |
| Triton | 3.6 |
| AITER | 0.1.16, built from source for gfx1201 |
| flash-attention | 2.8.3 |
| ROCm userspace | 7.2, bundled |
| Base | Ubuntu 24.04, Python 3.12 |

## What it patches (to make vLLM run on gfx1201)

- GPU enumeration (amdsmi init order). Without it the platform is undetected and device count reads 0.
- AITER enablement for gfx12x (upstream gates it to MI3xx).
- Triton driver activation for the GPU-less model-inspection subprocess.
- Native sampler fallback (AITER's top-k/top-p kernel doesn't build on RDNA4).
- Tool-parser streaming vs non-streaming consistency.
- `from_json` Jinja filter for tool-calling chat templates.
- MTP drafter unpadding, so `--speculative-config`'s `disable_padded_drafter_batch:true` works (the single-stream MTP speed path).
- MTP drafter multimodal mask alignment, so speculative decoding works with image inputs (otherwise the vision-placeholder mask outlives the compacted draft batch and the engine crashes).

## Custom kernels and tuning (on by default, env-gated)

| Env var | Default | What it does |
|---|---|---|
| `RADIANCE_PRESHUFFLE` | `1` | preshuffled AITER FP8 blockscale GEMM |
| `RADIANCE_ATTN_TUNE` | `1` | RDNA4 attention tiling, gain grows with context length |
| `RADIANCE_GDN_WMMA` | `1` | for hybrid gated-delta-net (linear-attention) models, runs the KKt gram on the fp16 matrix cores (WMMA) instead of an fp32 scalar path. RDNA4 has no fp32 matrix-core path, so the stock kernel is both slow to run and very slow to compile (dominates cold start); the WMMA path is far faster on both. fp16 matches the precision of the TF32 path these models run on NVIDIA. A no-op for pure-transformer models. |
| `RADIANCE_VIT_FLASH` | `1` | native head_dim-72 flash-attention for the multimodal vision encoder (ViT). On RDNA4 the vendor flash kernels (CK / AITER) have no device code and torch SDPA runs a non-tiled path; this kernel handles the vision tower's odd head dim without padding and runs ~1.5-2x faster. Only used when serving a vision model; no effect for text-only. Per-image / windowed attention is preserved. |
| `RADIANCE_FAST_REDUCE` | `1` | custom PCIe peer-to-peer all-reduce for TP=2, byte-identical to RCCL, falls back to RCCL if P2P is unavailable |
| `RADIANCE_AR_MAX_KB` | `32768` | size gate for the P2P all-reduce, in KB (32768 = 32 MB); messages above it use RCCL |
| `RADIANCE_AR_QUANT` | `1` | quantize the all-reduce payload to block-scaled fp8 (e4m3) for large messages, halving the PCIe bytes. Speeds up prefill; leaves decode untouched. NOT bit-identical to RCCL (it is quantized). On by default; set `0` for the exact bf16 all-reduce. |
| `RADIANCE_AR_QUANT_MIN_KB` | `128` | when `RADIANCE_AR_QUANT=1`, only messages at least this large take the fp8 path; smaller ones keep the exact bf16 all-reduce (fp8 only pays off once the transfer is bandwidth-bound) |
| `RADIANCE_FUSE_RMS_QUANT` | `1` | folds group-FP8 quant into the RMSNorm epilogue |
| `RADIANCE_DYNAMIC_DRAFT` | `1` | **dynamic** MTP draft depth: per request, a per-slot confidence gate decides how deep to draft (up to `num_speculative_tokens`) and whether to take a verbatim n-gram continuation (deep on high-acceptance content like code and JSON, shallow on prose; see "Speculative decoding" below). Lossless. Needs `--speculative-config method=mtp`. |
| `RADIANCE_DRAFT_SCHEDULE` | `1:8,2:7,4:6,8:5,16:4` | `bs:max_depth` pairs (carry-forward): caps how many serial MTP forwards run at each batch size, so drafting stays deep single-stream and shallower at concurrency. The free n-gram tail is unaffected. |
| `RADIANCE_DRAFT_TAU` | `0.35` | confidence-product stop threshold: the drafter keeps drafting while the running product of its top-1 confidences stays `>= TAU`. Lower = draft deeper, higher = shallower. |
| `RADIANCE_MOE_ROUTER` | `1` | for fine-grained MoE models (e.g. Qwen3.6-35B-A3B), routes the bf16 MoE-gate GEMM `x[n,2048] @ W[256,2048]^T` to a custom gfx1201 kernel for the `n` in `[6,16]` batch band that rocBLAS serves poorly (~2.5x faster cold, bit-identical output; wvSplitK already covers `n<=5`). A no-op for models whose gate is not `[256,2048]`. Set `0` for rocBLAS. |

All of these are baked ON in the image. Set `RADIANCE_DYNAMIC_DRAFT=0` to turn draft control off (`RADIANCE_AR_MAX_KB`, `RADIANCE_DRAFT_SCHEDULE`, and `RADIANCE_DRAFT_TAU` are values, not toggles). `RADIANCE_DYNAMIC_DRAFT` only does anything when speculative decoding is enabled; it is lossless (it changes only *how many* tokens are drafted and whether they come from MTP or a verbatim copy of earlier text, never what the model verifies).

**NUMA pinning (`RADIANCE_NUMA_BIND` / `--numa-bind`, opt-in, off by default).** On a multi-socket or multi-NUMA-node host, pin the server and its TP workers to the NUMA node(s) local to the GPUs so memory stays off the cross-node link. Set `RADIANCE_NUMA_BIND=auto` (detect from the visible GPUs) or pass `--numa-bind[=SPEC]` in the command; the flag wins. `SPEC` = `auto` \| explicit nodes (`0`, `0,1`) \| `bind=<nodes>` \| `interleave[=<nodes>]` \| `preferred=<node>` \| `none`. It is a no-op on single-node hosts and requires `--cap-add SYS_NICE` under Docker's default seccomp (already covered if you run `--security-opt seccomp=unconfined`).

## Requirements

- AMD Radeon AI PRO R9700 (gfx1201). Compiled for gfx1201 only, won't run on other GPUs. Two GPUs (TP=2) is the only configuration tested so far.
- Linux host with the amdgpu kernel driver and `/dev/kfd` + `/dev/dri`. ROCm userspace is inside the image.
- Docker with device passthrough.

## Run

On start the image prints a RADIANCE banner and runs a quick preamble (GPU count, gfx1201 check, P2P, enabled optimizations, component versions), then hands off to `vllm serve`. (An optional `rocm-bandwidth-test` topology/bandwidth sweep is available with `RADIANCE_RUN_BWTEST=1`; it is off by default.) First argument is the model path, the rest are `vllm serve` flags. The `RADIANCE_*` vars below are the custom optimizations (see the table above). They are already baked ON in the image; they are listed here so they are visible and easy to flip off.

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro \
  -v "$PWD/vllm-cache:/cache" \
  -p 8000:8000 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_PRESHUFFLE=1 \
  -e RADIANCE_ATTN_TUNE=1 \
  -e RADIANCE_GDN_WMMA=1 \
  -e RADIANCE_VIT_FLASH=1 \
  -e RADIANCE_FAST_REDUCE=1 \
  -e RADIANCE_AR_MAX_KB=32768 \
  -e RADIANCE_AR_QUANT=1 \
  -e RADIANCE_FUSE_RMS_QUANT=1 \
  -e RADIANCE_DYNAMIC_DRAFT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  stilldeadcode/vllm-radiance:0.2.8 \
    /models/YourOrg/Your-Model-FP8 \
    --served-model-name my-model \
    --quantization fp8 --kv-cache-dtype fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --enable-prefix-caching --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' \
    --no-async-scheduling \
    --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello!"}]}'
```

### compose

```yaml
services:
  vllm:
    image: stilldeadcode/vllm-radiance:0.2.8
    restart: unless-stopped
    command:
      - /models/YourOrg/Your-Model-FP8
      - --served-model-name=my-model
      - --quantization=fp8
      - --kv-cache-dtype=fp8
      - --tensor-parallel-size=2
      - --gpu-memory-utilization=0.92
      - --attention-backend=ROCM_AITER_UNIFIED_ATTN
      - --enable-prefix-caching
      - --mamba-cache-mode=align
      - '--speculative-config={"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'
      - --no-async-scheduling
      - --host=0.0.0.0
      - --port=8000
    environment:
      HIP_VISIBLE_DEVICES: "0,1"
      VLLM_ROCM_USE_AITER: "1"
      VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: "1"
      VLLM_ROCM_USE_AITER_MHA: "0"
      VLLM_ROCM_USE_AITER_MLA: "0"
      VLLM_ROCM_USE_AITER_MOE: "0"
      VLLM_ROCM_USE_AITER_LINEAR: "0"
      VLLM_ROCM_USE_AITER_FP8BMM: "0"
      VLLM_ROCM_USE_AITER_FP4BMM: "0"
      VLLM_ROCM_USE_AITER_RMSNORM: "0"
      NCCL_PROTO: Simple
      RADIANCE_PRESHUFFLE: "1"
      RADIANCE_ATTN_TUNE: "1"
      RADIANCE_GDN_WMMA: "1"
      RADIANCE_VIT_FLASH: "1"
      RADIANCE_FAST_REDUCE: "1"
      RADIANCE_AR_MAX_KB: "32768"
      RADIANCE_AR_QUANT: "1"
      RADIANCE_FUSE_RMS_QUANT: "1"
      RADIANCE_DYNAMIC_DRAFT: "1"
      # point the torch.compile / Triton / AITER caches at the mounted /cache so they persist
      # across restarts (without these the ./vllm-cache mount does nothing and every start re-autotunes)
      VLLM_CACHE_ROOT: /cache/vllm
      TORCHINDUCTOR_CACHE_DIR: /cache/inductor
      TRITON_CACHE_DIR: /cache/triton
      AITER_ROOT_DIR: /cache/aiter
      TRITON_CACHE_AUTOTUNING: "1"
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - "RENDER_GID"   # getent group render | cut -d: -f3
      - "VIDEO_GID"    # getent group video  | cut -d: -f3
    shm_size: "4gb"        # vLLM's TP workers share tensors via /dev/shm; the 64 MB default is too small
    cap_add: [SYS_PTRACE]
    security_opt: ["seccomp=unconfined"]
    ports: ["8000:8000"]
    volumes:
      - /path/to/models:/models:ro
      - ./vllm-cache:/cache   # persists the caches pointed at /cache above; first start is slow, restarts fast
```

## First run is slow

With an empty cache the first start spends about 15 to 20 minutes autotuning Triton kernels. It looks idle but it's compiling. Mount a persistent cache so restarts take about 1 to 2 minutes:

```bash
  -v /path/to/vllm-cache:/cache \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1
```

## Flags

| Flag | Suggested | Notes |
|---|---|---|
| `--tensor-parallel-size` | `2` | one rank per R9700 |
| `--quantization` | `fp8` | tuned for FP8 weights |
| `--kv-cache-dtype` | `fp8`, `bf16`, or `auto` | fp8 = 1 byte/elem (most KV capacity); bf16 / `auto` keep full precision |
| `--attention-backend` | `ROCM_AITER_UNIFIED_ATTN` | required for the tuned attention path |
| `--max-model-len` | model dependent | context length per request |
| `--max-num-seqs` | workload dependent | max concurrent sequences |
| `--gpu-memory-utilization` | `0.90` to `0.97` | VRAM fraction for weights + KV |
| `--enable-prefix-caching` | on for shared prefixes | enables automatic prefix caching; **required** — hybrid (GDN/mamba) models leave it off by default even though the engine default looks on |
| `--mamba-cache-mode` | `align` (hybrid models) | makes the linear-attention (GDN) layers prefix-cacheable; pair with `--enable-prefix-caching` on this hybrid. `none` disables mamba-layer caching; `all` is unsupported by this model |
| `--numa-bind` | omit (off) | multi-NUMA-node hosts only: pin the fleet to the GPU-local node(s). `auto` / `<nodes>` / `interleave` / `preferred=<n>` / `none`. Same as `RADIANCE_NUMA_BIND`; needs `--cap-add SYS_NICE`. See NUMA pinning above. |

Speculative decoding (models with MTP layers):

```
--speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'
```

**What `num_speculative_tokens` means here.** In stock vLLM it is a *fixed* draft length: every decode step drafts exactly that many tokens and verifies them. With `RADIANCE_DYNAMIC_DRAFT=1` (baked on) it becomes a **ceiling, not a fixed cost**: per request the controller drafts *up to* that many tokens, stops early on low-acceptance content, and may take a verbatim n-gram continuation when it matches the drafter's own guess, but the total draft is always clamped to `num_speculative_tokens`. So a larger value like **8** is the recommended default: it gives the dynamic drafter more room to run deep on high-acceptance content (code, JSON, boilerplate) without adding fixed overhead on prose. There is no separate depth-ceiling knob to keep in sync; the ceiling is `num_speculative_tokens` itself. (Set `RADIANCE_DYNAMIC_DRAFT=0` to get the classic fixed-length behavior, in which case a smaller value such as 3 is more typical.)

`disable_padded_drafter_batch:true` is the key single-stream lever (~+50% on Qwen3.6-27B): it drops the drafter's batch padding, and the image bakes the vLLM unpad patch this relies on. Leave it on. Note it is incompatible with async scheduling: pass `--no-async-scheduling` to disable it explicitly (otherwise vLLM auto-enables async scheduling and then disables it with a runtime warning; `--async-scheduling` would hard-error).

Prefix caching (shared system prompts, RAG, agentic context):

```
--enable-prefix-caching --mamba-cache-mode align
```

Automatic prefix caching reuses a shared prompt prefix across requests so only the new suffix is prefilled — a large time-to-first-token drop when many requests share a system prompt or document. On this **GDN hybrid you must pass both flags**: hybrid models default their prefix-caching support flag off ("experimental"), so vLLM **silently disables** prefix caching unless `--enable-prefix-caching` is given, and `--mamba-cache-mode align` is what makes the linear-attention (GDN) layers cacheable — it snapshots and restores their conv + recurrent state at block boundaries. That restore is **verified bit-identical to a full recompute** (including under MTP), so outputs are unchanged; the win is purely latency (measured ~3.6x faster TTFT on shared prefixes). Trade-offs: align reconciles the mamba and attention page sizes, which raises the attention block size to 1664 tokens and adds one state block per linear-attention layer (slightly lower max concurrency at full context), and prefix hits land on 1664-token boundaries. Do **not** use `--mamba-cache-mode all` (unsupported by this model, raises at startup) and do **not** set `VLLM_SSM_CONV_STATE_LAYOUT=DS` (asserts under MTP + align).

Tool-calling and reasoning:

```
--enable-auto-tool-choice --tool-call-parser <parser> --reasoning-parser <parser>
```

Pass a template with `--chat-template file.jinja` if the model needs one. The image ships the `from_json` filter those templates often rely on.
