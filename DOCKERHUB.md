# vllm-radiance

vLLM inference server for the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4), combining pinned stable
vLLM v0.27.1 with libr4d's hand-written RDNA4 kernels and Radiance's FP8/speculative paths.

> **Status: experimental.** This fork publishes as `magiccodingman/vllm-radiance`.
> `stilldeadcode/vllm-radiance:0.7.4` is DeadCode's separate upstream release
> and the external comparison baseline. Current pins and qualification evidence
> are in `docs/UPGRADE_PROGRESS.md` and `docs/LIBR4D_BETTERBENCH.md`.

The current fork is validated primarily with
`Qwen3.8-27B-heretic-ara-fp8-magiccodingman` and
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, mandatory FP8 KV, and two R9700s
(TP2). Its reusable default is 16K, 85% GPU allocation, and at most eight
sequences; it deliberately leaves VRAM headroom instead of finding the largest
batch that fits. Earlier 0.5.8 performance numbers below remain useful history,
but are not claims about the new compiler stack.

## Current fork stack (`0.8.0-dev.vllm0.27.1-r4d0.4.0-mxfp4.dflash2.fusedsplitk`)

| Component | Exact version/pin |
|---|---|
| vLLM | `0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| PyTorch | AMD ROCm 2.12 commit `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5` |
| Triton | AMD 3.7.1 commit `f0b55c07da61c71775bef6d1a15ebf846430ac75` |
| torchvision | 0.27.1 |
| AITER | 0.1.20 (`fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`) |
| Transformers | 5.14.1 |
| XGrammar | 0.2.3 |
| libr4d | reports 0.4.0; fixed source `b9e42ab7202f53a3bc13d415f5d41481f9ca311b` |
| ROCm userspace | 7.14, bundled |

Use the repository `docker-compose.yml` for the current local image and model
defaults. Do not substitute an unpinned `main` checkout or generic PyTorch
2.13/upstream Triton pair.

### Current kernel and draft controls

| Env var | Default | Purpose |
|---|---:|---|
| `RADIANCE_USE_R4D` | `1` | master switch for the hand-written libr4d operator family |
| `RADIANCE_USE_R4D_GDN` | `1` | R4D GDN prefill, decode, and speculative-state paths |
| `RADIANCE_USE_R4D_AR` | `1` | exact TP2 P2P all-reduce, with safe RCCL fallback |
| `RADIANCE_USE_R4D_AR_QUANT` | `1` | rotated six-bit compressed collective for qualifying large messages; numerically lossy |
| `RADIANCE_AR_MAX_KB` | `49152` | largest P2P all-reduce payload; 48 MiB covers 4,096 x hidden-size-5,120 BF16 |
| `RADIANCE_AR_QUANT_MIN_KB` | `128` | minimum payload for rotated six-bit compression |
| `RADIANCE_R4D_REPORT` | `1` | report libr4d version and resolved kernel coverage at startup |
| `RADIANCE_PRESHUFFLE` | `1` | Radiance preshuffled block-FP8 target GEMM dispatcher |
| `RADIANCE_FUSE_RMS_QUANT` | `1` | fused RMSNorm plus FP8 quantization |
| `RADIANCE_DYNAMIC_DRAFT` | `1` | confidence- and batch-aware MTP depth controller |
| `RADIANCE_NGRAM_EXTENSION` | `1` | verbatim n-gram tail for speculative drafting |
| `RADIANCE_DRAFT_SCHEDULE` | `1:8,2:7,4:6,8:5,16:4` | MTP maximum depth by active batch size |
| `RADIANCE_DRAFT_TAU` | `0.28` | confidence-product stop threshold |
| `RADIANCE_FAST_DRAFT` | `0` | opt-in INT2-g128 MTP head with exact top-32 reranking |
| `RADIANCE_MXFP4` | `0` | allow native Quark/OCP MXFP4 routing on gfx1201 |
| `RADIANCE_MXFP4_W4A8` | `0` | packed MXFP4 weights with dynamic FP8 activations on RDNA4 WMMA |
| `RADIANCE_MXFP4_W4A8_MIN_M` | `0` | first M routed to W4A8; keep 0 for the qualified AMD checkpoint |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `48` | upper bound for the fused small-M split-K decode kernel |
| `RADIANCE_MXFP4_TN4_MIN_M` | `2048` | switch the prefill kernel to its wider N tile |
| `RADIANCE_QUARK_BF16_MTP` | `0` | load AMD's embedded BF16 MTP head outside the global Quark recipe |
| `RADIANCE_SPECULATIVE_CONFIG` | unset | raw vLLM speculative-config JSON appended by the entrypoint; explicit CLI config wins |
| `RADIANCE_COMPILATION_CONFIG` | unset | raw vLLM compilation-config JSON appended by the entrypoint; explicit CLI config wins |
| `RADIANCE_RUN_BWTEST` | `1` | background topology/P2P bandwidth report at startup |

R4D attention is selected with `--attention-backend=R4D`. AITER attention and
the older operator implementations remain fallbacks/controls. The portable
Compose defaults to native FP8 weights, FP8 KV, TP2, 16K, 85% allocation,
eight admitted sequences, 4,096 batched tokens, and automatic prefix caching
with `--mamba-cache-mode=align`. Speculative decoding is off because its strict
cross-mode output gate has not qualified. For Qwen's
in-checkpoint MTP head, opt in without editing Compose by adding these two lines
to `.env`:

```dotenv
RADIANCE_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D","disable_padded_drafter_batch":true}'
RADIANCE_FAST_DRAFT=1
```

The first enables R4D MTP with dynamic K8-as-a-ceiling drafting; the second
enables the INT2-g128 draft-head copy with exact top-32 reranking. An explicit
`--speculative-config` command argument overrides the environment value. Gemma-4
is the head-512 exception and must use the AITER backend shown below.

For the experimental DFlash2/V2 alternative, use the complete matching profile
below (replace the drafter path if its directory name differs):

```dotenv
MAX_MODEL_LEN=8192
PREFIX_CACHING_FLAG=--enable-prefix-caching
MAMBA_CACHE_MODE=align
VLLM_USE_V2_MODEL_RUNNER=1
RADIANCE_COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE"}'
RADIANCE_SPECULATIVE_CONFIG='{"method":"dflash","model":"/models/Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","max_model_len":8192}'
```

That is the measured selective-FP8 K7 lane: R4D target attention, Triton draft
attention, draft TP2, and piecewise graphs. It remains opt-in because strict
cross-mode equivalence has not qualified. Published BetterBench results used
`--no-enable-prefix-caching --mamba-cache-mode=none` only to enforce a cold,
nonce-disjoint benchmark contract; those are not the recommended production
defaults.

For the AMD Quark target, set `WEIGHT_QUANTIZATION=auto`, enable both MXFP4
switches above, and use the target-matched
`tcclaviger/Qwen3.8-27B-DFlash2-FP8` drafter path. Stable vLLM 0.27.1 contains
the older DFlash runtime; this image selectively backports Qwen3.8 DFlash2 from
the exact reviewed PR #52816 head and applies block scales to its fused FP8
context-K/V projection.

## Tested so far

Four target setups have been run and measured:

| | |
|---|---|
| Models | **Qwen3.8-27B-Quark-AWQ-MXFP4** (dense Quark MXFP4/W4A8), **Qwen3.6-27B-FP8** (dense), **Qwen3.6-35B-A3B-FP8** (fine-grained MoE: 256 experts, top-8), **Gemma-4-31B-it-FP8** (dense, sliding + global attention, vision) |
| KV cache | FP8 (bf16 / `auto` also supported) |
| GPUs | 2x R9700, tensor parallel (TP=2) |

Untested (may or may not work): any other model/quantization recipe, more than
two GPUs, or non-R9700 hardware. Treat the defaults below as a starting point
for these measured setups, not a general recommendation.

**Qwen3.8 Quark MXFP4/W4A8.** The final 10-pass/category BetterBench result was
43.7 weighted non-spec TPS, 101.9 with fast MTP, 129.8 with target-matched
DFlash K5, and 139.8 with K7. K7 won c1/c2/c4 at 125.9/225.6/357.3 TPS; K5
won c8 at 506.7 TPS versus K7's 360.1. All category rows, acceptance, prefill,
headroom, checksums, exact run IDs, and the failed strict-equivalence status are
in `docs/MXFP4_W4A8_R9700.md` in the source repository.

The recommended long-context MXFP4 profile is 128K/C4 at 90% GPU allocation,
FP8 KV, DFlash K5, `PIECEWISE` graphs, prefix caching, and GDN `align` state. It
reported 576,001 KV tokens / 4.39x full-request capacity and completed four
disjoint full-context requests with zero OOM or preemption and 5.17 GiB minimum
headroom per GPU. A 32K repeated prefix reduced TTFT from 9.04 seconds cold to
0.70-0.71 seconds warm (12.8x) with byte-identical sequential outputs. The
prefix-disabled maximum-capacity sweep completed 32K C11, 64K C7, 128K C4, and
256K C2; use the more conservative 32K C8 / 64K C6 settings in production.

**Fine-grained MoE (Qwen3.6-35B-A3B-FP8).** Supported and tuned. Two MoE paths are baked in and activate automatically for it: RDNA4-tuned fused-MoE Triton configs (removes the stock config's `M>=96` cliff, lower prefill TTFT, lossless) and a custom bf16 MoE-gate GEMM (`RADIANCE_MOE_ROUTER`, on by default) for the skinny `n` in `[6,16]` batch band that rocBLAS serves poorly (~2.5x faster cold, bit-identical output; wvSplitK already covers `n<=5`). A serving requirement: with `--mamba-cache-mode=align` this model's attention block size is 2240, and align asserts `block_size <= max_num_batched_tokens`, so keep **`--max-num-batched-tokens >= 2240`**; the Compose default is 4096.

**Gemma-4-31B-it-FP8.** Supported (0.4.0), e.g. `RedHatAI/gemma-4-31B-it-FP8-block`. Quantization is auto-detected from `config.json` (compressed-tensors, 128x128 blocks) and the RDNA4-tuned block-FP8 GEMM configs for its shapes load automatically (measured -5% TTFT at 8K prompt, decode unchanged). Text and vision both work. It is not a GDN hybrid, so drop `--mamba-cache-mode`; it also uses its own chat template and parsers rather than the Qwen ones. It carries a lot of KV (60 layers: 50 sliding-window + 10 global head-512), so at a given `--gpu-memory-utilization` it wants a smaller `--max-model-len` than the Qwen models. Long-context prefill is tuned for its head-512 global-attention layers (a head-size-keyed 2D attention config, measured up to -38% TTFT at 64K, -46% at 120K vs the untuned kernel; inert on other head sizes).

**Gemma-4-31B MTP speculative decoding.** For a large decode speedup, pair the target with Google's official drafter `google/gemma-4-31B-it-assistant` (vLLM loads its `gemma4_assistant` checkpoint as an MTP model). Lossless (the target verifies every drafted token), and the `RADIANCE_DYNAMIC_DRAFT` controller applies. The drafter has a head-512 layer, so its speculative `attention_backend` must be `ROCM_AITER_UNIFIED_ATTN` (`flash_attn` caps at head 256). Example: `--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' --no-async-scheduling --trust-remote-code`.

**bf16 / `auto` KV cache is supported** (0.1.5). Earlier builds crashed at startup with a Triton shared-memory (`OutOfResources`) error on any 2-byte KV cache (including the `--kv-cache-dtype auto` default), because the attention decode kernel's tile didn't fit the R9700's 64 KiB LDS at head_size 256. As of 0.4.0 that fit is enforced generally, for any head size and KV dtype, so fp8, bf16, and `auto` all work, including models with a head_size of 512, where AITER's own pick overflows LDS the same way. fp8 packs more KV per GB of VRAM; bf16 / `auto` keeps full KV precision.

## Why this exists

vLLM's ROCm builds target datacenter cards (MI300 / CDNA). RDNA4 workstation cards like the R9700 (gfx1201) don't work out of the box: AITER isn't enabled for the arch, GPU enumeration fails, several kernels need patching, and the vendor attention and GEMM paths aren't tuned for RDNA4. This image pins a working combination, builds AITER from source for gfx1201, applies the fixes, and adds tuned kernels.

## Legacy upstream 0.5.8 reference

Everything below this heading documents the historical upstream 0.5.8 image
and is retained for migration context. Its AITER attention flags, old
`RADIANCE_FAST_REDUCE` / `RADIANCE_AR_QUANT` names, 0.92 memory setting, and
`stilldeadcode/*:0.5.8` examples are not the current fork defaults. New
deployments should use the repository's portable `docker-compose.yml` and the
current controls above.

### Historical stack

Everything below is compiled from source for `gfx1201` in the image build (0.5.0 onward); nothing is
pulled from a prebuilt wheel index.

| Component | Version |
|---|---|
| vLLM | 0.26.0 |
| PyTorch | 2.11.0 |
| Triton | 3.6.0 |
| torchvision | 0.24.1 |
| AITER | 0.1.17 |
| ROCm userspace | 7.14, bundled |
| Base | `rocm/dev-ubuntu-24.04:7.14.0-full` (Ubuntu 24.04, Python 3.12) |

The PyTorch / Triton / torchvision versions are the ones vLLM 0.26.0 itself pins, not a newer combination
chosen for this image. That is deliberate: see the tensor-parallel hang note below.

There is no flash-attention package: the vendor flash kernels have no gfx1201 device code. Attention
runs on the AITER unified path, and the vision tower on the image's own Triton flash kernel.

## What it patches (to make vLLM run on gfx1201)

- GPU enumeration (amdsmi init order). Without it the platform is undetected and device count reads 0.
- AITER enablement for gfx12x (upstream gates it to MI3xx).
- Triton driver activation for the GPU-less model-inspection subprocess.
- Native sampler fallback (AITER's top-k/top-p kernel doesn't build on RDNA4).
- Tool-parser streaming vs non-streaming consistency.
- `from_json` Jinja filter for tool-calling chat templates.
- MTP drafter unpadding, so `--speculative-config`'s `disable_padded_drafter_batch:true` works (the single-stream MTP speed path).
- MTP drafter multimodal mask alignment, so speculative decoding works with image inputs (otherwise the vision-placeholder mask outlives the compacted draft batch and the engine crashes).
- `torch.compile` telemetry JSON encoding, which otherwise raises `TypeError: Object of type function is not JSON serializable` at startup on this torch version (harmless but alarming: the serve came up anyway).

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
| `RADIANCE_RUN_BWTEST` | `1` | run the GPU topology + bandwidth sweep at startup (`rocm-bandwidth-test`, compiled into the image): device list, P2P access matrix, NUMA distances, and peak uni/bidirectional copy bandwidth per agent pair. Backgrounded and takes about a second, so it never delays the serve; the report lands in the log a few seconds in. Set `0` to skip it. |
| `RADIANCE_BWTEST_TIMEOUT` | `150` | seconds to bound the sweep, in case it stalls on an unusual topology |
| `RADIANCE_NUMA_BIND` | unset (off) | NUMA pinning for multi-node hosts; see below. Same as `--numa-bind`, which wins if both are given |
| `RADIANCE_BANNER_PLAIN` | `0` | set `1` for a startup banner without ANSI colour (log scrapers, CI). `NO_COLOR` does the same |

> **Fixed in 0.5.7 — tensor-parallel GPU hang under sustained load (multi-GPU only).** Builds 0.5.0 through 0.5.5-pre could hang a GPU during long agentic sessions: both cards pegged at 100% utilisation while drawing a fraction of their power cap, the driver then reporting `HW Exception ... GPU Hang`, the engine dying on an RPC timeout and the container restarting. **The cause was a dependency mismatch, not a kernel bug.** vLLM 0.26.0 pins `torch == 2.11.0`, and this image's build strips torch/torchvision pins (via vLLM's own `use_existing_torch.py`, which exists so pip does not refetch them) — earlier 0.5.x builds then compiled against torch 2.13 / triton 3.7.1 / torchvision 0.28, a combination upstream never tests. The pinned trio (torch 2.11.0, triton 3.6.0, torchvision 0.24.1) is restored, and the hang is gone under the workload that reproduced it. Single-GPU serves were never affected, and nothing is disabled: speculative drafting and the fp8 all-reduce both remain on by default. If you build your own image, take the versions upstream pins — they are not free choices on this architecture.

All of these are baked ON in the image. Set `RADIANCE_DYNAMIC_DRAFT=0` to turn draft control off (`RADIANCE_AR_MAX_KB`, `RADIANCE_DRAFT_SCHEDULE`, and `RADIANCE_DRAFT_TAU` are values, not toggles). `RADIANCE_DYNAMIC_DRAFT` only does anything when speculative decoding is enabled; it is lossless (it changes only *how many* tokens are drafted and whether they come from MTP or a verbatim copy of earlier text, never what the model verifies).

**NUMA pinning (`RADIANCE_NUMA_BIND` / `--numa-bind`, opt-in, off by default).** On a multi-socket or multi-NUMA-node host, pin the server and its TP workers to the NUMA node(s) local to the GPUs so memory stays off the cross-node link. Set `RADIANCE_NUMA_BIND=auto` (detect from the visible GPUs) or pass `--numa-bind[=SPEC]` in the command; the flag wins. `SPEC` = `auto` \| explicit nodes (`0`, `0,1`) \| `bind=<nodes>` \| `interleave[=<nodes>]` \| `preferred=<node>` \| `none`. It is a no-op on single-node hosts and requires `--cap-add SYS_NICE` under Docker's default seccomp (already covered if you run `--security-opt seccomp=unconfined`).

## Requirements

- AMD Radeon AI PRO R9700 (gfx1201). Compiled for gfx1201 only, won't run on other GPUs. Two GPUs (TP=2) is the only configuration tested so far.
- Linux host with the amdgpu kernel driver and `/dev/kfd` + `/dev/dri`. ROCm userspace is inside the image.
- Docker with device passthrough.

## Run

On start the image prints a RADIANCE banner and runs a quick preamble (GPU count, gfx1201 check, P2P, enabled optimizations, component versions), then hands off to `vllm serve`. (It also runs a GPU topology + bandwidth sweep — device list, P2P access matrix, NUMA distances, and peak uni/bidirectional copy bandwidth for every agent pair. `rocm-bandwidth-test` is compiled into the image and the sweep is **on by default**: it is backgrounded and takes about a second, so it never delays the serve, and its report appears in the log a few seconds in. Set `RADIANCE_RUN_BWTEST=0` to skip it, or `RADIANCE_BWTEST_TIMEOUT` to bound it.) First argument is the model path, the rest are `vllm serve` flags. The `RADIANCE_*` vars below are the custom optimizations (see the table above). They are already baked ON in the image; they are listed here so they are visible and easy to flip off.

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
  stilldeadcode/vllm-radiance:0.5.8 \
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
    image: stilldeadcode/vllm-radiance:0.5.8
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

## First run is slower

With an empty cache the first start spends a few extra minutes compiling Triton / inductor kernels before the engine comes up; it looks idle but it is compiling. (Older builds spent 15 to 20 minutes here, dominated by the gated-delta-net fp32 autotune; that path is gone on this stack.) Mount a persistent cache so restarts stay fast:

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
| `--attention-backend` | `R4D` | current tuned target attention path; AITER remains a fallback/control |
| `--max-model-len` | model dependent | context length per request |
| `--max-num-seqs` | workload dependent | max concurrent sequences |
| `--gpu-memory-utilization` | `0.85` starting point | Leaves measured runtime/drafter headroom; raise only after a model-specific capacity qualification |
| `--enable-prefix-caching` | on for shared prefixes | enables automatic prefix caching; **required**: hybrid (GDN/mamba) models leave it off by default even though the engine default looks on |
| `--mamba-cache-mode` | `align` (hybrid models) | makes the linear-attention (GDN) layers prefix-cacheable; pair with `--enable-prefix-caching` on this hybrid. `none` disables mamba-layer caching; `all` is unsupported by this model |
| `--numa-bind` | omit (off) | multi-NUMA-node hosts only: pin the fleet to the GPU-local node(s). `auto` / `<nodes>` / `interleave` / `preferred=<n>` / `none`. Same as `RADIANCE_NUMA_BIND`; needs `--cap-add SYS_NICE`. See NUMA pinning above. |

Speculative decoding (MTP). Two forms depending on where the MTP head lives:

```
# Qwen3.6-27B / 35B: the MTP head is in the target checkpoint, so no separate drafter model
--speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'

# Gemma-4-31B: the drafter is a separate model, so add "model" (and --trust-remote-code --no-async-scheduling)
--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'
```

**What `num_speculative_tokens` means here.** In stock vLLM it is a *fixed* draft length: every decode step drafts exactly that many tokens and verifies them. With `RADIANCE_DYNAMIC_DRAFT=1` (baked on) it becomes a **ceiling, not a fixed cost**: per request the controller drafts *up to* that many tokens, stops early on low-acceptance content, and may take a verbatim n-gram continuation when it matches the drafter's own guess, but the total draft is always clamped to `num_speculative_tokens`. So a larger value like **8** is the recommended default: it gives the dynamic drafter more room to run deep on high-acceptance content (code, JSON, boilerplate) without adding fixed overhead on prose. There is no separate depth-ceiling knob to keep in sync; the ceiling is `num_speculative_tokens` itself. (Set `RADIANCE_DYNAMIC_DRAFT=0` to get the classic fixed-length behavior, in which case a smaller value such as 3 is more typical.)

`disable_padded_drafter_batch:true` is the key single-stream lever (~+50% on Qwen3.6-27B): it drops the drafter's batch padding, and the image bakes the vLLM unpad patch this relies on. Leave it on. Note it is incompatible with async scheduling: pass `--no-async-scheduling` to disable it explicitly (otherwise vLLM auto-enables async scheduling and then disables it with a runtime warning; `--async-scheduling` would hard-error).

Prefix caching (shared system prompts, RAG, agentic context):

```
--enable-prefix-caching --mamba-cache-mode align
```

Automatic prefix caching reuses a shared prompt prefix across requests so only the new suffix is prefilled, a large time-to-first-token drop when many requests share a system prompt or document. On this **GDN hybrid you must pass both flags**: hybrid models default their prefix-caching support flag off ("experimental"), so vLLM **silently disables** prefix caching unless `--enable-prefix-caching` is given, and `--mamba-cache-mode align` is what makes the linear-attention (GDN) layers cacheable by snapshotting and restoring their conv + recurrent state at block boundaries. That restore is **verified bit-identical to a full recompute** (including under MTP), so outputs are unchanged; the win is purely latency (measured ~3.6x in the original qualification and 12.8x for a 32K shared prefix in the MXFP4+DFlash production profile). Trade-offs: align reconciles the mamba and attention page sizes, which raises the attention block size to 1664 tokens and adds one state block per linear-attention layer (slightly lower max concurrency at full context), and prefix hits land on 1664-token boundaries. Do **not** use `--mamba-cache-mode all` (unsupported by this model, raises at startup) and do **not** set `VLLM_SSM_CONV_STATE_LAYOUT=DS` (asserts under MTP + align).

Tool-calling and reasoning:

```
--enable-auto-tool-choice --tool-call-parser <parser> --reasoning-parser <parser>
```

The portable Compose intentionally uses each checkpoint's native chat template.
Pass `--chat-template file.jinja` in a private override only when a checkpoint
needs a deployment-specific template. The image ships the `from_json` filter
those templates often rely on.
