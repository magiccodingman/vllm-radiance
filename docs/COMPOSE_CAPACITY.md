# Public Compose and capacity qualification

## Distribution boundary

`docker-compose.yml` is publication-safe. It defaults to the published
`magiccodingman/vllm-radiance:latest` image and relative `./models` and
`./vllm-cache` host directories; it contains no developer filesystem paths or
host-specific group IDs. `.env` and `docker-compose.dev.yml` are ignored by both
Git and the Docker build context. `.gitlab-ci.yml` also refuses to publish when
either developer overlay filename is present in the release checkout.

`docker-compose.dev.example.yml` is the tracked template. Each developer copies
it to `docker-compose.dev.yml`, adds local image/path/GID overrides there, and
starts the merged configuration with:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## What was measured

Capacity was qualified on 2026-08-23 with:

- two Radeon AI PRO R9700 32 GiB GPUs, TP2;
- native-FP8 `Qwen3.8-27B-heretic-ara-fp8-magiccodingman` target;
- selective-FP8 `Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman` drafter
  (2.34 GiB), five speculative tokens, draft TP2, `TRITON_ATTN`;
- mandatory FP8 KV, V2 runner, `PIECEWISE` graphs, and 85% GPU allocation;
- Radiance preshuffle, target unified attention, and custom TP2 reduction
  retained; FP8 all-reduce payload disabled for the experimental DFlash lane;
- 2,048 maximum batched tokens and no CPU/KV offload.

Each capacity sample sent disjoint random prompts with `temperature=0`, exactly
64 output tokens, and enough input tokens to bring each request to the listed
maximum context. It submitted one full concurrent wave and required every
request to complete. These are capacity/serving checks, not steady-state TPS
benchmarks.

## Results

| Exact server maximum | Suggested resident setting | Tested API burst | Engine KV tokens / theoretical full requests | Duration | Minimum physical headroom/GPU | Run ID |
|---:|---:|---:|---:|---:|---:|---|
| 8K | 8 | 8 | 107,193 / 13.09x | 18.3 s | 4.41 GiB | `20260823T154634Z_compose-capacity-dflash-k5-8k-exact` |
| 16K | 7 | 8 | 130,065 / 7.94x | 38.5 s | 6.36 GiB | `20260823T153836Z_compose-capacity-dflash-k5-16k-exact` |
| 32K | 5 | 6 | 192,565 / 5.88x | 63.2 s | 6.43 GiB | `20260823T154221Z_compose-capacity-dflash-k5-32k-exact` |
| 64K | 3 | 3 | 264,104 / 4.03x | 75.8 s | 6.68 GiB | `20260823T151617Z_compose-capacity-dflash-k5-64k-pressure` |
| 128K | 2 | 2 | 304,826 / 2.33x | 134.4 s | 6.53 GiB | `20260823T152611Z_compose-capacity-dflash-k5-128k-c2` |
| 256K | 1 | 1 | 336,777 / 1.28x | 197.6 s | 6.55 GiB | `20260823T153133Z_compose-capacity-dflash-k5-256k` |

The 16K×8 and 32K×6 request bursts completed without offload, OOM, or request
failure, but logs show normal scheduler waiting while chunked prefill admitted
the waves. The suggested settings use the floor of the engine's fully resident
KV estimate (and leave an extra full-request margin at 64K). They are the better
choice when predictable latency matters. Operators who prefer a larger accepted
queue can use the tested burst column.

The 8K recommendation remains capped at eight because that was the qualified
admission ceiling; the engine's 13.09x KV estimate is not a claim that a
different graph capture with `MAX_NUM_SEQS=13` has been validated.

## MXFP4/W4A8 continuation

Capacity was requalified on 2026-08-25 after adding the native gfx1201 W4A8
path, using:

- two Radeon AI PRO R9700 32 GiB GPUs, TP2;
- `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` target (18.44 GiB safetensors);
- `tcclaviger/Qwen3.8-27B-DFlash2-FP8` target-matched drafter (1.97 GiB), five
  speculative tokens, draft TP2, `TRITON_ATTN`;
- mandatory FP8 KV, R4D target attention/GDN/all-reduce, V2 runner,
  `PIECEWISE` graphs, 4,096 batched-token budget, and 90% GPU allocation;
- published image
  `magiccodingman/vllm-radiance@sha256:8df90677c0f1fb013d958184aa0bf24af91e34688b61b924fa4facd7da333430`;
- no CPU/KV offload.

### Prefix-disabled high-capacity sweep

The initial sweep used `--no-enable-prefix-caching --mamba-cache-mode=none` to
measure the largest practical all-GPU envelope:

| Maximum context | Conservative production C | Highest completed burst | End-to-end output TPS at highest burst | Total token TPS | Minimum physical headroom/GPU |
|---:|---:|---:|---:|---:|---:|
| 32K | 8 | 11 | 27.91 | 3,544.1 | 5.06 GiB |
| 64K | 6 | 7 | 12.52 | 3,191.8 | 5.06 GiB |
| 128K | 4 | 4 | 5.18 | 2,648.1 | 5.21 GiB |
| 256K | 2 | 2 | 1.93 | 1,969.6 | 5.28 GiB |

Each request used approximately `maximum context - 512` input tokens plus 256
generated tokens. Every request completed with no OOM or engine failure. Output
TPS is low by design because these figures include the enormous cold prefills;
total TPS shows prompt processing throughput. The conservative 32K/64K settings
leave room below the completed boundary.

### Recommended prefix-cached production profile

The final production choice is:

```text
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=4
GPU_UTIL=0.90
--enable-prefix-caching
--mamba-cache-mode=align
```

Align mode preserves/restores the hybrid GDN convolution and recurrent state at
cache block boundaries. On boot, vLLM reported 576,001 usable KV tokens and
4.39x theoretical concurrency at 131,072 tokens. A four-request pressure wave
with 130,816 input plus 256 output tokens per request completed 4/4 with zero
preemptions, zero OOMs, and 5.17 GiB minimum physical headroom per GPU. vLLM
chunk-scheduled two of the four giant cold prefills while two temporarily waited
for capacity; all four reached peak admitted concurrency and completed.

A deterministic 32,768-token shared-prefix probe established both performance
and state integrity:

| Sample | TTFT | Prefix reuse | Output qualification |
|---|---:|---:|---|
| Cold | 9.036 s | 0% | reference |
| Warm 1 | 0.705 s | 93.7% | byte-identical to cold |
| Warm 2 | 0.707 s | 93.7% | byte-identical to cold |

That is a 12.8x TTFT improvement. A separate C4 shared-prefix wave reused 93.7%
of queried tokens with zero preemptions. During subsequent live agent traffic,
the incremental cache-hit rate was 96.2%; decode-heavy windows measured
90.4-106.8 TPS at 59-63% DFlash acceptance, while a lower-acceptance 42% window
measured 65.0 TPS. The cache was not the source of that variation.

The disjoint 512-input/512-output C4 control completed 12/12 at 355.80 output
TPS, 308.35 ms mean TTFT, 9.79 ms mean TPOT, and 64.19% draft acceptance—within
3.2% of the preceding 256K/prefix-disabled control. The public Compose therefore
keeps prefix caching and `align` enabled by default. Disabling cache/align or
graphs can expose more context, but that is a capacity trade rather than a free
performance improvement and requires a new qualification.

## Scope and correctness warning

This matrix is a sizing aid for this exact FP8 target/drafter and dual-R9700
profile. It is not transferable to a 35B model, BF16 weights or KV, a larger
drafter, another graph mode, or a different GPU count without re-running the
capacity workload.

DFlash2 remains experimental because its strict greedy-equivalence gate fails
for the deployment configuration. These successful runs establish VRAM fit,
scheduler behavior, and long-context execution only; they do not reclassify
DFlash2 as lossless or make it the public Compose default. The normal public
Compose remains the qualified non-speculative profile.

## Validation

- Public Compose resolves with `.env.example`; public plus developer example
  and public plus the ignored local developer overlay also resolve.
- A live start through `docker-compose.yml` plus `docker-compose.dev.yml`
  reached healthy state and returned the requested deterministic chat
  completion; the container and Compose network were removed afterward.
- Docker BuildKit `--check` completed with no warnings.
- `pip check` reported no broken requirements. Device-scoped imports reported
  PyTorch `2.12.0+rocm7.14`, Triton `3.7.1`, AITER `0.1.20`, and vLLM
  `0.28.0.dev0+a014e35`.
- YAML parsing, all Compose resolutions, shell syntax, the release privacy
  grep, and `git diff --check` passed.
