# RDNA4 upgrade progress

This document is the durable checkpoint log for merge request !5. All performance
numbers use native FP8 weights and mandatory FP8 KV cache with the reusable
`benchmarks/` workload contract. Positive deltas mean improvement.

## Reproducibility pins

- Model: `/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman`
  (28.75 GiB safetensors; checksum validation passed).
- vLLM main: `a014e35f38c80fb0652387740193ad2147fed6a3` (2026-08-22).
- Radiance 0.4.0 image: `sha256:f233e3e071653adac6821f9582070a941430a6b795482f6ff04115b17df37047`.
- Radiance 0.5.8 image: `sha256:2e788f7475907cc86d82acddb5f1e50360b1ffd39e1ffbd2d00ac705b728ccc1`.

## Benchmark contract

- TP2: 90% GPU-memory limit, server can host 32K, routine workloads stop at 8K,
  decode concurrency 1/2/4/8.
- TP1 reference: native/no-offload, eager, 95%, 8K, decode concurrency 1/2/4.
- Decode: 256 input + 256 forced output tokens, two repetitions and at least two
  request waves.
- Prefill/mixed: 2048 input + 64 output; TP2 concurrency 1/4/8, TP1 concurrency 1.
- A 32K request and longer repetitions are qualification-only. The routine gate
  is not a maximum-throughput or VRAM-saturation search.
- FP8 KV is a hard precondition checked from the resolved container command.

The checkpoint lacks calibrated attention q/k/v/prob scale tensors, so vLLM
records that FP8 KV attention uses scale 1.0. Dynamic KV scale calculation is
not forced on this hybrid GDN model because the current vLLM path disables it
for hybrid recurrent-state correctness. This remains a documented accuracy
risk to revisit on pinned main.

## Baselines

### Radiance 0.4.0

The corrected matrix completed TP2 non-spec, TP1 non-spec, and TP2 MTP. TP1 MTP
is unsupported natively: creation of the draft LM head requested 2.37 GiB with
only 1.06 GiB free. It is no longer in the routine matrix and is not forced with
CPU offload.

Key non-spec output TPS:

| Lane | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| TP2 | 34.24 | 61.62 | 117.12 | 215.53 |
| TP1 | 17.08 | 32.01 | 61.85 | excluded going forward |

TP2 prompt-token throughput (total TPS) at 2K input was 893.83 / 1557.12 /
2217.96 for concurrency 1/4/8. TP2 MTP output TPS was 45.62 / 66.33 / 135.73 /
239.03 at concurrency 1/2/4/8, with workload-dependent acceptance recorded in
the raw result.

### Radiance 0.5.8 source-head checkpoint

The non-spec quick gate completed in about 16 minutes. Against 0.4.0, TP2 decode
was -0.66%, -0.24%, +1.18%, and +1.22% at concurrency 1/2/4/8. TP2 2K prefill
total TPS improved +2.28%, +1.83%, and +1.01% at concurrency 1/4/8, while median
TTFT improved roughly 4.6-5.1%.

TP1 c1/c2/c4 stayed within +0.6-1.3%. TP1 c8 was capacity-distorted: 0.5.8
retained 28,216 KV tokens (3.44 full 8K requests), produced ~18 second tail
TTFTs, and lost 29.8% aggregate TPS while per-token latency stayed flat. This is
why TP1 now stops at c4; TP2 remains the c8 concurrency lane.

## Pinned-main patch audit

- Removed the old broad AITER enablement patch. Current vLLM deliberately routes
  RDNA4 Triton kernels separately from CDNA CK/MFMA/ASM kernels.
- Removed the sampler workaround now owned upstream.
- Removed Radiance's RMS/group-FP8 fusion patch because pinned main selects the
  native quant matcher on RDNA4 itself.
- Retained deterministic gfx1201 discovery, FP8 preshuffle/dispatcher and split-K
  alignment, unified-attention LDS fit/tuning, GDN triangular-solve WMMA, TP2
  custom all-reduce, router GEMM, and MTP correctness/controller hooks.
- Every retained vLLM/AITER patch was applied to a scratch copy of the exact
  pinned sources with its anchor-count and Python-parse guards enabled.

## Pinned-main smoke checkpoint

The fast iteration image built successfully as
`vllm-radiance:dev-a014e35` (local digest
`sha256:067f249f15ea6b2f63672c4bd706d80d084aaaaee65b24ed2023bc6b98e87d67`).
The exact stack is vLLM `0.28.0.dev0+a014e35`, PyTorch 2.11.0 ROCm 7.14,
Triton 3.6.0, and AITER 0.1.17.

TP2 non-spec smoke passed with native FP8 weights and mandatory FP8 KV. Startup
confirmed the Radiance preshuffle hook, unified-attention tuning, custom TP2
all-reduce plus FP8 payload path, Triton/FLA GDN prefill, and Triton GDN decode.
At the bounded 0.90 memory setting, each GPU used 13.98 GiB for model weights,
reserved 0.75 GiB for graphs, and retained 10.59 GiB for KV (525,501 tokens,
16.04 full 32K requests). This leaves the intended non-engine headroom instead
of tuning to the VRAM edge.

## Next checkpoints

1. Run the pinned-main non-spec performance gate and compare with Radiance 0.5.8.
2. Stage AITER and the official AMD PyTorch/Triton pair independently.
3. Integrate GDN prefill work, then qualify DFlash2 separately with reserved
   drafter headroom.
