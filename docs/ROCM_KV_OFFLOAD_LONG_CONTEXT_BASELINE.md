# Native ROCm KV offload: 128K/256K pressure baseline

Status: baseline complete. This report is the pre-tuning reference for native
CPU KV offload on dual Radeon AI PRO R9700. It deliberately measures cache
placement and long-context scheduling; it is not a BetterBench quality run or
a claim of maximum decode throughput.

## Executive result

Native CPU offload is a fast **external prefix-cache tier**, not extra live GPU
KV capacity. Restoring one retained 256K prefix moved 8.67 GB across both TP
ranks in 308 ms of aggregate transfer-counter time. Its end-to-end output rate
was 58.10 tokens/s, only 3.5% below the matched 60.20 tokens/s local-GPU hit.

The capacity boundary matters more than the transfer speed:

- 128K C1 and C2 retained every cacheable prompt token on GPU. At C4, two
  prefixes survived and two were recomputed.
- 256K C1 retained one full cacheable prefix. At C2, one prefix survived and
  one was recomputed.
- At 256K C3, the no-offload repeat recomputed all three prompts. The 24 GiB
  tier restored one and recomputed two, raising end-to-end output throughput
  from 1.854 to 2.735 tokens/s (+47.5%).
- At 256K C4, the repeat wave churned both local and external caches. The final
  forced-restore phase had zero retained prefix tokens and recomputed all four
  prompts. It remained at 1.848 tokens/s.

No measured phase OOMed or preempted. All 34 offload-path repeat/restore versus
cold comparisons were byte-identical. The no-offload control had one known
DFlash numerical divergence at 256K C4, for 67/68 exact comparisons overall.
Four admitted 256K requests were not four simultaneously resident requests:
the scheduler peaked at two running and three capacity-waiting observations.
`max-num-seqs=4` is an admission ceiling, not a promise that four
maximum-length sequences fit on GPU.

## Reproducibility manifest

- Source branch: `agent/rocm-kv-offload-registration-fix`
- Source commit at measurement: `1402d1e`
- Candidate image: `vllm-radiance:kv-restore-candidate`
- Image digest:
  `sha256:5f0f77117efc7e4fdf77959d700be6fce304b5341d77163c281152613a42ea59`
- Radiance version:
  `0.9.3-dev.vllm0.28.0-r4d0.5.0-mxfp4.rx4.dflash2.xgrammar.openobj.kvoffload.restore2`
- vLLM 0.28.0, PyTorch 2.12.0+rocm7.14, AMD Triton 3.7.1,
  AITER 0.1.20, ROCm 7.14, libr4d 0.5.0
- Target:
  `/nvme/lexar-2/ai/models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd`
- Drafter:
  `/nvme/lexar-2/ai/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger`
- Common envelope: TP2, 90% GPU allocation, FP8 KV, 256K model limit,
  C4 admission, 4,096 batched tokens, prefix caching + GDN align, piecewise
  graphs, R4D target attention, DFlash K7, fast draft, temperature 0, seed 17,
  and 256 generated tokens/request
- Control: native CPU offload disabled
- Candidate: 24 GiB native CPU offload with automatic pageable fallback

Immutable runs:

- no-offload: `20260901T234819Z-kv-long-allgpu`
- 24 GiB offload: `20260902T004340Z-kv-long-offload24`

The raw phase name `gpu_hit` means **immediate repeat**. It is not assumed to
be a GPU hit: the server's source counters below determine whether each token
came from local GPU APC, external CPU KV, or recomputation.

## Cold long-context cost

TTFT is median/max across requests. Output TPS includes prompt processing and
generation, so it intentionally collapses for million-token waves. Prompt TPS
is total prompt tokens divided by the last first-token time. It is the useful
prefill comparison; it is not steady-state decode TPS.

| Context | C | No-offload TTFT | Offload TTFT | No-offload prompt TPS | Offload prompt TPS | No-offload output TPS | Offload output TPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 1 | 46.98 / 46.98 s | 50.63 / 50.63 s | 2,785 | 2,584 | 5.339 | 4.955 |
| 128K | 2 | 72.23 / 96.29 s | 72.61 / 96.77 s | 2,717 | 2,704 | 5.264 | 5.238 |
| 128K | 4 | 121.55 / 194.01 s | 122.24 / 195.07 s | 2,697 | 2,682 | 5.252 | 5.223 |
| 256K | 1 | 137.38 / 137.38 s | 137.99 / 137.99 s | 1,906 | 1,898 | 1.849 | 1.840 |
| 256K | 2 | 206.25 / 274.90 s | 206.92 / 275.74 s | 1,905 | 1,899 | 1.855 | 1.849 |
| 256K | 3 | 274.93 / 412.73 s | 275.75 / 413.82 s | 1,904 | 1,899 | 1.856 | 1.851 |
| 256K | 4 | 344.39 / 551.03 s | 345.55 / 552.93 s | 1,901 | 1,895 | 1.855 | 1.848 |

Except for the isolated 128K C1 observation, offload changed cold throughput by
less than 0.6%. The C1 difference should be treated as run variance until a
repeat establishes otherwise; the C2/C4 and 256K series do not reproduce it.

For perspective, the 256K C1 request still decoded at 228-231 tokens/s after
its 137-138 second cold TTFT. The low 1.84 output-TPS figure is not slow token
generation; it is 256 output tokens amortized across the enormous prefill.

## Immediate-repeat placement

| Context | C | No-offload output TPS | Offload output TPS | GPU-hit tokens | CPU-restored tokens | Recomputed tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 128K | 1 | 105.048 | 104.831 | 128,544 | 0 | 2,272 |
| 128K | 2 | 128.360 | 128.638 | 257,088 | 0 | 4,544 |
| 128K | 4 | 10.111 | 10.047 | 257,088 | 0 | 266,176 |
| 256K | 1 | 60.540 | 60.198 | 258,736 | 0 | 3,152 |
| 256K | 2 | 3.607 | 3.608 | 0 | 258,736 | 265,040 |
| 256K | 3 | 1.854 | 2.735 | 0 | 258,736 | 526,928 |
| 256K | 4 | 1.854 | 1.848 | 0 | 0 | 1,047,552 |

The first net capacity benefit is 256K C3: the GPU-only control had already
lost every prefix, while the external tier still supplied one. At 256K C2 the
source changes from one local hit in the control to one external hit in the
offload run, but aggregate throughput is effectively unchanged.

## Forced CPU restore

Before this phase the harness clears only local GPU APC. External CPU contents
remain as left by the preceding cold and immediate-repeat waves.

| Context | C | Median/max TTFT | Output TPS | CPU tokens | Recomputed | CPU to GPU | Transfer-counter time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 1 | 1.50 / 1.50 s | 103.523 | 128,544 | 2,272 | 4.41 GB | 156.8 ms |
| 128K | 2 | 2.22 / 2.91 s | 124.378 | 257,088 | 4,544 | 8.81 GB | 326.3 ms |
| 128K | 4 | 75.13 / 100.64 s | 10.051 | 257,088 | 266,176 | 8.81 GB | 340.3 ms |
| 256K | 1 | 3.30 / 3.30 s | 58.100 | 258,736 | 3,152 | 8.67 GB | 308.3 ms |
| 256K | 2 | 72.39 / 141.47 s | 3.591 | 258,736 | 265,040 | 8.67 GB | 329.5 ms |
| 256K | 3 | 141.90 / 279.88 s | 2.733 | 258,736 | 526,928 | 8.67 GB | 330.8 ms |
| 256K | 4 | 345.62 / 552.96 s | 1.848 | 0 | 1,047,552 | 0 | 0 |

The byte and time counters are summed Prometheus deltas across TP workers;
they are not a claim of single-link PCIe bandwidth. Two load operations in the
lower-level restore gate remain one logical TP2 restore.

The phase ordering is intentional but important: the 256K C4 result proves
that a realistic repeat wave can churn the external tier until nothing is
restorable. It does not prove that a local-only reset immediately after the
initial cold fill would also find zero blocks. A future replacement-policy
experiment should run those paths from independent cold fills.

## Resource and correctness observations

| Run | Peak VRAM GPU0/GPU1 | Minimum host available | Peak power GPU0/GPU1 | Peak junction GPU0/GPU1 |
|---|---:|---:|---:|---:|
| No offload | 27.99 / 27.99 GiB | 22.31 GiB | 341 / 484 W | 89 / 90 C |
| 24 GiB offload | 27.80 / 27.80 GiB | 20.11 GiB | 358 / 465 W | 91 / 91 C |

- All 85 main HTTP request responses completed across 35 case/phase waves.
  Every offload-path comparison was byte-identical. In the no-offload control,
  one of four 256K C4 immediate-repeat responses differed from its cold output;
  the other 33/34 control comparisons matched. This is the already-documented
  DFlash/APC numerical qualification boundary, not an offload-only failure.
- Preemptions: zero in every phase.
- OOMs and request failures: zero.
- Peak sampled GPU KV utilization was 55.1% in the control and 69.6% with
  offload. Utilization is allocator occupancy, not total board VRAM usage.
- DFlash acceptance was 99.5-100% on these deliberately repetitive fixtures.
  That is useful for timing stability but is not representative of BetterBench
  acceptance and does not supersede the existing strict DFlash/APC
  qualification failure.
- The preliminary 8K harness smoke exposed that known DFlash numerical
  variability on one repeat. It is preserved as
  `20260901T234753Z-kv-long-smoke-allgpu`, not hidden or counted as a main
  pressure result.

## What this establishes for tuning

The baseline separates three independent opportunities:

1. **Transfer path:** already fast for one retained prefix; it is not the
   dominant 256K C2-C4 cost.
2. **External retention/replacement:** the 24 GiB tier provides a useful C3
   win but collapses under the C4 repeat wave. This is the largest observed
   offload-specific gap.
3. **Active scheduling:** only two requests ran during the 256K pressure waves.
   Increasing CPU offload size alone does not make four full-length sequences
   simultaneously GPU-resident. Admission, chunked prefill, and cache policy
   must be measured separately.

The current evidence supports keeping 24 GiB/C4 as a safe production admission
profile, but not advertising it as four simultaneous 256K streams. A 36 GiB/C6
profile remains experimental until retention, host-headroom, and sustained
decode tests demonstrate a benefit over the extra queueing.

## Reproduction

The harness requires `VLLM_SERVER_DEV_MODE=1` for selective cache reset. Never
enable that endpoint in production.

```bash
# No-offload control
python benchmarks/bin/run_kv_offload_long_context.py \
  --model Qwen3.8-27B \
  --tokenizer /models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd \
  --output-dir benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-kv-long-allgpu \
  --label allgpu \
  --cases 131072:1 131072:2 131072:4 262144:1 262144:2 262144:3 262144:4 \
  --max-tokens 256 --continue-on-error

# Matched server with --kv-offloading-size=24
python benchmarks/bin/run_kv_offload_long_context.py \
  --model Qwen3.8-27B \
  --tokenizer /models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd \
  --output-dir benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-kv-long-offload24 \
  --label offload24 \
  --cases 131072:1 131072:2 131072:4 262144:1 262144:2 262144:3 262144:4 \
  --max-tokens 256 --cpu-restore --continue-on-error
```

On the host, use the canonical `/nvme/lexar-2/ai/models/...` tokenizer path;
`/models/...` is the container spelling shown for portable examples.
