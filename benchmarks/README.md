# Radiance benchmark lab

This directory holds the reusable online-serving benchmark harness and immutable
run history for Radiance builds.  The suite measures stable, bounded concurrency
rather than searching for maximum throughput.

## Safety and storage

- Models are read-only from `/nvme/lexar-2/ai/models`.
- Docker engine data remains under `/nvme/lexar-1/docker/data`.
- Compilation caches live in `/nvme/ediloca-1/scratch/vllm-radiance-cache`.
- Every script verifies the required mounts before starting a container or run.
- A run directory is never reused or overwritten.

## Profiles and matrix

The default `quick` profile is the everyday A/B gate. It uses one server warmup,
two bounded decode repetitions at concurrency 1/2/4/8 for TP2 and 1/2/4 for
the constrained TP1 reference, a 2K prefill sweep,
and a single 8K context check. `standard` adds a third decode sample and a
second prefill sample. `qualification` adds sustained decode and concurrent
long-context capacity checks and is reserved for milestone builds.
Each exact workload/concurrency shape gets one unmeasured request wave before
its first sample. Warmups and every measured repetition use distinct,
deterministic prompt seeds, so prefix caching cannot leak across samples. This
prevents lazy Triton compilation or autotuning from being misreported as model
latency while adding only a bounded amount of runtime.
Every request also sets `temperature=0` explicitly. This makes outputs and
speculative acceptance reproducible instead of inheriting a model/server
sampling default. Set `BENCH_TEMPERATURE` only for a separately labeled sampling
experiment; temperature is part of the comparison key.

`bin/run_matrix.sh` defaults to the model-neutral, non-speculative gate:

1. TP=2, speculative decoding off.

Add the current-model TP1 reference with
`BENCH_CONFIGS=tp2_spec-off,tp1-eager8k_spec-off`, and add `tp2_spec-on`
explicitly at speculative milestones. The current 27B model's TP=1 MTP head
requires another 2.37 GiB when only about 1.06 GiB remains, so
`tp1-eager8k_spec-on` is retained as an explicit diagnostic profile but is not
part of the routine matrix. CPU offload is not used to force it to fit.

The default model is
`/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman`. Override
`MODEL_HOST` for another checkpoint; `MODEL_NAME` defaults to that directory's
basename and can be overridden separately. The model configuration and resolved
container command are checked before every run. Every
configuration explicitly forces `--kv-cache-dtype=fp8`.

TP=2 uses a model-neutral 16K server envelope at 85% GPU utilization. This
reserves about 4.8 GiB outside vLLM's allocation on each 32 GiB card, leaves
room for a DFlash2 drafter and runtime variation, and does not try to turn the
performance gate into a capacity test. The 28.75 GiB checkpoint does not leave
KV space on one 31.9 GiB R9700 with compiled execution, so TP=1 uses an 8K
maximum and eager execution at 95% utilization. TP1 is a fit-specific reference,
not a requirement for larger checkpoints such as 35B. Its decode concurrency
sweep remains useful, while long-context/capacity is reported separately. CPU
offload and near-100% settings are not part of the routine TP2 matrix.
The default scheduler budget is 4,096 batched tokens. This is the measured R4D
chunked-prefill/MTP operating point and, for Qwen hidden size 5,120, keeps the
40 MiB TP2 collective payload within libr4d's exact-message ceiling. Override it
for a separately labeled model/shape experiment; do not silently compare a
2,048-token run with a 4,096-token run.

Set a note for the run with:

```bash
BENCH_NOTES="reason for this baseline" BENCH_SUITE=quick benchmarks/bin/run_matrix.sh
```

Limit an iterative run to relevant configurations with, for example:

```bash
BENCH_CONFIGS=tp2_spec-off BENCH_SUITE=quick benchmarks/bin/run_matrix.sh
```

Include the supported speculative lane at a milestone with:

```bash
BENCH_CONFIGS=tp2_spec-off,tp1-eager8k_spec-off,tp2_spec-on \
  BENCH_SUITE=quick benchmarks/bin/run_matrix.sh
```

Set `RADIANCE_IMAGE` to compare an exact image reference. For a startup-only
validation, call `bin/run_configuration.sh` with `--suite smoke`.

For an operator-specific iteration, select only the affected families while
retaining the same shapes and warmup rules. For example, a GDN prefill change
can use `BENCH_WORKLOADS=prefill,context`; an all-reduce/decode change can use
`BENCH_WORKLOADS=decode`. The selected filter is stored in the manifest. Full
`quick` and `qualification` gates remain milestone requirements.

`BENCH_WORKLOADS=correctness` runs eight fixed, meaningful greedy prompts and
stores their exact text, prompt/completion token lengths, seeds, and fixture
checksum. Use it with `bin/verify_outputs.py` when synthetic random-token
prompts expose near-tie numerical drift or when qualifying speculative decode.
Set `BENCH_CORRECTNESS_REPETITIONS` to audit within-server repeatability and
`BENCH_CORRECTNESS_LOGPROBS` to retain top-logprob evidence around a first
divergence. `BENCH_CORRECTNESS_PROMPTS` selects an alternate immutable fixture;
the DFlash hybrid-GDN graph diagnostic uses
`fixtures/gdn-shape-alias-prompts.json` to exercise short prompt lengths around
the speculative batch-shape boundary.

For deeper output triage, compare two retained correctness payloads with:

```bash
benchmarks/bin/analyze_correctness.py \
  BASE/raw/correctness_fixed.json CANDIDATE/raw/correctness_fixed.json \
  --output CANDIDATE/correctness-analysis.json
```

The report keeps strict token equality intact while adding repeatability,
first-divergence position, nearby token/text context, and top-two logprob
margins when available. A deterministic near-tie is evidence for numerical
drift, not permission to weaken the strict qualification gate.

Compare two completed run directories with direction-normalized deltas (positive
always means better). Speculative comparisons also show candidate acceptance
rate, acceptance-length, and the rate change in percentage points:

```bash
benchmarks/bin/compare.py benchmarks/runs/BASELINE benchmarks/runs/CANDIDATE
```

When both lanes share one matrix run (for example non-spec versus DFlash2),
filter and normalize their configuration keys explicitly:

```bash
benchmarks/bin/compare.py benchmarks/runs/RUN benchmarks/runs/RUN \
  --baseline-config tp2_spec-off --candidate-config tp2_spec-on
```

Pass `--fail-below -5` to make any common decode-throughput regression worse
than 5% fail a milestone gate. Inspect the recorded CV before treating a small
delta as meaningful.

For a greedy speculative-decoding correctness gate, compare the matching
configuration directories. The verifier requires identical seeds, input
lengths, generated text, and output token lengths for every common raw case:

```bash
benchmarks/bin/verify_outputs.py RUN/tp2_spec-off RUN/tp2_spec-on \
  --output RUN/spec-output-equivalence.json
```

The workload does not try to fill VRAM. The canonical TP2 gate is c1/c2/c4/c8
inside a 16K, 85%-allocation envelope; it never searches for the largest batch
that fits. `MODEL_HOST`, `MODEL_NAME`,
`WEIGHT_QUANTIZATION`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, `TP1_GPU_UTIL`,
`TP2_GPU_UTIL`, `TP1_MAX_MODEL_LEN`, and `TP2_MAX_MODEL_LEN` are profile
inputs, so the same core workloads can be reused for larger models. Select only
the configurations a model can safely host; for example a 35B model may use
`BENCH_CONFIGS=tp2_spec-off` while retaining directly comparable TP2 cases.
Maximum-context requests are qualification checks, not routine performance
samples. `ATTENTION_BACKEND`, `ADDITIONAL_CONFIG_JSON`,
`SPECULATIVE_CONFIG_JSON`, and `COMPILATION_CONFIG_JSON` provide recorded,
command-line-visible experiment
switches without editing the harness. The latter defaults to the normal MTP
configuration only when a speculative lane is requested and no explicit JSON
is supplied.

For an explicit no-offload capacity qualification, select the `capacity`
workload and provide `context_tokens:concurrency` pairs. Every pair submits one
full simultaneous wave and retains normal telemetry/manifests:

```bash
BENCH_WORKLOADS=capacity \
BENCH_CAPACITY_CASES='8192:8 16384:7 32768:5 65536:3' \
MAX_NUM_SEQS=8 \
benchmarks/bin/run_configuration.sh \
  --run-root benchmarks/runs/RUN_ID \
  --label capacity --tp 2 --spec on \
  --max-model-len 65536 --gpu-memory-utilization 0.85 --suite quick
```

The server's `MAX_MODEL_LEN` must cover the largest pair. Capacity success means
all requests completed without CPU/KV offload; inspect the server log for the
engine's fully resident KV ceiling and any scheduler queueing before promoting a
submitted burst size to a default.

Set `TP2_ENFORCE_EAGER=1` for a recorded eager TP2 lane. This is intended for
first qualification of a new speculative runtime, where graph correctness is
not yet established, and does not alter the portable default.
`VLLM_USE_V2_MODEL_RUNNER=0|1` is passed through only when explicitly set, so
V1/V2 correctness controls can be compared without changing vLLM's default
runner selection.
Focused diagnostics may set a whitespace-separated
`BENCH_DECODE_CONCURRENCIES` (for example `"1 4"`); the canonical gate leaves
it unset and therefore remains c1/c2/c4/c8.

Kernel control cases can likewise override the compose defaults from the host,
including `RADIANCE_PRESHUFFLE`, `RADIANCE_USE_R4D`,
`RADIANCE_USE_R4D_GDN`, `RADIANCE_USE_R4D_AR`,
`RADIANCE_USE_R4D_AR_QUANT`, `RADIANCE_FAST_DRAFT`, and
`VLLM_ROCM_USE_AITER_LINEAR`. Their resolved values are retained by the
container inspection in each manifest; explicitly supplied values are also
listed in the manifest environment section.

### BetterBench publication lane

The synthetic harness remains the fast smoke/quick engineering gate. Published
cross-mode results use BetterBench v0.2.2 at exact commit
`575cc3925bac922d6ad4a39e62502673799979d9`, installed in the registered
`/nvme/ediloca-1/venv/vllm-bench-env` environment. The wrapper refuses to run
against another checkout. It records the profile, commit, version, raw JSON,
standalone HTML, Markdown report, server manifest, and one-second telemetry.

`standard` is the 10-pass/category comparison profile; `qualification` doubles
that to 20 passes/category. Both use corpus v1, greedy temperature-zero decoding,
seed `20260823`, unique nonce prefixes, c1/c2/c4/c8, and cold 2K/4K/7K prefill:

```bash
PREFIX_CACHING=off MAX_NUM_BATCHED_TOKENS=4096 \
BETTERBENCH_PROFILE=standard \
benchmarks/bin/run_configuration.sh \
  --run-root benchmarks/runs/RUN_ID --label CONFIG \
  --tp 2 --spec off --image IMAGE --max-model-len 8192 \
  --suite betterbench --notes 'exact experiment description'
```

Prefix caching is explicitly off across vLLM versions for the cross-image lane;
BetterBench also prefixes every prompt with a unique nonce. `run_betterbench.sh`
enforces the shared 8K envelope so non-spec, MTP, and DFlash results cannot
quietly drift into unlike capacity profiles.

### Experimental DFlash2 lane

DFlash2 remains explicit-only. The best bounded headroom profile found on the
dual R9700 host uses the selective-FP8 drafter at
`/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman`,
seven speculative tokens, draft TP2, `TRITON_ATTN` for the drafter, the V2-compatible model
runner, and `PIECEWISE` graph mode. The native-FP8 target, FP8 KV, 8K DFlash
envelope, 85% allocation, and normal Radiance target paths remain unchanged.
It is an experimental benchmark profile—not the compose default—because strict
greedy equivalence still fails even though repeated outputs are deterministic.

Use explicit JSON so manifests retain the full experiment contract:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
RADIANCE_USE_R4D_AR_QUANT=1 \
MAX_NUM_BATCHED_TOKENS=4096 \
COMPILATION_CONFIG_JSON='{"cudagraph_mode":"PIECEWISE"}' \
SPECULATIVE_CONFIG_JSON='{"method":"dflash","model":"/models/Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","max_model_len":8192}' \
BETTERBENCH_PROFILE=standard \
benchmarks/bin/run_configuration.sh \
  --run-root benchmarks/runs/RUN_ID --label tp2-r4d-dflash-k7 \
  --tp 2 --spec on --image IMAGE --max-model-len 8192 --suite betterbench
```

K5 remains an important matched control. K7 won the 10-pass BetterBench corpus
and c1/c2/c4/c8 sweep, but DFlash remains explicit-only because strict greedy
equivalence against matched non-spec passes only 3/8 fixed prompts. K5, K7, and
MTP were repeatable and produced the same fixed output stream, which localizes
the difference to the shared speculative/runner numerical path without
weakening the gate.

## Results

Each timestamped directory beneath `runs/` includes exact manifests, resolved
server commands, raw vLLM JSON, logs, two-second GPU/host telemetry, checksums,
and consolidated CSV/JSON/Markdown summaries. `telemetry-summary.json` and
`.csv` promote each case's peak VRAM/use/power/temperature and minimum VRAM and
host-memory headroom so model-size changes remain auditable. Only GPUs named
by the configuration's recorded `HIP_VISIBLE_DEVICES` are
promoted; the host's 2 GiB integrated GPU is intentionally excluded from
inference headroom summaries while its raw samples remain available.
Fixed seed `20260822`, exact input/output lengths, forced output length, warmups,
and repetitions make runs directly comparable across future images and forks.
