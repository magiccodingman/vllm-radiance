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

Compare two completed run directories with direction-normalized deltas (positive
always means better). Speculative comparisons also show candidate acceptance
rate, acceptance-length, and the rate change in percentage points:

```bash
benchmarks/bin/compare.py benchmarks/runs/BASELINE benchmarks/runs/CANDIDATE
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

Kernel control cases can likewise override the compose defaults from the host,
including `RADIANCE_PRESHUFFLE`, `RADIANCE_FAST_REDUCE`,
`RADIANCE_AR_QUANT`, `RADIANCE_GDN_WMMA`, and
`VLLM_ROCM_USE_AITER_LINEAR`. Their resolved values are retained by the
container inspection in each manifest; explicitly supplied values are also
listed in the manifest environment section.

## Results

Each timestamped directory beneath `runs/` includes exact manifests, resolved
server commands, raw vLLM JSON, logs, two-second GPU/host telemetry, checksums,
and consolidated CSV/JSON/Markdown summaries. `telemetry-summary.json` and
`.csv` promote each case's peak VRAM/use/power/temperature and minimum VRAM and
host-memory headroom so model-size changes remain auditable. Fixed seed
`20260822`, exact
input/output lengths, forced output length, warmups, and repetitions make runs
directly comparable across future images and forks.
