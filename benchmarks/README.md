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
second prefill sample. `qualification` adds sustained decode and concurrent 16K
context plus a near-32K capacity request and is reserved for milestone builds.

`bin/run_matrix.sh` defaults to the model-neutral, non-speculative gate:

1. TP=2, speculative decoding off.
2. TP=1 native FP8 at 8K with eager execution, speculative decoding off.

Add `tp2_spec-on` explicitly at milestone checkpoints. The current 27B model's
TP=1 MTP head requires another 2.37 GiB when only about 1.06 GiB remains, so
`tp1-eager8k_spec-on` is retained as an explicit diagnostic profile but is not
part of the routine matrix. CPU offload is not used to force it to fit.

The model is always
`/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman`, and both
the model configuration and container command are checked before a run. Every
configuration explicitly forces `--kv-cache-dtype=fp8`.

TP=2 remains the production/default 32K envelope. The 28.75 GiB checkpoint
does not leave KV space on one 31.9 GiB R9700 with compiled execution, so TP=1
uses an 8K maximum and eager execution at 95% utilization. Its decode
concurrency sweep remains useful, while long-context/prefill capacity is
reported separately. CPU offload and near-100% memory settings are not part of
the routine matrix.

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

Compare two completed run directories with direction-normalized deltas (positive
always means better):

```bash
benchmarks/bin/compare.py benchmarks/runs/BASELINE benchmarks/runs/CANDIDATE
```

Pass `--fail-below -5` to make any common decode-throughput regression worse
than 5% fail a milestone gate. Inspect the recorded CV before treating a small
delta as meaningful.

The workload does not try to fill VRAM. `MODEL_HOST`, `MODEL_NAME`,
`WEIGHT_QUANTIZATION`, `MAX_NUM_BATCHED_TOKENS`, `TP1_GPU_UTIL`,
`TP2_GPU_UTIL`, `TP1_MAX_MODEL_LEN`, and `TP2_MAX_MODEL_LEN` are profile
inputs, so the same core workloads can be reused for larger models. Select only
the configurations a model can safely host; for example a 35B model may use
`BENCH_CONFIGS=tp2_spec-off` while retaining directly comparable TP2 cases.
Maximum-context requests are qualification checks, not routine performance
samples.

## Results

Each timestamped directory beneath `runs/` includes exact manifests, resolved
server commands, raw vLLM JSON, logs, two-second GPU/host telemetry, checksums,
and consolidated CSV/JSON/Markdown summaries.  Fixed seed `20260822`, exact
input/output lengths, forced output length, warmups, and repetitions make runs
directly comparable across future images and forks.
