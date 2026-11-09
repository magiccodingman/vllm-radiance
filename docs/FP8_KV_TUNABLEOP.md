# FP8-KV calibration and persisted TunableOp

This continuation adds two independent, opt-in workflows for dual gfx1201
systems. Neither changes the qualified default. FP8-KV calibration targets
attention fidelity; persisted TunableOp targets any residual rocBLAS/
hipBLASLt GEMMs not already served by Radiance's MXFP4, FP8, R4D, or fused
kernels.

The workflows deliberately do not replace Radiance's DFlash fused-KV
construction. The existing identity-through-the-quantized-projection path is
the compatibility-first implementation and remains unchanged.

The feature selection was informed by the public behavior documented for
[`tcclaviger/vllm`](https://hub.docker.com/r/tcclaviger/vllm). The
implementation here is independent: its image did not expose a corresponding
licensed source tree to transplant. TunableOp orchestration follows the
[official PyTorch offline-tuning contract](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md),
and FP8-KV capture uses vLLM's public worker-extension and module-hook APIs.

## Safety contract

- The source model is never modified.
- Calibration outputs are new safetensors sidecars plus checksum/model-bound
  manifests. Existing output names are refused.
- Runtime loading accepts only positive scalar `q_scale`, `k_scale`,
  `v_scale`, and `prob_scale` tensors. Arbitrary checkpoint tensors are
  forbidden.
- Calibration checks finite numeric parameters and FP32 scale representability
  before writing an artifact. Each rank's observations are validated before
  maxima are merged, so another rank cannot hide invalid or unobserved data.
  CPU preflight runs before importing vLLM or constructing the model.
- Sidecars are loaded only when `RADIANCE_FP8_KV_SCALES` is explicitly set.
  Checksum and source-checkpoint verification are on by default.
- TunableOp is off by default. Collection records signatures but never tunes
  in live traffic. Actual tuning is a separate GPU maintenance command.
- Tuned results are checksum-verified and runtime tuning remains disabled in
  serving mode. PyTorch's own CSV validators additionally reject mismatched
  PyTorch, ROCm, hipBLASLt, and rocBLAS stacks.

## FP8-KV calibration

vLLM v0.28 no longer exposes the older one-shot `calc_kv_scales` path. The
Radiance calibrator instead installs a documented vLLM worker extension and
public PyTorch forward-pre-hooks on each full-attention layer. An eager run
observes the BF16 Q/K/V tensors immediately before attention, accumulates one
device scalar per tensor/layer without per-step CPU synchronization, then
reduces maxima across TP ranks.

The default artifact uses OCP E4M3's maximum magnitude 448 with 5% headroom:

```text
scale = observed_absolute_max * 1.05 / 448
```

Softmax probability is mathematically bounded by one, so `prob_scale` uses
`1.05 / 448`. On ROCm's FNUZ representation, vLLM performs its existing
checkpoint-scale conversion during load. The important target values for R4D
are K and V; Q/prob scales also make the artifact safe for supported fallback
attention paths.

Run this only after production has been stopped and the GPUs have been released:

```bash
docker run --rm --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e RADIANCE_MXFP4=1 \
  -e RADIANCE_MXFP4_W4A8=1 \
  -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -v /path/to/models:/models:ro \
  -v /path/to/calibration-output:/calibration:rw \
  -v "$PWD/benchmarks/fixtures:/fixtures:ro" \
  --entrypoint python \
  magiccodingman/vllm-radiance:TAG \
  -m radiance_kv_calibration calibrate \
  --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd \
  --quantization auto \
  --corpus /fixtures/fp8-kv-calibration.jsonl \
  --output /calibration/qwen38-mxfp4-fp8kv-YYYYMMDD.safetensors \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --batch-size 4 \
  --seed 17
```

The JSONL format accepts either `text` or chat-template `messages`, optional
`tools`, and a bounded `repeat` multiplier. The checked-in corpus covers
agent/tool structure, code, reasoning, multilingual text, structured data,
and longer retrieval inputs. A deployment should extend it with representative
licensed/private traffic and retain a disjoint held-out correctness corpus.
Changing even a fine-tune can change activation ranges; calibrate each exact
target checkpoint rather than reusing scales by architecture name.

Verify without starting vLLM:

```bash
docker run --rm \
  -v /path/to/models:/models:ro \
  -v /path/to/calibration-output:/calibration:ro \
  --entrypoint python magiccodingman/vllm-radiance:TAG \
  -m radiance_kv_calibration verify \
  /calibration/qwen38-mxfp4-fp8kv-YYYYMMDD.safetensors \
  --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd
```

Enable it in a private `.env`/Compose override:

```dotenv
RADIANCE_FP8_KV_SCALES=/calibration/qwen38-mxfp4-fp8kv-YYYYMMDD.safetensors
RADIANCE_FP8_KV_SCALES_VERIFY=1
```

Mount the artifact and its sibling `.manifest.json` read-only at that path.
Never make `VERIFY=0` a deployment setting; it exists only to isolate an
intentional mismatch during diagnosis.

Qualification must compare default-scale and calibrated-scale runs with the
same image, target/drafter, prompts, seeds, sampling, graphs, and concurrency.
Required gates are meaningful fixed-prompt greedy output, sampled tool schema,
long-context retrieval, vision inputs when vision is deployed, BetterBench,
DFlash acceptance by category, and sustained c1/c2/c4/c8. Calibration is not a
TPS claim: expected direct throughput change is approximately neutral. Its
success criterion is better or unchanged fidelity, with DFlash acceptance an
important possible secondary effect.

## Persisted TunableOp

This uses PyTorch's standard two-stage offline workflow rather than tuning
inside model loading or production requests.

Tuning requires a namespace with no existing `results*.csv` or manifest.
Completed and partial results, including empty output files from failed runs,
are preserved: select a fresh namespace to retry. A manifest cannot be
overwritten to bless modified results. It is created only when every requested
GPU ordinal has a non-empty result, and its GPU count must be a positive integer.

### 1. Collect signatures

Choose a fresh, explicit namespace and start the candidate with:

```dotenv
RADIANCE_TUNABLEOP_MODE=collect
RADIANCE_TUNABLEOP_CACHE_DIR=/cache/tunableop
RADIANCE_TUNABLEOP_NAMESPACE=IMAGE-PIN-MODEL-PROFILE-YYYYMMDD
```

The entrypoint exports:

```text
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
```

Run the normal warmup plus representative prefill/decode concurrency shapes.
Stop the process gracefully so PyTorch writes `untuned0.csv`,
`untuned1.csv`, and any current results. No candidate timing is valid yet.

### 2. Tune offline

With production stopped and both GPUs free, run the same image and cache mount:

```bash
docker run --rm --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e RADIANCE_VERSION=IMAGE_VERSION \
  -v /path/to/cache:/cache:rw \
  --entrypoint python magiccodingman/vllm-radiance:TAG \
  -m radiance_tunableop tune \
  --root /cache/tunableop/IMAGE-PIN-MODEL-PROFILE-YYYYMMDD \
  --input-pattern 'untuned?.csv' \
  --num-gpus 2 \
  --numerical-check 1e-3_1e-3
```

PyTorch deduplicates the signatures, distributes them over the visible GPUs,
and writes per-device results. Radiance then hashes every non-empty
`results*.csv` into `manifest.json`. The namespace should be treated as
immutable after qualification. The command enables PyTorch's candidate
numerical check at BF16-appropriate `atol=rtol=1e-3`; tighten or loosen it only
as a separately labeled experiment. PyTorch's standard 30 ms / 100-iteration
tuning bounds and L2-sized rotating buffer remain the defaults.

### 3. Serve from verified results

```dotenv
RADIANCE_TUNABLEOP_MODE=serve
RADIANCE_TUNABLEOP_CACHE_DIR=/cache/tunableop
RADIANCE_TUNABLEOP_NAMESPACE=IMAGE-PIN-MODEL-PROFILE-YYYYMMDD
```

Startup refuses a missing/tampered manifest, absent per-GPU results, or a
different Radiance/PyTorch version. PyTorch runtime tuning remains disabled.
Benchmark `serve` against `off`; promote it only if repeated quick/standard
runs show a real improvement and all correctness gates pass. Because the
qualified MXFP4 path already bypasses BLAS for most large projections, the
expected gain is modest and could be zero.

## Qualification status

CPU-only validation covers parsability, sidecar restrictions, exact
checkpoint binding, tamper rejection, TunableOp result completeness, and
tamper rejection:

```bash
python benchmarks/bin/check_fp8_kv_calibration.py
python benchmarks/bin/check_tunableop.py
```

GPU calibration completed on the exact AMD Quark MXFP4 checkpoint with TP2,
R4D attention, seed 17, and the checked-in eight-prompt corpus. The immutable
artifact contains 64 scalar tensors covering Q/K/V/prob scales for all 16
full-attention layers; its checksum and exact source-model binding passed
verification. This proves the capture and artifact pipeline, not a quality
gain. Runtime sidecar promotion still requires held-out quality, long-context,
vision, DFlash-acceptance, and matched BetterBench qualification.

Persisted TunableOp collection and offline tuning remain unpromoted. The
preflight, namespace, completeness, and tamper checks pass, but no performance
claim is made until a populated cache wins a repeated matched benchmark.
