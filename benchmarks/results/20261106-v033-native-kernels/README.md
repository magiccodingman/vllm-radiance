# v0.33 candidate: native kernel preflight on GPU 0

**Native-kernel evidence only. Full candidate Torch/vLLM integration and real
model serving are still pending. No TP2, capacity, performance or release claim.**

| Gate | Cases | Largest relative error | Result |
|---|---:|---:|---|
| Maintained paged attention | 12 | 0.1771% | FP64 oracle, guards and changed-input graphs pass |
| Maintained GDN scan | 24 | 0.3571% | FP64 output/state recurrence, guards and changed-input graphs pass |
| Radiance MXFP4 × FP8 GEMM | 16 | 0.1682% | FP64 stored-operand dot product, guards, decode-counter reset and changed-input graphs pass |

Every numerical case stays below the fixed 1% relative-error gate. Attention
and GEMM additionally bound maximum absolute error. The previous local image
also passes the paged-attention oracle control. The native runs are independent
correctness checks, not matched performance comparisons.

## What ran

One AMD Radeon AI PRO R9700 / gfx1201, PCI `0000:03:00.0`, exposed through its
`renderD128` node and `/dev/kfd`. Both ROCm/HIP visibility variables select device
0, and each GPU process asserts exactly one visible compute device. Other GPUs,
production containers and host GPU settings are unchanged.

The GPU processes run in the exact ROCm 10.0 dev-base digest in
[manifest.json](manifest.json), using the compiled maintained extension and a
newly compiled Radiance MXFP4 extension. They load NumPy and the HIP runtime;
**they do not load the old Torch runtime into the new ROCm process**. CPU-only
fixture producers use the pinned previous image and independent FP64 formulas.
The manifest records component refs, source hashes, binary hashes and results.
No binary or large fixture archive is checked in.

- Attention covers BF16 queries with BF16 and FP8 caches, shuffled pages,
  partial pages/tiles, unequal context lengths and decode/prefill widths
  1/7/10/17/65/129. Native output guards and changed-query graph replay pass.
- GDN runs every original extreme-decay case in
  `benchmarks/bin/check_gdn_extreme_decay.py`, for both 48/16 and 24/8 head
  geometries. It includes partial chunks, unequal sequence lengths, severe
  decay, mixed affected/unaffected heads and large value amplitude. Output,
  final state and three changed-query graph replays pass. Both geometries run
  on one GPU: this is **not distributed TP2 qualification**.
- MXFP4 tests folded and per-block scale paths, the M=64/65 boundary, K=192
  (divisible by 64 but not 128), partial N, N=48 gates, large-M/TN4 prefill and
  TP1-wide N=34816. Activation, E2M1 weight and E8M0 scale values are decoded
  independently for the FP64 reference. Output/partial/counter guards and
  counter reset across repeated graph replay pass. Loader/AITER/framework
  selection remains an installed-image gate.

## Evidence and reproduction

Each family retains its CPU producer, raw-HIP runner, fixture manifest and
numerical results. The GDN and MXFP4 runners reuse the attention runner's HIP
memory/graph bindings through `/helpers/run_raw_hip.py`.

For reproduction, use fresh writable output directories rather than modifying
this immutable record. Copy each family's two Python scripts there. Mount the
repository at `/repo:ro` and the family output directory at `/out` in the pinned
CPU producer image; run `/out/make_fixtures.py` with its `/opt/vllm/bin/python`.
The generated archives contain exact inputs and independent FP64 expectations.

Build the maintained extension from the legacy and maintained commits in the
manifest, applying `r4d_radiance_extras.patch` and `tools/prepare_r4d.py`, using
ROCm 10 and pybind11 3.0.4. Compile `radiance_mxfp4_fp8.hip` with the same compiler
and `--offload-arch=gfx1201`, as in Dockerfile. Mount the maintained extension
directory at `/probe:ro`; place the MXFP4 extension in its writable `/out`.
Mount the attention harness directory at `/helpers:ro`, and a compatible Python
3.12 NumPy installation at `/host-packages:ro`.

The raw runner's container requires:

```text
--network none --device=/dev/kfd --device=/dev/dri/renderD128
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/opt/rocm/lib
PYTHONPATH=/out:/probe:/host-packages
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RADIANCE_MXFP4_DECODE_MAX_M=64 RADIANCE_MXFP4_EPIFAST=1 RADIANCE_MXFP4_WPERM=0
```

Confirm the physical PCI-to-render mapping before using a render-node number on
another host. Invoke the dev base's `python3 /out/run_raw_hip.py`. Preserve fresh
logs and results, then repeat the registered framework-native gates in the
actual final image. This preflight cannot substitute for those checks.

## Retained negative evidence

The first attention harness invocation omitted selector `q_len`/FP8 dtype keys
and failed before numerical kernel execution. The corrected harness passes the
old-image control and new ROCm native runs. Separately, the unadapted official
LFM3 config failed a pinned Transformers 5.16.1 representation check; the narrow
config adaptation and a real pinned-Transformers check are documented in
`docs/LFM3.md`. Original setup logs remain in the task's scratch audit.
