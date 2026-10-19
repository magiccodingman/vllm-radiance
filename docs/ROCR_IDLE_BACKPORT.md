# Experimental ROCr idle-spin backport

This is a default-off candidate for investigating [Radiance issue #12](https://github.com/magiccodingman/vllm-radiance/issues/12).
It carries three upstream host-runtime fixes on the existing ROCm 7.14.0 source.
The standard Dockerfile, qualified stack, HSA environment and release workflow do not activate it.
The contributor has **no R9700**. GPU execution, idle-CPU improvement, callback latency,
R9700 TP2 correctness/performance and release qualification remain **NOT RUN** until retained maintainer evidence says otherwise.
Do not publish/promote this experimental image as a qualified release from CPU evidence.

## Identity and scope

`backports/rocr-idle/manifest.json` records immutable source, patch and build-package hashes.
The audited base is `rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7`.
Its `/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0` has:

```text
ROCR BUILD ID: 1.21.0-local-build-2b22ab01
SHA256: 15f8c717f4ebf0d00e64b95090466f35fcda78e0a4728fac010af55e91da3395
source: 2b22ab0195cc1461cd9abf3b969e9dd7c10af350
```

The OCI manifest/config digests were verified. The library was extracted from an HTTPS
layer prefix (the entire 7.8 GB layer digest was not independently recomputed) and
matched byte-for-byte to AMD's SHA256-verified `amdrocm-runtime7.14=7.14.0-3` deb.
The binary records AMD clang 23.0.0git at the exact compiler revision in the manifest.
Build SDK packages are pinned to that release with SHA256; no ROCm/compiler upgrade is proposed.
TheRock's `therock-7.14` commit `418cd5f63abb7a604bad5874cd7b2e29334e640f`
points its `rocm-systems` submodule at this exact revision and carries no separate ROCr source patch series.
Its ROCr build selects AMD LLVM, profiler registration and bundled ELF/DRM/NUMA dependencies.
The candidate uses those bundled packages and rejects a changed compiler, lost versioned exports,
changed SONAME, missing profiler support or substituted host ELF/DRM/NUMA libraries.

All three repairs are absent from this source revision:

| Upstream commit / author | Repair | Port |
| --- | --- | --- |
| [c06ea68a](https://github.com/ROCm/rocm-systems/commit/c06ea68a592f6fcf08ada1e10e7b5b54b21aeae7), Ada Rosa | Preserve KFD ages by event identity across compacted wait-slot reordering; clear removed/stale identities | Original patch |
| [46558b7a](https://github.com/ROCm/rocm-systems/commit/46558b7af4dc79b8b8014619c1afdb82db079a9f), Sean Whalen and original coauthors | Back off forced polling from 20 us to 200 us for mixed interrupt batches, or 2 ms when interrupts are globally unavailable | Two context lines retain 7.14's `init_age` state; added upstream code unchanged |
| [8cc8159a](https://github.com/ROCm/rocm-systems/commit/8cc8159ac3e60cb786d21499389bb920035dccb1), Robert Imschweiler | Notify KFD on async registration/shutdown wake stores, closing a lost-wakeup window | Original patch |

Patch headers retain original authors, messages and trailers; the upstream license is included.
Mixed CRLF/LF source requires whitespace-tolerant context matching. Every patched file is
then checked against an exact SHA256 after CRLF-to-LF normalization only; other whitespace/code drift fails.
The initial strict application of the polling patch failed on the two `init_age` context lines;
this mechanical mismatch is the only source adaptation.
The first CPU candidate compiled and loaded but was rejected during dependency review because
plain CMake selected host ELF/DRM/NUMA libraries. The final build explicitly selects the pinned
bundled dependencies and checks the resulting ELF `DT_NEEDED`; that earlier candidate is not promoted.

Related upstream reports include [TheRock #7051](https://github.com/ROCm/TheRock/issues/7051),
[rocm-systems #12674](https://github.com/ROCm/rocm-systems/issues/12674) and
[polling-fix PR #7898](https://github.com/ROCm/rocm-systems/pull/7898).
These support a candidate, not a confirmed root cause for every process named EngineCore in #12.
The polling fix deliberately trades CPU consumption for callback observation latency. Upstream
rejected an earlier 2 ms mixed-batch cap after observing sparse-callback latency around 2 ms;
the accepted mixed cap is 200 us. Preserve that negative result and measure serving latency locally.

## GPU-free validation and build

On Linux x86-64 / Ubuntu 24.04, install `cmake ninja-build g++ libelf-dev libdrm-dev libnuma-dev pkg-config libssl-dev libgtest-dev`.
Use fresh directories so interrupted attempts remain available as evidence:

```bash
python -m unittest discover -s tests -p test_rocr_idle_backport.py -v
python tools/rocr_idle_backport.py prepare /tmp/rocr-idle-source
python tools/check_rocr_idle_source.py /tmp/rocr-idle-source
python tools/rocr_idle_backport.py fetch-sdk /tmp/rocr-idle-sdk
python tools/rocr_idle_backport.py check-runtime /tmp/rocr-idle-sdk/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0
python tools/rocr_idle_backport.py build /tmp/rocr-idle-source /tmp/rocr-idle-build /tmp/rocr-idle-candidate --sdk /tmp/rocr-idle-sdk/opt/rocm/core-7.14
```

The path-selected CPU workflow executes this plan and dynamically loads the candidate without
calling `hsa_init`. CPU contracts compile the actual event-age, handler-removal, polling and
notification bodies against a deterministic KFD stub, and reject three independently reverted
controls. The five original upstream backoff tests run explicitly; an unselected/orphaned rocrtst
target is insufficient. The complete runtime must also compile with profiler registration enabled.
Neither source fragments nor dynamic loading demonstrate real KFD delivery or GPU correctness.

## Explicit candidate image

Build locally on a Linux builder with space for the full pinned ROCm development image:

```bash
docker build -f Dockerfile.rocr-idle \
  --build-arg RADIANCE_IMAGE='your-qualified-radiance-image@sha256:EXACT_DIGEST' \
  -t radiance-rocr-idle:experiment .
```

Use an actual 64-digit digest. The original library must match the audited SHA256 before
replacement. Only the HSA shared object is replaced, including its statically linked matching
thunk; existing symlinks/loader configuration and profiler support are retained. Complete
candidate dynamic loading is checked in the image. No host library is installed by these tools.
Retain the original image for rollback; do not combine this experiment with new HSA environment flags.
The optional image build is a separate candidate-image gate; the CPU workflow does not build a Radiance image.

## Required maintainer hardware evidence

Run matched original/candidate images on the same R9700 host, driver, model, launch flags and
workload, with profiler registration enabled and no workaround flags. Record image digests,
kernel/KFD versions, all relevant environment, actual `/proc/<pid>/maps` HSA path and SHA256,
candidate provenance, per-thread CPU samples and raw logs for both arms.

1. Warm a plain Torch context, synchronize, and hold it idle for at least 120 seconds. Repeat
   after async copies, events and callbacks. Compare each native thread's `/proc/<pid>/task/<tid>/stat`
   user+system CPU deltas over matched idle intervals; process names alone do not identify the cause.
2. Exercise pure-interrupt, IPC/DefaultSignal mixed batches and explicit polling-only waits. Measure
   sparse callback latency (p50/p99/max), including the CPU-vs-latency tradeoff of the two caps.
3. Stress handler registration/removal, wakeup near entry to KFD wait, clean shutdown, event reuse,
   profiler registration and profiling enabled/disabled controls. Reject hangs, missed callbacks,
   crashes or changed profiling behavior; collect symbolized traces for any remaining spin.
4. Warm the same vLLM model at TP1 and R9700 TP2, then sample idle workers and serve matched
   streaming/non-streaming requests, graph capture, long context and repeated start/stop cycles.
   Retain selected correctness/tool/vision/speculative/collective gates and matched throughput,
   TTFT/ITL/TPOT/variance. Add TP4 evidence before claiming TP4 support.
5. Run `python tools/radiance.py test rocr-idle --dry-run`, `python tools/radiance.py qualify upgrade`
   and `python tools/radiance.py qualify release`; retain all applicable gates and negative results.
   CPU success does not authorize production-default promotion or image publication.

If the original does not reproduce #12, record that result and do not describe the candidate as
fixing that host's idle spin. Prefer removing this overlay when an equivalent upstream release
is incorporated and qualified through the normal stack migration policy.
