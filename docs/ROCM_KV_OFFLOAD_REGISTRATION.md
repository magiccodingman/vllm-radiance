# ROCm mmap KV-offload registration hardening

Status: implementation, standalone dual-R9700 registration matrix, patched-
image startup, correctness, and bounded long-context pressure gates complete.
The host-registration failure is fixed. Native restore and mmap lifecycle work
continued in `docs/ROCM_KV_OFFLOAD_RESTORE_BASELINE.md`.

Development branch: `agent/rocm-kv-offload-registration-fix`, based on merged
`main` commit `5107c29` and pinned vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a`
(v0.28.0).

## Incident

The vLLM v0.28.0 mmap-backed CPU KV tier is healthy at the deployed 24 GiB
setting, but larger startup experiments exposed an unsafe best-effort failure:

| Requested offload | Shared mmap | Observed startup result |
|---:|---:|---|
| 24 GiB | 25.76 GB | Both TP ranks registered; server healthy |
| 32 GiB | 34.35 GB | One rank returned host-register code 1; its next torch operation failed with `hipErrorInvalidValue` |
| 36 GiB | 38.64 GB | Same failure, with the affected TP rank varying between runs |

Unlimited Docker `memlock` did not change the 32/36 GiB result. This points
away from the container's RLIMIT and toward runtime/shared-backing behavior.
The old vLLM function logged that pageable DMA would be used after a failed
`cudaHostRegister`, but did not consume the pending thread-local HIP error.
The next unrelated torch runtime call therefore inherited the stale error.
Independent registration in both TP processes could also leave one rank pinned
and the other pageable.

## Implemented behavior

`patch_kv_offload_registration.py` is an exact-anchor overlay for the pinned
vLLM v0.28.0 source. It installs the following invariants:

1. Host register, pending-error drain, and unregister use the same
   `CudaRTLibrary`/`libamdhip64` handle.
2. A nonzero `hipHostRegister` is drained immediately, before any later torch
   operation.
3. TP/PCP/PP ranks finish mapping first, then register in deterministic model-
   parallel-rank order.
4. Registration is group-atomic: either every rank remains pinned or every
   successful rank rolls back and the group uses pageable DMA.
5. Pageable memory is forced through the DMA copy implementation. GPU kernels
   are never allowed to dereference it directly.
6. Every successful row-aligned registration chunk is retained for exact
   reverse-order rollback and shutdown cleanup.

The implementation draws narrowly from these active upstream efforts rather
than importing an unfinished branch wholesale:

- [vLLM issue #51762](https://github.com/vllm-project/vllm/issues/51762) — mixed pinned/pageable TP workers;
- [vLLM PR #50070](https://github.com/vllm-project/vllm/pull/50070) — same-handle pending-error drain;
- [vLLM PR #52296](https://github.com/vllm-project/vllm/pull/52296) — coordinated model-parallel registration and rollback;
- [vLLM PR #51081](https://github.com/vllm-project/vllm/pull/51081) — row-aligned chunk tracking.

## Runtime controls

| Variable | Default | Meaning |
|---|---:|---|
| `RADIANCE_KV_OFFLOADING_SIZE` | unset | Compose-friendly value for vLLM `--kv-offloading-size`, in GiB |
| `RADIANCE_KV_OFFLOAD_PIN_POLICY` | `auto` | `auto`, `required`, or `disabled` |
| `RADIANCE_KV_OFFLOAD_REGISTER_CHUNK_GIB` | `0` | Registration chunk size; `0` preserves one whole-region call |

`auto` is the safe production policy. An ordinary registration failure is
reported and drained, every successful rank rolls back, and serving continues
with slower pageable DMA. `required` makes the same condition a startup error.
`disabled` is a diagnostic control.

Chunking is implemented but remains opt-in. The measured 8 GiB registration
chunks did not raise the effective pin limit on this host, so the shipped
default remains one whole-region call.

## GPU-free validation completed

- helper syntax/import compilation;
- policy validation and GiB parsing;
- complete row-aligned chunk coverage;
- successful multi-chunk registration;
- failed registration drains before rollback;
- rollback retains ownership of any chunk that could not be unregistered;
- exact overlay applies cleanly to all three vLLM v0.28.0 source files;
- a second overlay application is a complete no-op;
- patched sources parse as valid Python;
- Compose and entrypoint syntax validation (recorded in the merge request).

Run the pure regression check with:

```bash
python benchmarks/bin/check_kv_offload_registration.py
```

## Dual-R9700 host-registration matrix

The representative probe was run inside the root ROCm production image with
host IPC, both GPUs, distributed pre-faulting, and the mmap resident before
registration. Every case also performed a post-registration HIP allocation and
exact unregister/rollback cleanup.

| mmap | Whole, sequential | Whole, simultaneous | 8 GiB chunks, sequential | 8 GiB chunks, simultaneous |
|---:|---|---|---|---|
| 24 GiB | both pinned | both pinned | both pinned | both pinned |
| 28 GiB | both pinned | both pinned | both pinned | both pinned |
| 30 GiB | one failed | one failed | one failed | one failed |
| 32 GiB | one failed | one failed | one failed | one failed |
| 36 GiB | one failed | one failed | one failed | both failed |

Every failed registration returned code 1, `hipGetLastError` drained code 1,
the following HIP allocation succeeded, and cleanup succeeded. Which rank won
could change under simultaneous registration. The threshold is therefore a
shared-backing/global pin-accounting ceiling between 28 and 30 GiB, not a
single-call-size limit; chunking is not a workaround.

The raw matrix is retained on the qualification host as
`/nvme/ediloca-1/scratch/kv-offload-host-registration-container-matrix-20260901.json`
with SHA-256
`7c0598626b0c334c09b5986280601113e575ccd02112a7d2f6263c463d106b21`.

To reproduce it during a maintenance window, run the probe from inside the
same root ROCm image used for serving. The probe refuses to start if local port
8000 is accepting connections and requires an explicit acknowledgement.

```bash
python benchmarks/bin/probe_rocm_host_registration.py \
  --confirm-maintenance \
  --sizes-gib 24 28 30 32 36 \
  --chunk-gib 0 8 \
  --modes sequential simultaneous \
  --prefault distributed \
  --gpus 0 1 \
  --output benchmarks/results/kv-offload-host-registration/probe.json
```

## Patched-image qualification

Image `sha256:53de70ab2b56132d41db5e120073f07965f6fb39dd3ac6296ff40e152a170781`
used vLLM 0.28.0, PyTorch 2.12.0+rocm7.14, Triton 3.7.1, AITER 0.1.20,
ROCm 7.14, and the normal MXFP4+DFlash K7 production profile.

- At 24 GiB, both TP ranks pinned the 25.76 GB shared mmap. The server became
  healthy and eight fixed prompts were byte-identical across two repetitions.
- At 36 GiB under `auto`, rank 1 returned code 1 and drained code 1. Rank 0
  rolled its successful registration back, both workers selected pageable DMA,
  and startup completed without a later `hipErrorInvalidValue`.
- At 36 GiB under `required`, both workers failed startup with the explicit
  `KV mmap registration is required` error.
- The bounded 256K pressure gates completed 4/4 streams at 24 GiB/C4 and 6/6
  at 36 GiB/C6, with zero allocation failures and zero preemptions.
- A matched hot short-decode sweep showed no material pageable penalty because
  it did not load KV from CPU: 36 GiB versus 24 GiB was -0.28% at C1, +0.54%
  at C4, and +1.09% at C6.

Full 256K results are capacity/queue measurements, not six simultaneous
resident decodes. The 24 GiB/C4 wave admitted roughly two full streams at once
and produced 3.64 aggregate cold-wave output TPS. The 36 GiB/C6 wave likewise
staged roughly two at a time and produced 3.47 TPS. This makes 36 GiB/C6 useful
as a safe admission ceiling, not a latency-oriented production recommendation.

## Superseded native-offload limitations

This registration checkpoint originally recorded zero CPU hits and a leaked
named mmap. Both root causes were repaired and requalified by the continuation
report in `docs/ROCM_KV_OFFLOAD_RESTORE_BASELINE.md`. Keep this section as the
historical observation that motivated that work, not as current behavior.

The merged upstream unlink-after-rendezvous lifecycle repair is now backported,
and the selective DFlash/MTP group annotation restores real CPU hits. DFlash
strict equivalence remains unqualified for a separate speculative/APC reason;
non-spec and MTP pass the strict restore gate.

The production recommendation remains 24 GiB/C4 with `auto`: it pins on this
host and retains the established 256K admission envelope. Values at or above
30 GiB should be treated as slower pageable capacity experiments.
