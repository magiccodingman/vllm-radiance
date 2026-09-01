# ROCm mmap KV-offload registration hardening

Status: implementation and GPU-free source checks complete; dual-R9700
qualification is intentionally pending a production maintenance window.

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

Chunking is implemented but remains opt-in. The 32/36 GiB R9700 failure has not
yet been proven to be a per-call-size ceiling; enabling an arbitrary chunk size
without measurement could add registrations without improving the effective
pin limit.

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

## Maintenance-window qualification plan

Production must be stopped before these steps. The probe refuses to start if
local port 8000 is accepting connections and also requires an explicit
maintenance acknowledgement.

```bash
python benchmarks/bin/probe_rocm_host_registration.py \
  --confirm-maintenance \
  --sizes-gib 24 28 30 32 36 \
  --chunk-gib 0 8 \
  --modes sequential simultaneous \
  --gpus 0 1 \
  --output benchmarks/results/kv-offload-host-registration/probe.json
```

The probe records each rank's registration result, drained error, exact chunks,
registration time, cleanup result, and a post-failure HIP allocation smoke. The
follow-up image qualification must then cover:

1. 24 GiB control startup, restore correctness, and throughput;
2. 32 and 36 GiB with whole registration under `auto`;
3. only chunk sizes that the standalone probe shows can pin both ranks;
4. forced-failure verification that serving falls back without a subsequent
   `hipErrorInvalidValue`;
5. 256K C2/C3/C4 pressure, restore correctness, sustained decode, host-memory
   headroom, and pageable-versus-pinned performance;
6. clean shutdown with no leftover `/dev/shm/vllm_offload_*.mmap` file.

No 32/36 GiB configuration should be called production-qualified until those
GPU gates are complete.
