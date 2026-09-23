# Experimental rank-local native CPU KV offload registration

Status: implementation and GPU-free layout/failure-path checks complete. **Dual
Radeon AI PRO R9700 qualification has not been run for this branch.** In
particular, 40 GiB and 45 GiB are qualification targets, not claimed working
capacities.

Pinned vLLM remains
`2cf0a6915ce544dc493a0990f2ea38d81601128a` (v0.28.0). This work builds on the
registration, restore, and lifecycle repairs documented in:

- `docs/ROCM_KV_OFFLOAD_REGISTRATION.md`
- `docs/ROCM_KV_OFFLOAD_RESTORE_BASELINE.md`
- `docs/ROCM_KV_OFFLOAD_LONG_CONTEXT_BASELINE.md`

## Objective and default behavior

The existing native CPU KV mmap is physically block-major. For TP2, each block
row contains rank 0 followed by rank 1. Each worker transfers only its own slot,
but both workers historically call `hipHostRegister` on the entire mmap. The
measured host therefore sees two registrations of the same 24-36 GiB shared
backing.

The experimental mode changes only the **physical placement of the native
direct CPU tier** to rank-major:

```text
rank 0: block 0 | block 1 | ... | block N
rank 1: block 0 | block 1 | ... | block N
```

The shared mmap remains one allocation with the same total byte size. The
logical block count and block IDs remain identical on every worker. The only
change is that each worker's rows are contiguous in one rank-local span and its
CPU tensors use that local row stride.

The feature is off unless explicitly requested:

```bash
RADIANCE_KV_OFFLOAD_RANK_SHARDED=1
```

Ordinary `docker compose up` does not forward or enable that variable. The
optional `docker-compose.kv-rank-sharded.example.yml` overlay forwards it and
still defaults it to `0`.

## Capacity semantics

`--kv-offloading-size` / `RADIANCE_KV_OFFLOADING_SIZE` keeps its existing
meaning: total external-cache capacity before vLLM's existing whole-block
rounding. Rank sharding does not multiply the requested capacity by TP size.

For a TP2 mmap whose final physical size is 40 GiB, the planned registration is
20 GiB per worker. For 45 GiB, it is 22.5 GiB per worker. Runtime mmap size can
remain slightly below the requested capacity because the pinned vLLM code
already floors the block count to whole aligned rows; logs therefore report the
actual rank-local byte span, not a promise that the requested GiB divides with
zero remainder.

## Safety invariants

The implementation deliberately reuses the existing registration hardening:

1. `hipHostRegister`, `hipGetLastError`, and `hipHostUnregister` still use the
   same `CudaRTLibrary` / `libamdhip64` handle.
2. A failed registration is drained immediately.
3. Registration remains serialized and coordinated across TP/PCP/PP workers.
4. One-rank failure still rolls every successfully registered rank back before
   the group falls back to pageable DMA under the `auto` policy.
5. Pageable CPU memory still forces the DMA copy implementation in both
   directions where required by the existing hardening.
6. Successful chunks are stored as offsets from the **full mmap base** even
   though registration begins at a rank-local pointer. Existing rollback and
   shutdown cleanup therefore own exactly the ranges that were registered.
7. The unlink-after-rendezvous shared-memory lifecycle is unchanged: every
   worker maps the one shared file, the creator unlinks the pathname after the
   barrier, and pages are reclaimed when the last mapping closes.

The overlay does not modify Radiance FP8, MXFP4/W4A8, R4D attention/GDN,
DFlash, speculative decoding, or all-reduce paths.

## Canonical and hybrid handling

The pinned vLLM canonical secondary-tier contract is intentionally **not**
transposed. `create_kv_memoryview()` exposes a zero-copy 2-D block-major
`memoryview` in which downstream tiers address block `b` as `view[b]`. A
rank-major physical buffer cannot satisfy that contract with a normal
contiguous Python `memoryview`.

Therefore only `CPUOffloadingSpec`'s native direct-worker region opts into
rank-major mode. Tiering/canonical callers continue constructing
`SharedOffloadRegion(..., rank_sharded=False)` and retain the legacy physical
layout, canonical mappings, writer rotation, and hybrid-model behavior. Guards
reject accidental calls to `create_next_canonical_view()` or
`create_kv_memoryview()` on a rank-major region instead of returning a
non-contiguous object that downstream code may misinterpret.

The opt-in also fails closed for vLLM's replicated CPU layout because that
layout intentionally has all ranks share one canonical slot; giving each rank a
private span would change its ownership semantics.

## Geometry requirements

The rank-major planner requires the already-aligned global block-row stride to
divide evenly by model-parallel world size, and each rank-local row stride must
be page aligned. This ensures rank registration ranges never share a host page.
Unsupported geometry raises only when the experimental flag is enabled; the
default block-major path is unchanged.

## GPU-free checks

Run both existing registration checks and the new layout checks from the repo
root:

```bash
python benchmarks/bin/check_kv_offload_registration.py
python benchmarks/bin/check_kv_offload_rank_sharded.py
```

The rank-sharded checks cover:

- TP2 40 GiB -> exactly two contiguous 20 GiB synthetic ranges;
- TP2 45 GiB -> exactly two contiguous 22.5 GiB synthetic ranges;
- disjoint ranges whose summed physical allocation equals the input total;
- logical block/view byte addressing under rank-major placement;
- simulated one-rank registration failure followed by successful-rank rollback;
- shutdown-style unregister ownership constrained to exactly the registered
  rank span.

These checks prove layout arithmetic and failure-path ownership only. They are
not a substitute for HIP registration on the R9700 host.

## Dual-R9700 qualification

Run the following only in a declared maintenance window with the serving API
stopped. The probe refuses to start if local port 8000 is accepting
connections.

First compare the historical full-shared registration with rank-sharded
registration at the target sizes:

```bash
python benchmarks/bin/probe_rocm_host_registration.py \
  --confirm-maintenance \
  --sizes-gib 40 45 \
  --chunk-gib 0 \
  --layouts shared rank-sharded \
  --modes sequential simultaneous \
  --prefault distributed \
  --gpus 0 1 \
  --output benchmarks/results/kv-offload-host-registration/rank-sharded-40-45.json
```

For `rank-sharded`, confirm in the JSON that rank 0 and rank 1 have disjoint
`registration_offset`/`registration_size` intervals, their sizes sum to the
case's mmap size, both `register_ok` values are true, the post-registration HIP
allocation succeeds, and both `cleanup_ok` values are true. A failure is a
qualification result, not a reason to weaken rollback or error draining.

Build a candidate image from the current local `vllm-radiance:dev` base:

```bash
docker build \
  -f Dockerfile.patch \
  -t vllm-radiance:kv-rank-sharded \
  .
```

If the host's qualified development base uses another tag, pass it explicitly
with `--build-arg BASE_IMAGE=<qualified-tag>`; do not change the vLLM commit.

Start the 40 GiB candidate with the optional Compose overlay:

The restore gate requires `VLLM_SERVER_DEV_MODE=1`; this is a qualification-only endpoint and must remain disabled in normal serving.

```bash
IMAGE=vllm-radiance:kv-rank-sharded \
RADIANCE_KV_OFFLOADING_SIZE=40 \
RADIANCE_KV_OFFLOAD_RANK_SHARDED=1 \
VLLM_SERVER_DEV_MODE=1 \
docker compose \
  -f docker-compose.yml \
  -f docker-compose.kv-rank-sharded.example.yml \
  up -d
```

Follow startup and confirm that both TP workers report rank-local registration,
not the complete shared mmap:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.kv-rank-sharded.example.yml \
  logs -f vllm
```

Then run the established deterministic restore gate:

```bash
python benchmarks/bin/run_kv_offload_restore_gate.py \
  --model Qwen3.8-27B \
  --output benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-rank-sharded40/restore-gate.json
```

Stop the candidate before changing capacity:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.kv-rank-sharded.example.yml \
  down
```

Repeat startup at 45 GiB:

```bash
IMAGE=vllm-radiance:kv-rank-sharded \
RADIANCE_KV_OFFLOADING_SIZE=45 \
RADIANCE_KV_OFFLOAD_RANK_SHARDED=1 \
VLLM_SERVER_DEV_MODE=1 \
docker compose \
  -f docker-compose.yml \
  -f docker-compose.kv-rank-sharded.example.yml \
  up -d

python benchmarks/bin/run_kv_offload_restore_gate.py \
  --model Qwen3.8-27B \
  --output benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-rank-sharded45/restore-gate.json
```

After correctness passes, use the existing long-context harness rather than
inventing a new workload:

```bash
python benchmarks/bin/run_kv_offload_long_context.py \
  --model Qwen3.8-27B \
  --tokenizer /models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd \
  --output-dir benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-kv-long-rank-sharded \
  --label rank-sharded \
  --cases 131072:1 131072:2 131072:4 262144:1 262144:2 262144:3 262144:4 \
  --max-tokens 256 --cpu-restore --continue-on-error
```

Qualification should also confirm ordinary shutdown leaves no named
`/dev/shm/vllm_offload_*.mmap` and that a forced registration failure under
`RADIANCE_KV_OFFLOAD_PIN_POLICY=auto` rolls every rank back to pageable DMA
without a later stale HIP error.

## Qualification boundary

Until the commands above pass on both R9700s, keep
`RADIANCE_KV_OFFLOAD_RANK_SHARDED=0` in normal deployments. This branch makes a
layout and registration hypothesis testable; it does **not** establish that 40
GiB or 45 GiB is a safe production setting.
