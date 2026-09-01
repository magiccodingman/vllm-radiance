# Native ROCm KV restore repairs and first baseline

Status: correctness and lifecycle repairs implemented; non-spec and MTP restore
qualified; DFlash restore operational but not strict-equivalence qualified;
first C1/C2/C4 shared-prefix baseline complete. Performance tuning is explicitly
out of scope for this checkpoint.

Branch: `agent/rocm-kv-offload-registration-fix`, based on merged `main`
`5107c29`. Pinned vLLM: `2cf0a6915ce544dc493a0990f2ea38d81601128a`
(v0.28.0).

## Reproducibility manifest

- Source-consistent candidate image: `vllm-radiance:kv-restore-candidate`
- Candidate digest: `sha256:5f0f77117efc7e4fdf77959d700be6fce304b5341d77163c281152613a42ea59`
- Benchmark image digest:
  `sha256:bf5f98542f2787baf4980b58dc2302d32a21c54571ffaee43b6cf7aec4f076c9`.
  Its executable and Python layers are identical to the candidate; the final
  rebuild only normalized the Docker build-argument version string to the
  already-recorded `/opt/radiance_version` value.
- Radiance version:
  `0.9.3-dev.vllm0.28.0-r4d0.5.0-mxfp4.rx4.dflash2.xgrammar.openobj.kvoffload.restore2`
- vLLM 0.28.0, PyTorch 2.12.0+rocm7.14, AMD Triton
  3.7.1+gitf0b55c07, AITER 0.1.20, ROCm 7.14, libr4d 0.5.0,
  transformers 5.15.0, XGrammar 0.2.3.
- Host kernel/amdgpu: Ubuntu kernel `7.0.0-30-generic`.
- GPUs: two ASRock Radeon AI PRO R9700, gfx1201, 64 CUs each; IFWI
  `00158738`, built 2025-07-25.
- Deployment envelope: TP2, 90% GPU allocation, FP8 KV, 24 GiB native CPU
  offload, 256K maximum model length, C4 admission, prefix caching on,
  piecewise graphs, R4D attention, MXFP4 target, DFlash K7 unless a control
  explicitly says otherwise.

The source-consistent candidate became healthy and passed the 10.5K-token
non-spec cold/GPU/CPU smoke gate under run
`20260901T231624Z-candidate-nonspec-smoke`.

Target checkpoint:
`/nvme/lexar-2/ai/models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd`
(19,798,196,184 weight bytes). SHA-256: `config.json`
`04c9b07a3a9260cbc8a2ea5b5e5f84ced8274cf412deb9895e91204383ed20e3`,
`generation_config.json`
`07f857aba5260b2ea2513f80de8062d086661f00eada0a3794964e665ba680f5`,
and `tokenizer_config.json`
`b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`.

DFlash checkpoint:
`/nvme/lexar-2/ai/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger`
(2,118,882,784 weight bytes). `config.json` SHA-256:
`5b5668a00b26aaebd88c7e3d961f7d1cdef025867fee158dfccb84f29fd8caec`.

## Root causes and repairs

### Selective speculative cache-group annotation

Pinned vLLM conservatively marked every hybrid cache group as EAGLE when no
group carried an explicit annotation. Native offload therefore stored data but
could not satisfy a restore on Qwen hybrid models. This is the failure family
reported in [vLLM #54360](https://github.com/vllm-project/vllm/issues/54360),
[vLLM #52735](https://github.com/vllm-project/vllm/issues/52735), and
[vLLM #53670](https://github.com/vllm-project/vllm/issues/53670).

`patch_kv_offload_restore.py` now identifies only real draft attention:

- DFlash layers appended after the 64 target layers: group `[8]`, not all nine
  hybrid groups;
- embedded Qwen MTP layers under `mtp.layers.*`: group `[3]`, not all four
  hybrid groups;
- the upstream DSpark marker and DeepSeek-V4 fallback remain intact.

The pre-fix MTP run `20260901T225345Z-mtp-k4-final-restore` recorded zero
external hits and no CPU→GPU bytes. The post-fix full run
`20260901T230220Z-mtp-k4-final-restore2` restored 38,784 of 41,992 prompt
tokens, moved 1,508,179,968 bytes CPU→GPU in 53.1 ms, and was byte-identical
across cold, GPU-hit, and CPU-restore paths.

### Shared-prefix load convoy

The native scheduler previously delayed every request sharing chunks with an
in-flight CPU load, the convoy documented by
[vLLM #44294](https://github.com/vllm-project/vllm/issues/44294) and proposed
fix [#44295](https://github.com/vllm-project/vllm/pull/44295). The bounded
backport keeps duplicate CPU loads prohibited but lets later requests recompute
while the first transfer warms GPU APC.

### Shared-memory lifecycle

Ordinary Compose shutdown left a root-owned 24 GiB
`/dev/shm/vllm_offload_*.mmap`, which could make the next identical deployment
fail before model load. The image carries the exact narrow repair merged in
[vLLM #52596](https://github.com/vllm-project/vllm/pull/52596), commit
`4c58a0c398b056b135b98bd93c644945be7e3109`: all workers map, synchronize,
then the creator unlinks the name. Mappings stay valid until their last process
exits, including SIGKILL.

The upstream lifecycle test selection passed `6 passed, 34 deselected`.
During live TP2 startup the file was unlinked after both workers mapped it;
normal Compose shutdown returned `/dev/shm` to about 118 MiB used and left no
named offload mmap.

## Deterministic restore qualification

`run_kv_offload_restore_gate.py` performs an external reset, cold fill, local
GPU hit, local-only reset, then CPU restore. It uses a fixed 41,992-token
meaningful prompt, temperature 0, seed 17, and immutable JSON output.

| Mode | Run ID | Cold | GPU hit | CPU restore | Restored tokens | Strict result |
|---|---|---:|---:|---:|---:|---|
| Non-spec | `20260901T230936Z-nonspec-final2-restore` | 13.654 s | 1.160 s | 1.195 s | 40,768 | pass |
| MTP K4 | `20260901T230220Z-mtp-k4-final-restore2` | 11.361 s | 1.351 s | 1.362 s | 38,784 | pass |
| DFlash K7 | `20260901T230507Z-dflash-final2-restore` | 11.176 s | 1.215 s | 1.052 s | 39,552 | fail |

The DFlash failure is not a missing restore: it moved 1,490,616,320 bytes in
52.7 ms and served 39,552 external tokens. The three outputs remained coherent
but chose different near-synonymous continuations. The same strict divergence
also occurs on the local GPU APC hit, and conservative one-page and two-page
Mamba recompute experiments did not remove it. Those unproven experiments are
recorded in the immutable negative runs and are not shipped.

Therefore DFlash+APC/native offload remains **not lossless-qualified**. This is
consistent with the unresolved hybrid speculative APC correctness reports in
[vLLM #53912](https://github.com/vllm-project/vllm/issues/53912) and
[vLLM #50188](https://github.com/vllm-project/vllm/issues/50188). It must not be
described as byte-identical merely because restore is now fast and functional.
The live structured-tool regression lane nevertheless passed 30/30 on final2:
`20260901T230645Z_kv-offload-final2-tool-schema`.

## First CPU-restore/convoy baseline

Run: `20260901T230544Z-dflash-cpu-kv-baseline-final2`.

Each case used its own 16,390-token prompt, primed CPU once, reset only local
GPU APC, then released an identical-prefix request wave at a barrier. Output
length was 64 tokens/request, temperature 0, seed 17. This is a restore and
scheduler baseline, not BetterBench model-quality or maximum-throughput data.

| C | Median TTFT | Max TTFT | Aggregate output TPS | Median request | CPU→GPU | Load ops | External hits | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.814 s | 0.814 s | 46.22 | 1.385 s | 626.6 MB | 2 | 13,184 | 29.9% |
| 2 | 1.267 s | 1.714 s | 59.21 | 2.121 s | 626.6 MB | 2 | 13,184 | 49.8% |
| 4 | 2.446 s | 3.350 s | 43.04 | 3.756 s | 626.6 MB | 2 | 13,184 | 28.1% |

Two load operations are one logical restore across two TP ranks. The operation
count, transferred bytes, and external-hit tokens remained constant from C1
through C4. This proves duplicate CPU loads were suppressed without queuing the
entire wave behind the transfer. Additional external queries were exactly the
3,206-token suffix recomputed by each duplicate request.

C4 is slower than C2 because three requests recompute that suffix concurrently
and DFlash acceptance fell. That is the baseline opportunity for the next
tuning phase, not a result to hide or optimize inside this repair checkpoint.
A prior repeat, `20260901T224102Z-dflash-cpu-kv-baseline`, measured
46.11/59.39/42.05 TPS and demonstrates close repeatability.

## Reproduction

The development-only reset route requires `VLLM_SERVER_DEV_MODE=1`; do not add
it to production Compose.

```bash
python benchmarks/bin/check_kv_offload_restore.py

python benchmarks/bin/run_kv_offload_restore_gate.py \
  --model Qwen3.8-27B \
  --output benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-restore/restore-gate.json

python benchmarks/bin/run_kv_offload_baseline.py \
  --model Qwen3.8-27B \
  --output benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-baseline/baseline.json
```

## Qualification boundary

This checkpoint repairs restore, shared-memory cleanup, and convoy scheduling.
It does not tune offload size, transfer chunking, prefetch, NUMA placement, or
concurrency policy. The established 24 GiB/C4 production recommendation remains
unchanged until the next phase compares those controls against this baseline.
