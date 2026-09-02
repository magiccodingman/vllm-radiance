# ROCm KV-offload registration qualification

Date: 2026-09-01 UTC

Branch: `agent/rocm-kv-offload-registration-fix`

Base: `5107c29`; implementation under test: `95eb769` plus the probe
maintenance hardening recorded with this report.

Patched image:
`sha256:53de70ab2b56132d41db5e120073f07965f6fb39dd3ac6296ff40e152a170781`

## Reproducibility

| Component | Value |
|---|---|
| GPU | 2× Radeon AI PRO R9700, gfx1201, 64 CU |
| GPU firmware | ASRock Navi48 XTW 32GB 300W IFWI `00158738`, 2025-07-25 |
| Kernel | Ubuntu `7.0.0-30-generic` |
| vLLM | 0.28.0, pinned commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| PyTorch | 2.12.0+rocm7.14 |
| Triton | 3.7.1 |
| AITER | 0.1.20 |
| ROCm/HIP | 7.14.0 / 7.14.60850 |
| Target model | `Qwen3.8-27B-Quark-AWQ-MXFP4-amd` |
| Target weights SHA-256 | `be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272` |
| Drafter model | `Qwen3.8-27B-DFlash2-FP8-tcclaviger` |
| Drafter weights SHA-256 | `7dbb99a8d0120f502e66b256aa7c0866d933ceeee4a02463d9db591811e8404e` |

Serving used TP2, native MXFP4 W4A8, DFlash K7, mandatory FP8 KV, R4D
attention/GDN/all-reduce, prefix caching with Mamba `align`, PIECEWISE graphs,
256K maximum context, and the normal non-experimental Radiance switches.

## Registration matrix

All 20 combinations of 24/28/30/32/36 GiB, whole/8 GiB registration, and
sequential/simultaneous TP registration completed their post-HIP and cleanup
checks. Both ranks pin through 28 GiB. At 30 GiB and above at least one rank
returns code 1; the error is drained and the next HIP call succeeds. Chunking
does not improve the ceiling.

Raw artifact SHA-256:
`7c0598626b0c334c09b5986280601113e575ccd02112a7d2f6263c463d106b21`.

## Server and pressure gates

| Configuration | Gate | Result | Aggregate output TPS | GPU→CPU | CPU→GPU |
|---|---|---|---:|---:|---:|
| 24 GiB pinned, C4 | 4×261,120 in + 512 out | 4/4 PASS | 3.64 | 89.20 GiB | 0 |
| 36 GiB pageable, C6 | 6×261,120 in + 512 out | 6/6 PASS | 3.47 | 133.87 GiB | 0 |
| 36 GiB pageable, C6 | 6×131,072 prime | 6/6 PASS | 5.15 | 66.34 GiB | 0 |
| 36 GiB pageable, C6 | 6×229,376 eviction | 6/6 PASS | 2.26 | 117.81 GiB | 0 |
| 36 GiB pageable, C6 | 6×135,168 follow-up | 6/6 PASS | 9.77 | 70.07 GiB | 0 |

The 24 GiB result JSON SHA-256 is
`1330ad4baef754b493b858d5df07cfe14bb9b5870fe3123e2370134bbc8b6dbe`.
The 36 GiB result JSON SHA-256 is
`bfa347f245cc41fb16569d349d6544e81aabae829d9f26a3879f8c2a238b3502`.

The deterministic fixed-prompt gate completed 8/8 prompts twice under both
24 GiB pinned and 36 GiB pageable startup. Repetitions within each startup were
byte-identical. There were no allocation failures, preemptions, inherited HIP
errors, or request failures.

## Negative results and deployment decision

- `RADIANCE_KV_OFFLOAD_REGISTER_CHUNK_GIB=8` does not raise the pin ceiling.
- An exact repeated-prefix control still recorded zero CPU→GPU loads and zero
  cached prompt tokens. Native prefix restore is not qualified.
- `docker stop` can leave the mmap file in `/dev/shm`; the test harness removed
  only verified orphans after the container exited.
- Full 256K C4/C6 waves are capacity admissions that queue internally, not four
  or six simultaneously resident full-context streams.

Deploy 24 GiB/C4 with `RADIANCE_KV_OFFLOAD_PIN_POLICY=auto`. The 36 GiB/C6
configuration is now crash-safe but remains a pageable, high-latency capacity
experiment rather than the production default.
