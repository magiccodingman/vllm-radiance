# Benchmark summary

Medians across repetitions. TPS is output-token throughput; TPOT and TTFT are milliseconds.

| Config | Workload | In/out | C | Temp | N | Output TPS | Total TPS | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Spec accept % | CV % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention-bf16-control | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| attention-bf16-control | prefill | 2048/64 | 1 | 0 | 2 | 37.83 | 1248.38 | 459.11 | 464.58 | 19.57 | 19.70 |  | 0.81 |
| attention-bf16-control | prefill | 2048/64 | 4 | 0 | 2 | 78.45 | 2588.81 | 1343.55 | 1802.27 | 31.77 | 45.12 |  | 4.21 |
| attention-bf16-control | prefill | 2048/64 | 8 | 0 | 2 | 100.62 | 3320.30 | 2029.65 | 3569.44 | 51.86 | 72.86 |  | 0.06 |
| attention-bf16-control | warmup | 128/64 | 1 | 0 | 1 | 48.59 | 145.78 | 107.70 | 117.67 | 19.19 | 19.23 |  | 0.00 |
| attention-fp8qk-pv-candidate | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| attention-fp8qk-pv-candidate | prefill | 2048/64 | 1 | 0 | 2 | 37.75 | 1245.88 | 469.99 | 478.89 | 19.45 | 19.49 |  | 1.74 |
| attention-fp8qk-pv-candidate | prefill | 2048/64 | 4 | 0 | 2 | 80.87 | 2668.69 | 1336.87 | 1796.30 | 28.62 | 42.35 |  | 0.01 |
| attention-fp8qk-pv-candidate | prefill | 2048/64 | 8 | 0 | 2 | 100.91 | 3329.93 | 2024.17 | 3556.98 | 51.68 | 72.65 |  | 0.13 |
| attention-fp8qk-pv-candidate | warmup | 128/64 | 1 | 0 | 1 | 47.41 | 142.22 | 124.75 | 139.20 | 19.45 | 19.49 |  | 0.00 |
| candidate-dflash-betterbench-standard | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| candidate-dflash-betterbench-standard-warm | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| candidate-dflash-smoke | smoke | 128/64 | 1 | 0 | 1 | 150.10 | 450.29 | 109.93 | 126.04 | 5.02 | 5.57 | 53.44 | 0.00 |
| candidate-nonspec-smoke | smoke | 128/64 | 1 | 0 | 1 | 46.87 | 140.61 | 66.80 | 67.30 | 20.61 | 20.64 |  | 0.00 |
| dynwidth-min5-c4c8-final | decode | 256/256 | 4 | 0 | 3 | 321.60 | 643.20 | 193.57 | 317.13 | 8.24 | 15.79 | 47.73 | 22.53 |
| dynwidth-min5-c4c8-final | decode | 256/256 | 8 | 0 | 3 | 508.55 | 1017.11 | 250.09 | 536.21 | 8.97 | 27.02 | 56.45 | 3.39 |
| dynwidth-min5-c4c8-final | warmup | 128/64 | 1 | 0 | 1 | 151.50 | 454.50 | 101.57 | 114.24 | 5.09 | 5.63 | 53.44 | 0.00 |
| dynwidth-off-c4c8-control | decode | 256/256 | 4 | 0 | 3 | 402.16 | 804.32 | 186.32 | 285.67 | 7.38 | 15.79 | 47.73 | 24.27 |
| dynwidth-off-c4c8-control | decode | 256/256 | 8 | 0 | 3 | 466.70 | 933.39 | 246.29 | 513.65 | 10.18 | 29.42 | 41.52 | 5.46 |
| dynwidth-off-c4c8-control | warmup | 128/64 | 1 | 0 | 1 | 162.35 | 487.04 | 64.79 | 65.29 | 5.23 | 5.77 | 53.44 | 0.00 |
| dynwidth-on-c4c8-candidate | decode | 256/256 | 4 | 0 | 3 | 392.12 | 784.24 | 193.19 | 322.39 | 6.92 | 15.72 | 64.72 | 22.72 |
| dynwidth-on-c4c8-candidate | decode | 256/256 | 8 | 0 | 3 | 511.52 | 1023.03 | 251.17 | 539.13 | 8.96 | 26.69 | 59.92 | 3.94 |
| dynwidth-on-c4c8-candidate | warmup | 128/64 | 1 | 0 | 1 | 146.91 | 440.72 | 119.67 | 139.38 | 5.01 | 5.55 | 53.44 | 0.00 |
| final-dflash-betterbench-standard-min5 | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| final-image-dflash-smoke | smoke | 128/64 | 1 | 0 | 1 | 162.69 | 488.06 | 65.09 | 65.15 | 5.21 | 5.76 | 53.44 | 0.00 |
| tool-schema-gate | unknown | / |  |  | 90 |  |  |  |  |  |  |  |  |
