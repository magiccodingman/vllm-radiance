# Benchmark summary

Medians across repetitions. TPS is output-token throughput; TPOT and TTFT are milliseconds.

| Config | Workload | In/out | C | Temp | N | Output TPS | Total TPS | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Spec accept % | CV % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rx3-production-digest-toolgate-control | smoke | 128/64 | 1 | 0 | 1 | 161.04 | 483.12 | 68.98 | 70.11 | 5.21 | 5.75 | 53.44 | 0.00 |
| rx3-production-digest-toolgate-control-100 | smoke | 128/64 | 1 | 0 | 1 | 148.52 | 445.57 | 113.33 | 129.22 | 5.04 | 5.58 | 53.44 | 0.00 |
| rx4-control-betterbench-standard | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| rx4-control-no-gdnshared-no-topk-toolgate | smoke | 128/64 | 1 | 0 | 1 | 161.99 | 485.97 | 67.16 | 68.14 | 5.20 | 5.75 | 56.08 | 0.00 |
| rx4-control-no-topk-toolgate | smoke | 128/64 | 1 | 0 | 1 | 162.72 | 488.16 | 64.69 | 65.31 | 5.21 | 5.77 | 56.08 | 0.00 |
| rx4-control-toolgate | smoke | 128/64 | 1 | 0 | 1 | 161.59 | 484.77 | 68.05 | 68.55 | 5.20 | 5.75 | 56.08 | 0.00 |
| rx4-core-nopatches-toolgate | smoke | 128/64 | 1 | 0 | 1 | 159.10 | 477.31 | 75.90 | 76.78 | 5.18 | 5.73 | 56.08 | 0.00 |
| rx4-core-old-mxfp4-toolgate | smoke | 128/64 | 1 | 0 | 1 | 162.81 | 488.43 | 64.61 | 65.11 | 5.21 | 5.76 | 56.08 | 0.00 |
| rx4-core-old-python-production-cache-v2-toolgate | smoke | 128/64 | 1 | 0 | 1 | 162.78 | 488.33 | 64.20 | 64.45 | 5.22 | 5.76 | 53.44 | 0.00 |
| rx4-core-old-python-toolgate | smoke | 128/64 | 1 | 0 | 1 | 162.51 | 487.54 | 64.89 | 65.02 | 5.22 | 5.76 | 56.08 | 0.00 |
| rx4-core-old-r4d-v2-toolgate | smoke | 128/64 | 1 | 0 | 1 | 159.53 | 478.60 | 74.43 | 76.30 | 5.18 | 5.73 | 56.08 | 0.00 |
| rx4-core-production-cache-toolgate | smoke | 128/64 | 1 | 0 | 1 | 162.60 | 487.79 | 64.37 | 64.49 | 5.22 | 5.76 | 53.44 | 0.00 |
| rx4-final-default-toolgate-100 | smoke | 128/64 | 1 | 0 | 1 | 162.97 | 488.90 | 64.47 | 64.65 | 5.21 | 5.75 | 53.44 | 0.00 |
| rx4-full-v2-betterbench-standard | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| rx4-v2-legacygraph-toolgate | smoke | 128/64 | 1 | 0 | 1 | 163.21 | 489.63 | 65.11 | 66.13 | 5.19 | 5.74 | 56.08 | 0.00 |
| tool-schema-gate | unknown | / |  |  | 560 |  |  |  |  |  |  |  |  |
