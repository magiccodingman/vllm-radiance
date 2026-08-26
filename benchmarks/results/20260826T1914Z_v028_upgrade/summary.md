# Benchmark summary

Medians across repetitions. TPS is output-token throughput; TPOT and TTFT are milliseconds.

| Config | Workload | In/out | C | Temp | N | Output TPS | Total TPS | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Spec accept % | CV % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parserfix-quick | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| tool-schema-gate | unknown | / |  |  | 160 |  |  |  |  |  |  |  |  |
| tool-schema-gate-100 | unknown | / |  |  | 100 |  |  |  |  |  |  |  |  |
| tool-schema-gate-parserfix-100 | unknown | / |  |  | 100 |  |  |  |  |  |  |  |  |
| v0271-control-mxfp4-dflash-k7 | decode | 256/256 | 1 | 0 | 2 | 147.57 | 295.14 | 92.37 | 94.55 | 6.75 | 10.80 | 50.51 | 41.85 |
| v0271-control-mxfp4-dflash-k7 | decode | 256/256 | 2 | 0 | 2 | 217.60 | 435.20 | 146.65 | 203.52 | 8.08 | 12.13 | 45.72 | 38.48 |
| v0271-control-mxfp4-dflash-k7 | decode | 256/256 | 4 | 0 | 2 | 302.27 | 604.53 | 159.51 | 323.18 | 10.33 | 20.76 | 38.79 | 3.32 |
| v0271-control-mxfp4-dflash-k7 | warmup | 128/64 | 1 | 0 | 1 | 144.56 | 433.69 | 74.98 | 75.10 | 5.84 | 6.25 | 58.79 | 0.00 |
| v0271-main-c8-betterbench-standard | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| v028-final-betterbench-standard | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| v028-mxfp4-dflash-k7 | smoke | 128/64 | 1 | 0 | 1 | 145.06 | 435.17 | 74.32 | 74.70 | 5.82 | 6.24 | 58.24 | 0.00 |
| v028-native-fp8-nonspec-smoke | smoke | 128/64 | 1 | 0 | 1 | 35.50 | 106.50 | 55.56 | 57.41 | 27.73 | 27.75 |  | 0.00 |
| v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 1 | 0 | 2 | 145.94 | 291.88 | 102.96 | 104.68 | 5.49 | 11.55 | 65.48 | 34.92 |
| v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 2 | 0 | 2 | 158.95 | 317.89 | 167.06 | 230.50 | 11.27 | 16.07 | 36.58 | 11.94 |
| v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 4 | 0 | 2 | 291.64 | 583.27 | 209.52 | 357.58 | 9.56 | 23.63 | 45.45 | 6.60 |
| v028-parserfix-mxfp4-dflash-k7 | warmup | 128/64 | 1 | 0 | 1 | 120.01 | 360.03 | 83.45 | 83.65 | 7.14 | 7.54 | 58.24 | 0.00 |
