# Benchmark comparison

Positive percentages are improvements: higher TPS or lower latency.

| Config | Workload | In/out | C | Temp | Output TPS | TPS delta | TTFT delta | TPOT delta | Accept % | Accept delta | Accept len |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0271-control-mxfp4-dflash-k7 -> v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 1 | 0 | 145.94 | -1.10% | -11.46% | +18.60% | 65.48 | +14.96 pp | 5.58 |
| v0271-control-mxfp4-dflash-k7 -> v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 2 | 0 | 158.95 | -26.95% | -13.91% | -39.48% | 36.58 | -9.15 pp | 3.56 |
| v0271-control-mxfp4-dflash-k7 -> v028-parserfix-mxfp4-dflash-k7 | decode | 256/256 | 4 | 0 | 291.64 | -3.52% | -31.35% | +7.40% | 45.45 | +6.65 pp | 4.18 |
| v0271-control-mxfp4-dflash-k7 -> v028-parserfix-mxfp4-dflash-k7 | warmup | 128/64 | 1 | 0 | 120.01 | -16.99% | -11.29% | -22.33% | 58.24 | -0.55 pp | 5.08 |
