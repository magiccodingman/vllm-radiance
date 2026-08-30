# Benchmark comparison

Positive percentages are improvements: higher TPS or lower latency.

| Config | Workload | In/out | C | Temp | Output TPS | TPS delta | TTFT delta | TPOT delta | Accept % | Accept delta | Accept len |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynwidth-off-c4c8-control -> dynwidth-min5-c4c8-final | decode | 256/256 | 4 | 0 | 321.60 | -20.03% | -3.89% | -11.65% | 47.73 | +0.00 pp | 4.34 |
| dynwidth-off-c4c8-control -> dynwidth-min5-c4c8-final | decode | 256/256 | 8 | 0 | 508.55 | +8.97% | -1.54% | +11.90% | 56.45 | +14.93 pp | 3.85 |
| dynwidth-off-c4c8-control -> dynwidth-min5-c4c8-final | warmup | 128/64 | 1 | 0 | 151.50 | -6.68% | -56.77% | +2.60% | 53.44 | +0.00 pp | 4.74 |
