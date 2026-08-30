# Benchmark comparison

Positive percentages are improvements: higher TPS or lower latency.

| Config | Workload | In/out | C | Temp | Output TPS | TPS delta | TTFT delta | TPOT delta | Accept % | Accept delta | Accept len |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynwidth-off-c4c8-control -> dynwidth-on-c4c8-candidate | decode | 256/256 | 4 | 0 | 392.12 | -2.50% | -3.68% | +6.18% | 64.72 | +16.99 pp | 4.33 |
| dynwidth-off-c4c8-control -> dynwidth-on-c4c8-candidate | decode | 256/256 | 8 | 0 | 511.52 | +9.60% | -1.98% | +11.94% | 59.92 | +18.40 pp | 3.82 |
| dynwidth-off-c4c8-control -> dynwidth-on-c4c8-candidate | warmup | 128/64 | 1 | 0 | 146.91 | -9.51% | -84.70% | +4.08% | 53.44 | +0.00 pp | 4.74 |
