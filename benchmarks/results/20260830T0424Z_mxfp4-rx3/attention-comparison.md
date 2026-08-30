# Benchmark comparison

Positive percentages are improvements: higher TPS or lower latency.

| Config | Workload | In/out | C | Temp | Output TPS | TPS delta | TTFT delta | TPOT delta | Accept % | Accept delta | Accept len |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention-bf16-control -> attention-fp8qk-pv-candidate | correctness_fixed | / |  | 0 |  |  |  |  |  |  |  |
| attention-bf16-control -> attention-fp8qk-pv-candidate | prefill | 2048/64 | 1 | 0 | 37.75 | -0.20% | -2.37% | +0.59% |  |  |  |
| attention-bf16-control -> attention-fp8qk-pv-candidate | prefill | 2048/64 | 4 | 0 | 80.87 | +3.09% | +0.50% | +9.92% |  |  |  |
| attention-bf16-control -> attention-fp8qk-pv-candidate | prefill | 2048/64 | 8 | 0 | 100.91 | +0.29% | +0.27% | +0.34% |  |  |  |
| attention-bf16-control -> attention-fp8qk-pv-candidate | warmup | 128/64 | 1 | 0 | 47.41 | -2.45% | -15.84% | -1.32% |  |  |  |
