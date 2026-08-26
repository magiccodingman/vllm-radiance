# BetterBench compare (offline, per-category decode t/s)

| category | A med | B med | Δ% | 95% CI | verdict |
|---|--:|--:|--:|---|---|
| chat | 95.8 | 101.9 | -0.60% | [-8.7%,+7.5%] | noise |
| code | 142.4 | 158.1 | +8.51% | [+1.0%,+16.0%] | SIG |
| file_edit | 176.3 | 173.1 | -1.00% | [-7.9%,+5.9%] | noise |
| json | 191.2 | 198.5 | +0.14% | [-6.1%,+6.4%] | noise |
| math | 193.3 | 193.7 | +1.75% | [-0.0%,+3.5%] | noise |
| prose | 94.1 | 118.1 | +8.93% | [-7.2%,+25.1%] | noise |
| reasoning | 121.4 | 118.9 | -0.35% | [-13.4%,+12.7%] | noise |
| summarization | 171.1 | 169.6 | +0.76% | [-3.9%,+5.5%] | noise |

*Cross-file compares are unpaired in time; prefer `betterbench ab` for interleaved, drift-cancelled comparisons.*

## Aggregate/concurrency delta

| Metric | v0.27.1 merged main | v0.28 final | Delta |
|---|---:|---:|---:|
| Weighted single-stream TPS | 145.5 | 152.5 | +4.81% |
| c1 aggregate TPS | 132.4 | 135.0 | +1.94% |
| c2 aggregate TPS | 234.6 | 222.3 | -5.24% |
| c4 aggregate TPS | 349.8 | 348.8 | -0.27% |
| c8 aggregate TPS | 418.8 | 474.0 | +13.19% |

The category table above is the exact output of `betterbench compare A B`.
The aggregate/concurrency table is calculated directly from the two retained
reports. The isolated c2 result crosses the nominal five-percent milestone
threshold by 0.24 percentage points, while c1/c4 are equivalent, c8 improves
substantially, and the weighted single-stream result improves. It is reported
as-is rather than hidden or averaged away.
