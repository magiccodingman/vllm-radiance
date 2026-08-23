# DFlash2 optimization on dual R9700

This is the durable continuation log for the DFlash2 optimization branch. It
starts from merged `main` commit
`46f484032c25fafd776afc6fe8aa16f58aa6307b` and preserves the benchmark
contract, compiler pins, Radiance paths, correctness fixtures, and immutable
run history established by merge request !5.

## Qualification policy

- Native FP8 target weights, FP8 KV, TP2, and the bounded 16K/85% deployment
  envelope remain the target contract.
- Other weight or KV formats are diagnostic controls only.
- Strict fixed-prompt greedy equivalence remains a non-negotiable gate; failed
  candidates may be characterized but are not called lossless or qualified.
- Radiance FP8 preshuffle/dispatch, custom TP2 all-reduce, FP8 payload
  reduction, unified-attention tuning, and split-K alignment remain enabled
  unless an isolated matched experiment establishes a better replacement.
- Cold JIT, prefix-cache leakage, sampling-policy changes, failed runs, and
  unlike runner modes are never hidden in reported comparisons.

## Checkpoints

Results, exact run IDs, upstream commit audits, negative experiments, and the
final deployment recommendation will be appended here as each gate completes.

