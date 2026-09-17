# DFlash sampling and GDN prefill corrections

These changes correct two independent numerical defects. The qualification uses
synthetic probabilities and random tensors; it requires no model or conversation.
It establishes the arithmetic corrections, not an end-to-end answer-quality or
repetition-rate improvement.

## DFlash selector proposal noise

The selector walk samples a proposal with the same `(seed, position)` noise that
target rejection sampling uses to choose a replacement. Conditioning on the
rejected proposal biases that replacement distribution. Offset the selector's
local Philox index by `1 << 30`, adapting the independent draft stream in
[vLLM #54282](https://github.com/vllm-project/vllm/pull/54282), commit
`fe755c88995ad468882517b6c4bdd60138d46a3a`.

This affects probabilistic drafting. It does not change model positions, cache
indices, greedy selection or the proposal probabilities passed to verification.
Upstream #54282 already fixes this same DFlash2 selector by passing
`IS_DRAFTING=True` to its updated `gumbel_noised_argmax` API. Radiance's pinned
vLLM v0.28.0 predates that API, so this backport adds the same salt directly to
the selector's local RNG index. It is not an additional correction to upstream
vLLM main. The selector's sampling-position buffer is already unclamped.

With target probabilities `[0.1, 0.5, 0.4]`, draft probabilities `[0.5, 0.3, 0.2]`
and 200,000 draws at each of three positions, the largest absolute probability
error is 0.01781–0.01953 under shared noise and 0.000965–0.001180 after correction.
Target-only controls pass, and greedy outputs remain exact.

## Extreme GDN decay spans

The pinned libr4d scan factorizes bounded decay products around a chunk midpoint.
It clamps growing exponentials at 80. A chunk decay span above 160 can attenuate
valid diagonal contributions and erase the carried state while leaving every
result finite. The source is
`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip` at libr4d
`e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`.

After the fast scan, a second GPU kernel detects affected sequence/head pairs
using a conservative span threshold of 128. It recomputes those pairs from the
bounded FP32 recurrence, starting from the original initial state:

```text
S = exp(g_step) * S
residual = beta * (v - S @ k)
S = S + residual @ k.T
output = scale * (S @ q)
```

Other heads retain the fast result. Both launches use the caller's stream; there
is no host readback, synchronization, temporary tensor allocation or decode-path
change. The existing scan ABI and tensor layouts are preserved. The correction
is included in the existing libr4d build patch, so all three image build paths
receive it.

On 128-token inputs, negative log decay 3.2 produces 46.4% relative output error
and 100% final-state error before correction; decay 8 produces 82.8% and 100%.
After correction, those output errors are approximately 0.17%, and their state
errors are below 0.000014%. Across 12 cases for each of the 48/16-head and
24/8-head layouts, maximum corrected output error is 0.345% and state error is
0.252% against an independent FP64 recurrence. Unequal sequences, partial chunks,
mixed affected heads, large values, and three graph replays with changed inputs
are covered. These are single-GPU tests of both layouts, not distributed TP2
qualification.

The extra launch and recurrence add prefill work. End-to-end throughput impact
has not been qualified. Previously computed context state must be rebuilt when
adopting the corrected math; persisted state made by the old kernel should not be
silently reused. No sampler penalty or repetition detector is introduced.

## Reproduction and recorded evidence

Build a corrected image using the usual image workflow, then run inside it with
this repository mounted as the working directory:

```bash
python benchmarks/bin/check_dflash_sampling_rng.py --output /tmp/sampling.json
python benchmarks/bin/check_gdn_extreme_decay.py --output /tmp/gdn-48.json
python benchmarks/bin/check_gdn_extreme_decay.py --heads 24 --output /tmp/gdn-24.json
```

The GDN checker also accepts `--library` for an independently compiled scan
translation unit. Use `--expect-failure` with an original unit to verify that the
control reproduces the defect. The sampler checker subtracts the proposal salt
in its control arm to reproduce the old coupling through the same native kernels.

The tests ran on gfx1201/R9700 using the pinned Radiance 1.0.16 image. The scan
translation units were built with its compiler and the libr4d build's
`-O3 -std=c++17 -fPIC --offload-arch=gfx1201 -mcumode` flags. This qualifies the
changed unit, not a full rebuild of every image variant. Compact numeric evidence
is in [qualification.json](../benchmarks/results/20260914-numerical-corrections/qualification.json).
