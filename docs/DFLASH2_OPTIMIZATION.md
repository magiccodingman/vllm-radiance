# DFlash2 optimization on dual R9700

This is the durable continuation report for merge request !6. Work starts from
merged `main` commit `46f484032c25fafd776afc6fe8aa16f58aa6307b` and preserves
the compiler pins and measured Radiance paths established by !5. Runs use
deterministic disjoint warmups, `temperature=0`, native FP8 target weights, FP8
KV, TP2, exact manifests, and telemetry unless a diagnostic exception is stated.

## Executive decision

DFlash2 is dramatically faster after enabling V2 `PIECEWISE` graph execution,
but remains experimental because strict greedy target equivalence fails. The
default deployment remains ordinary non-speculative Radiance at 16K/85%. The
recommended experimental profile is selective-FP8 drafter K5, V2 + `PIECEWISE`,
8K/85%, `TRITON_ATTN` for the drafter, and `RADIANCE_AR_QUANT=0`. It retains
preshuffle, unified target attention, and custom TP2 reduction. It passed smoke,
repeated requests, c1/c2/c4/c8, sustained decode, and two concurrent near-8K
requests, but must not be called lossless or production-qualified.

## Reproducibility

- Project baseline: `46f484032c25fafd776afc6fe8aa16f58aa6307b`.
- vLLM: `a014e35f38c80fb0652387740193ad2147fed6a3`,
  `0.28.0.dev0+a014e35`.
- PyTorch: AMD `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`,
  `2.12.0+rocm7.14`.
- Triton: AMD `f0b55c07da61c71775bef6d1a15ebf846430ac75`, `3.7.1`.
- AITER: `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`, `0.1.20`,
  built for gfx1201 with system Triton.
- ROCm/HIP: 7.14.0 / 7.14.60850; kernel `7.0.0-30-generic`.
- GPUs: two R9700s, gfx1201/wave32, VBIOS `113-APM107573-101`, SMC
  `00.104.76.00`, MES/MES-KIQ `0x89`, MEC 3290, PFP 3050, RLC 12484000,
  SDMA 7966358.
- Image: `vllm-radiance:dflash2-experimental-46f4840` (alias of the exact
  tested baseline), digest
  `sha256:a86976109996b9a615d14ef1fa5014a617cac8f89a3a74e2525b11d55b610a60`.

Target model checksums:

| File | SHA-256 |
|---|---|
| `config.json` | `335b8df549609165e2205e97a95f2f712115de72457ad2448dc533beb5791825` |
| `model.safetensors.index.json` | `39e0e44c0c9b54bdbc12aa6e3412562cab9e97de634066e81323e4dba27b2cd6` |
| `model-00001-of-00006.safetensors` | `634f2fe48af30553b3c59cf7572b06122346fe26153b35f8be701e877fbcdce4` |
| `model-00002-of-00006.safetensors` | `a29cd7bf0e19fea62d711c5be1f7b519d2f70e1f52f8dd6c60eb69a00c3ada82` |
| `model-00003-of-00006.safetensors` | `8ee439207aee8a981df27ac933f062995bbc5309aa492279efc3b917c5d2c8cb` |
| `model-00004-of-00006.safetensors` | `d687b63392fd4d23e57841d46af77e1cb0be642f13dfd82979a05aabe6ec3804` |
| `model-00005-of-00006.safetensors` | `292cfd883cde99cd5efcd9e6b76345dde1832e2eb1a2e452819e9425b9424ed1` |
| `model-00006-of-00006.safetensors` | `140b2d5cc5236532cbb9ef1fb73c7fd0880747e3e3536a08c47aac6d6964359c` |
| `model-auxiliary.safetensors` | `b7a735a24d2adb26c910a0ccd0181e8ffa681d5a2d6c16785bcf977e4ce38d30` |

Selective-FP8 draft checksums are
`1d090c2168d81d39a5e843f075c6580e35471a483beeccca0364ef90f2dfcaf6`
for `config.json` and
`ed5b3094c9053e5f2f1ad7b627a7a8a03c625a895764f207e24e3d7a251831e8`
for `model.safetensors`.

## Baseline confirmation

Fresh smoke `20260823T050325Z_..._smoke` passed. The clean merged-main quick
gate `20260823T051208Z_..._quick` delivered 35.32 / 65.95 / 125.41 / 222.94
output TPS at c1/c2/c4/c8, reproducing !5. Logs confirm Radiance preshuffle,
custom TP2 reduction and FP8 payload, unified attention, GDN WMMA, and the
pinned compiler stack. No stale branch image was inherited.

## Correctness investigation

The eight-prompt fixture gained repetitions, top-five logprobs, first-divergence
analysis, and a short-length hybrid-GDN graph-alias fixture. One variable was
changed per control.

- V1 eager and V2 eager non-spec match 8/8. V2 alone is not the cause.
- Disabling FP8 all-reduce payload quantization improves eager DFlash from 1/8
  to 4/8; it is disabled experimentally, but does not solve the issue.
- Diagnostic BF16 KV, disabled GDN WMMA, disabled prefix caching, and disabled
  mamba align state remain 4/8. FP8 KV and these state paths are not isolated.
- A BF16 target reaches 6/8, so FP8 weights amplify but do not fully cause it.
- DFlash K1 and matched built-in MTP are each 3/8. The issue begins at target
  speculative-verification shapes and is not DFlash-specific state corruption.
- Selective-FP8 and BF16 drafters produce the same verified output on 8/8
  prompts. Drafter quantization did not add an output difference here.
- Disabling custom TP2 reduction remains 3/8. The fast path is exonerated.
- The exact final 128-token comparison is repeatable on both sides but only 1/8
  fully exact. Six of seven first divergences have target top-two logit margin
  0-0.125; the seventh is 0.375. Positions are 0, 3, 30, 36, 54, 81, and 90.

The supported classification is deterministic target-runner numerical drift
caused by speculative verification shapes, amplified by native-FP8 weights and
FP8 payload quantization. No control indicates FP8-KV corruption,
nondeterminism, prefix corruption, or hybrid-GDN state loss. Coherent output is
not a strict pass. Evidence is `20260823T082609Z_..._quick` versus
`20260823T081642Z_..._qualification` and its
`correctness-analysis-vs-matched-piecewise-control.json`.

The open old-runner full-graph shape-alias bug (#53051/#53059) was not reproduced
in the V2 runner. A fixture spanning prompt lengths 4-10 kept the six-token K5
`1 + K` alias case exact and repeatable, so that unrelated patch was not pulled.

## Recovering the compatible base runner

| Runner | c1 | c2 | c4 | c8 | Run |
|---|---:|---:|---:|---:|---|
| V1 eager | 15.70 | 27.25 | 59.91 | 113.74 | `20260823T061726Z_..._quick` |
| V2 eager | 17.37 | 31.71 | 64.42 | 117.41 | `20260823T062705Z_..._quick` |
| V2 compile, graphs `NONE` | 22.52 | 38.53 | 77.02 | 142.32 | `20260823T063618Z_..._quick` |
| V2 compile, `PIECEWISE` | 34.65 | 65.62 | 123.87 | 222.44 | `20260823T064528Z_..._quick` |
| Normal Radiance | 35.32 | 65.95 | 125.41 | 222.94 | `20260823T051208Z_..._quick` |

Eager execution, not V2, caused the large loss. `PIECEWISE` recovers 98.1% /
99.5% / 98.8% / 99.8% of ordinary Radiance. All material Radiance paths still
fire. `FULL_AND_PIECEWISE` is rejected: selective-FP8 K5 falls to 66.33 /
118.45 / 201.37 / 357.56 TPS, 13.5-29.3% behind `PIECEWISE`.

## Speculative depth and draft quantization

BF16 3.58 GiB drafter sweep:

| Depth | c1 | c2 | c4 | c8 | Accepted length c1/c2/c4/c8 | Run |
|---|---:|---:|---:|---:|---|---|
| K3 | 75.98 | 124.55 | 244.25 | 399.18 | 3.13 / 3.06 / 3.24 / 3.15 | `20260823T070227Z_..._standard` |
| K5 | 86.24 | 142.24 | 285.98 | 429.35 | 3.59 / 3.88 / 4.05 / 4.04 | `20260823T070906Z_..._standard` |
| K7 | 98.94 | 172.01 | 252.42 | 405.17 | 4.24 / 4.85 / 3.95 / 4.49 | `20260823T071540Z_..._standard` |
| K9 | 98.76 | 183.86 | 301.79 | 412.86 | 4.33 / 5.00 / 4.71 / 5.18 | `20260823T072057Z_..._standard` |

Raw TPS prefers K7 at c1, K9 at c2/c4, and K5 at c8. K9 retains capacity for
only ~6.2 full 8K requests and has 12.5-22.1% CV, so it is not a safe global
choice. K5 is the mixed-concurrency/headroom choice; K7 is a faster explicitly
low-concurrency profile.

The selective-FP8 drafter is 2.34 GiB versus 3.58 GiB. Its quant map excludes
QKV/O, kernel projections, FC, and candidate-selector operations affected by
open raw-quantized-weight bugs. It loads correctly and is token-identical to the
BF16 drafter's verified output in matched mode.

| Candidate | c1 | c2 | c4 | c8 | Capacity |
|---|---:|---:|---:|---:|---:|
| FP8 K5 | 93.79 | 137.73 | 232.81 | 432.28 | 9.63x 8K |
| FP8 K6 | 99.00 | 148.36 | 234.92 | 351.61 | 8.70x 8K |
| FP8 K7 | 103.79 | 152.25 | 277.69 | 435.71 | 7.94x 8K |

K6 is a negative interpolation. K7 is fastest but lacks a safe c8/full-8K
capacity margin. K5 was qualified. Generic quantized-DFlash support is not
declared fixed; PRs #53122, #51620, and #51684 remain open.

## Final qualification and scoreboard

Final experimental run: `20260823T081642Z_..._qualification`.

| C | Output TPS | Request/s | TTFT ms | TPOT ms | Accept % | Accept len | Free GiB/R9700 | Peak W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 93.78 | 0.366 | 82.62 | 9.28 | 56.87 | 3.84 | 4.41 | 233 |
| 2 | 138.18 | 0.540 | 136.31 | 10.37 | 55.33 | 3.77 | 4.41 | 222 |
| 4 | 225.96 | 0.883 | 160.98 | 12.20 | 45.99 | 3.30 | 4.41 | 233 |
| 8 | 399.81 | 1.562 | 251.28 | 15.23 | 52.06 | 3.60 | 4.41 | 245 |

Peak decode junction temperature was 77 C. Sustained 512-token output reached
114.03 / 160.63 / 371.50 / 541.74 TPS. Short-decode CV was 7.54% / 14.03% /
5.85% / 11.45%, reflecting acceptance-sensitive prompt variance. The server
retained 107,193 KV tokens (13.09 full 8K requests) and completed two concurrent
7,936-token prompts. All requests passed; strict equivalence did not.

| Lane | c1 | c2 | c4 | c8 | Status |
|---|---:|---:|---:|---:|---|
| Radiance 0.5.8 non-spec | 35.30 | 66.36 | 125.51 | 224.99 | qualified historical |
| Radiance 0.5.8 + MTP | 74.32 | 120.04 | 176.86 | 274.56 | benchmark passed |
| Merged Radiance 0.6 non-spec | 35.32 | 65.95 | 125.41 | 222.94 | qualified default |
| Radiance 0.6 + MTP | 83.66 | 120.63 | 182.71 | 287.83 | passed; c2 noisy |
| Initial eager Radiance 0.6 + DFlash2 | 48.50 | 76.08 | 175.74 | 247.88 | strict failed |
| Final FP8-K5 `PIECEWISE` DFlash2 | 93.78 | 138.18 | 225.96 | 399.81 | operational pass; strict failed |
| FP8-K7 speed profile | 103.79 | 152.25 | 277.69 | 435.71 | capacity-edge experimental |

Final K5 improves over matched V2 `PIECEWISE` non-spec by 170.6% / 110.6% /
82.4% / 79.7%, over ordinary Radiance by 165.5% / 109.5% / 80.2% / 79.3%,
over current MTP by 12.1% / 14.6% / 23.7% / 38.9%, and over 0.5.8 MTP by
26.2% / 15.1% / 27.8% / 45.6%. Relative to initial eager DFlash it gains
93.3% / 81.6% / 28.6% / 61.3%. Comparison reports live in the final run root.

## Upstream and negative-result audit

At final audit, vLLM `main` was `e25c586b9030a10702d78856b43ccae9481cc28c`,
eight commits beyond the pin. None is a merged DFlash performance/correctness
fix. #52560 currently breaks DFlash draft loading; repairs #53435/#53449 are
open. ROCm verification #53383, quant-draft PRs #53122/#51620/#51684,
compile-cache isolation #53292, widest-shape capture #50488, RoPE reuse #53251,
and K=0 skip #53426 are open. Floating `main` would lose a working loader.

Retained negative results:

- Upstream AITER linear instead of preshuffle: 9-11% regression.
- No custom TP2 reduction: 7.6-8.3% base decode regression.
- `TRITON_ATTN` target: major long-context regression; it is draft-only.
- Full graphs: 13.5-29.3% DFlash regression.
- FP8 payload reduction worsens DFlash strict matching; normal Radiance keeps it.
- Repeated AITER GDN-prefill is +0.34/+0.37/+0.61% output TPS at c4/c1/c8
  and ~0.9-1.4% better median TTFT. That overlaps sample CV and its first
  gfx1201 JIT takes minutes, so it remains opt-in. Runs:
  `20260823T084909Z_..._standard` and `20260823T085259Z_..._standard`.
- Failed run `20260823T083233Z_..._quick` exposed empty
  `VLLM_USE_V2_MODEL_RUNNER` injection. The redundant Compose entry was removed;
  the conditional runner injection and the failed evidence are retained.

## Deployment recommendation

Deploy ordinary Radiance 0.6 non-spec: TP2, native FP8 target, mandatory FP8
KV, 16K, 85%, eight sequences, 2,048 batched tokens, unified AITER target
attention, and all proven Radiance optimizations. It is the production-qualified
profile.

For explicitly experimental serving, use the selective-FP8 DFlash K5 profile
in `benchmarks/README.md`: V2, `PIECEWISE`, 8K/85%, draft TP2, `TRITON_ATTN`
draft attention, and `RADIANCE_AR_QUANT=0`. It leaves about 4.41 GiB physical
headroom per R9700. Do not enable DFlash in default Compose until strict
verification-shape equivalence passes or the project deliberately adopts a
different, explicitly documented numerical-equivalence policy.
