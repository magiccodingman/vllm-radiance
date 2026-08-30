# RX3 MXFP4, GDN, and DFlash continuation

## Decision

This continuation ports the current production-relevant work from Brian's
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4)
onto this fork's stable-vLLM-v0.28, libr4d-0.5.0, XGrammar-correctness baseline.
The audited upstream point is main commit
`2d72e788c504ecc8f42754842589e142f23b26ed` (opti head
`e862d87e03d6baa5b7d0a1be08f3d803d6ee3c87`). Original authorship is retained
in Git rather than presenting the work as native to this fork.

The recommended MXFP4+DFlash profile keeps checkpoint-order weights, K7,
BF16 R4D prefill-attention legs, and the M<=64 decode boundary. It enables the
branchless/epilogue kernel work, tuned split-K/BK policy, capacity-aware KV
grouping, GDN input-projection merge, fused speculative GDN update, 96-block
compressed all-reduce, 64-wide exact reranking, sampling-aware verification
head, and per-request dynamic verify width at c5 and above.

DFlash remains experimental. It is repeatable and structurally healthy, but it
still fails the existing strict cross-mode non-spec equivalence requirement.

## Reproducibility

- Merged `main` base: `89a160aecc7cbdb85efeec40fe8628157e26d1e6`
- vLLM: 0.28.0, `2cf0a6915ce544dc493a0990f2ea38d81601128a`,
  plus the focused fixes documented in `V028_UPGRADE.md`
- AMD PyTorch: 2.12, `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`
- AMD Triton: 3.7.1, `f0b55c07da61c71775bef6d1a15ebf846430ac75`
- AITER: 0.1.20, `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`
- libr4d: 0.5.0, `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`,
  with `r4d_radiance_extras.patch`
- ROCm userspace 7.14; host kernel 7.0.0-30-generic
- GPUs: 2x Radeon AI PRO R9700, gfx1201, VBIOS `113-APM107573-101`,
  SMC `00.104.76.00`
- Target: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, revision
  `156be69f9cac862a41d8b32e773ea2d2754341e8`, weight content OID
  `be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272`
- Drafter: `tcclaviger/Qwen3.8-27B-DFlash2-FP8`, revision
  `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`, weight content OID
  `7dbb99a8d0120f502e66b256aa7c0866d933ceeee4a02463d9db591811e8404e`

The tested candidate image is
`vllm-radiance:mxfp4-rx3-candidate@sha256:5d536be268e8e54932cf0d63519da7f990a2117b09632c05e995b0f09f386b3e`.
The final clean-source handoff is
`vllm-radiance:mxfp4-rx3-final@sha256:1dda8641c3b1e9f99a04a7138549cc8f7554ed0dd35223dee4a2ad6b68c028d5`
(image config `sha256:ce8c5e9f13ba07a13cece0eb658e38288b861ddfb57a49b298baf7ae26138fe4`).
It passed `pip check`, imports, version assertions, the GPU-free speculative
XGrammar regression, and a live DFlash smoke.

## What changed

### MXFP4/W4A8

- All nine weight/activation staging sites clamp instead of using a per-load
  bounds predicate. This removes serialized wait regions on gfx1201.
- The full-tile output epilogue uses a branch-free fast path. The fallback is
  retained behind `RADIANCE_MXFP4_EPIFAST=0`.
- Decode split-K and BK are selected by actual `(M,N,K)` geometry rather than
  one global split. The M<=64 production band stays the default.
- M<=128 coverage is available for a deliberately qualified 16-sequence
  profile. It is not enabled generically because it doubles split-K scratch.
- Fragment-order weights are available with `RADIANCE_MXFP4_WPERM=1`. They win
  decode but still cost prefill, so the balanced default remains zero.

The upstream bit-identity gates cover all six production shapes, partial tails,
both weight layouts, and every active split/BK/tile combination. Local model
startup routed all 304/304 eligible linears through the Radiance kernel.

### DFlash and target verification

- `RADIANCE_DRAFT_RERANK=64` widens the exact rerank set used by the INT2
  fast-draft head.
- `RADIANCE_VERIFY_HEAD=1` reuses the packed draft head for target verification
  only when sampling support is provably inside the exact rerank set. Grammar,
  logprobs, min-p, and unsafe top-k requests fall back to the untouched head.
- `RADIANCE_DYNAMIC_WIDTH=1` tracks accepted length per request and caps future
  target verification width. The imported c3 floor measured -2.5% at c4 and
  +9.6% at c8 in the local matched gate, so this port defaults to c5: c1-c4
  retain full K7 while larger active batches can narrow.

Raw acceptance percentage is not a valid before/after score for dynamic width:
narrowing the denominator can raise the percentage without changing the
drafter. End-to-end TPS and accepted tokens per target update remain the
decision metrics.

### GDN, KV capacity, and TP2 reduction

- Each of 48 GDN layers merges `in_proj_qkvz` and `in_proj_ba` after weight
  processing, replacing two MXFP4 GEMM launches with one without changing
  stored values.
- libr4d adds `gdn_fused_update_w4k128v128_bf16`; the speculative R4D GDN path
  selects it, while unsupported/non-spec shapes retain the old path.
- The hybrid KV allocator now scores canonical group sizes by usable request
  capacity. For 48 GDN + 16 target attention + 5 draft attention layers it
  selects group size 8 instead of upstream's 5, avoiding repeated rounding of
  the expensive full-attention bucket. The serve logs make both choices visible.
- The rotated-six-bit TP2 collective raises its measured block cap from 48 to
  96 while retaining `RADIANCE_AR_QNB=48` as the control.

### Deliberately not promoted

- `R4D_ATTN_FP8=3` provides FP8 QK/PV prefill legs. A matched local 2K-prompt
  standard gate measured -0.2%/+3.1%/+0.3% output TPS at c1/c4/c8 and
  -2.4%/+0.5%/+0.3% TTFT: no operational win beyond noise. Both arms were
  internally repeatable, but only 1/8 fixed greedy outputs matched across the
  precision change. It therefore remains off.
- AutoRound kernels were audited but not ported into the active image. They are
  a separate checkpoint format, had no local model for qualification, and the
  upstream record still trails native MXFP4 at important prefill shapes.
- AR overlap, altered graph-capture sizes, and K8 experiments were negative or
  incompatible upstream experiments and were not copied.

## BetterBench result

Contract: BetterBench 0.2.2 commit
`575cc3925bac922d6ad4a39e62502673799979d9`, v1 corpus, ten passes per
category, greedy decoding, cold nonce-prefixed prompts, TP2, FP8 KV, 8K test
envelope, DFlash K7, draft TP2, and PIECEWISE graphs. Prefix caching is disabled
only for this cold benchmark contract.

Two imported-c3 candidate starts and the final locally tuned c5 start produced:

| Run | weighted | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| Merged v0.28 baseline | 152.5 | 135.0 | 222.3 | 348.8 | 474.0 |
| Imported c3 pass 1 | 171.1 | 152.4 | 269.5 | 406.6 | 571.0 |
| Imported c3 pass 2 | 170.8 | 152.3 | 268.3 | 417.3 | 503.2 |
| Final c5 default | **171.0** | **152.5** | **269.5** | **432.6** | **570.4** |
| Final gain | **+12.1%** | **+13.0%** | **+21.2%** | **+24.0%** | **+20.3%** |

C8 is acceptance/content-sensitive and had the widest imported-c3 run spread;
both the final c5 standard pass and the focused gate are reported rather than
selecting only the faster preliminary pass. C1, c2, weighted decode, and the
category rows reproduced tightly.

A separate three-repetition 256/256-token gate isolated dynamic width. C8 was
466.7 TPS with width disabled, 511.5 with the imported c3 floor (+9.6%), and
508.6 with the final c5 floor (+9.0%); CV fell from 5.5% to 3.4–3.9%. C4 was
too noisy to tune from aggregate TPS (22–24% CV), and the c5 arm is
functionally identical to disabled at c4. This is why the default begins at
five active requests: it preserves full K7 through c4 and takes the repeatable
c8 win without fitting the noisy c4 point.

Final single-stream categories:

| Category | v0.28 | RX3 | Simple median ratio |
|---|---:|---:|---:|
| chat | 101.9 | 131.1 | +28.7% |
| code | 158.1 | 176.4 | +11.6% |
| file edit | 173.1 | 204.5 | +18.1% |
| JSON | 198.5 | 233.2 | +17.5% |
| math | 193.7 | 227.4 | +17.4% |
| prose | 118.1 | 110.1 | -6.8% |
| reasoning | 118.9 | 125.8 | +5.8% |
| summarization | 169.6 | 196.2 | +15.7% |

BetterBench's unpaired bootstrap marks chat, code, file edit, JSON, math, and
summarization significant. Prose and reasoning remain noise-limited because of
their broad within-category distributions; the prose point is disclosed rather
than hidden in the weighted aggregate.

Prefill rose from 3622.7/4109.6/3937.6 to
3979.8/4542.3/4317.5 tokens/s at the 2K/4K/7K depths: +9.9%, +10.5%, and
9.7%. The final run retained 5.48 GiB minimum physical headroom per GPU and
observed peak package power of 221/311 W.

Immutable run IDs:

- baseline: `20260826T1914Z_v028_upgrade/v028-final-betterbench-standard`
- candidate pass 1:
  `20260830T0424Z_mxfp4-rx3/candidate-dflash-betterbench-standard`
- candidate pass 2:
  `20260830T0424Z_mxfp4-rx3/candidate-dflash-betterbench-standard-warm`
- final exact-default pass:
  `20260830T0424Z_mxfp4-rx3/final-dflash-betterbench-standard-min5`
- attention control/candidate:
  `20260830T0424Z_mxfp4-rx3/attention-bf16-control` and
  `20260830T0424Z_mxfp4-rx3/attention-fp8qk-pv-candidate`
- dynamic-width control/imported/final:
  `20260830T0424Z_mxfp4-rx3/dynwidth-off-c4c8-control`,
  `dynwidth-on-c4c8-candidate`, and `dynwidth-min5-c4c8-final`

## Correctness and stability

- All three standard runs passed the sampled required multi-tool fixture 30/30,
  for 90/90 total.
- All three server logs contain zero `Failed to advance FSM`, terminated-matcher,
  grammar-rejection, traceback, OOM, or engine-fatal events.
- The eight meaningful fixed greedy prompts were byte-identical across the two
  independent candidate starts.
- Candidate versus the prior v0.28 DFlash run matched only 1/8 prompts; first
  divergences occur at deterministic low-margin choices. This is numerical
  path drift, not evidence that strict DFlash/non-spec equivalence has passed.
- The startup log confirms 304/304 MXFP4 linears, 48/48 GDN projection merges,
  KV group size 8, the fused GDN kernel, INT2 buffer reuse, and R4D attention.
- The only lines labeled `ERROR` are Transformers' existing Qwen3-VL
  `min_frames`/`max_frames` docstring validator messages; they are cosmetic and
  predate this continuation.

## Deployment recommendation

Use the existing MXFP4 target plus matched tcclaviger DFlash drafter, K7,
`RADIANCE_FAST_DRAFT=1`, `RADIANCE_DRAFT_RERANK=64`,
`RADIANCE_VERIFY_HEAD=1`, `RADIANCE_DYNAMIC_WIDTH=1`, and
`RADIANCE_DYNW_MIN_BATCH=5`. Keep `RADIANCE_MXFP4_WPERM=0`,
`RADIANCE_MXFP4_DECODE_MAX_M=64`, and `R4D_ATTN_FP8=0` for the balanced
profile. Production prefix caching and `MAMBA_CACHE_MODE=align` remain enabled;
the benchmark-only cold controls do not replace them.

The normal non-spec and native-FP8 paths remain available and retain their
established qualification. The new switches are either inert outside their
target path or have explicit one-variable controls for regression work.
