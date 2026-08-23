# libr4d 0.4.0 port and BetterBench qualification

This report continues the pinned-main and DFlash2 work recorded in
`UPGRADE_PROGRESS.md` and `DFLASH2_OPTIMIZATION.md`. It does not replace those
immutable checkpoints. Its scope is the 2026-08-23 port of DeadCode Radiance
0.7.4/libr4d 0.4.0 and a matched BetterBench comparison of non-spec, MTP, and
DFlash2 on two Radeon AI PRO R9700 GPUs.

## Reproducibility

### Fork candidate

- Git branch: `agent/libr4d-better-bench`
- Runtime checkpoint commit: `765f3a0`
- Candidate image: `vllm-radiance:libr4d-port-candidate`
- Image digest: `sha256:04f23ad361bee9db943aa6661b9ff770157e52c1fbdd269157ae9b9e759fe6cc`
- vLLM: `a014e35f38c80fb0652387740193ad2147fed6a3`
  (`0.28.0.dev0+a014e35`)
- AMD PyTorch: 2.12 commit `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`
- AMD Triton: 3.7.1 commit `f0b55c07da61c71775bef6d1a15ebf846430ac75`
- AITER: 0.1.20 commit `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`
- ROCm: 7.14
- libr4d: v0.4.0 commit `000d5f91d0e47ee9faf3b5466f0a12995f0cbfd6`
- Target: native-FP8 `Qwen3.8-27B-heretic-ara-fp8-magiccodingman`
- DFlash drafter: selective-FP8
  `Qwen3.8-27B-heretic-ara-DFlash2-fp8-magiccodingman`
- Mandatory FP8 KV, TP2, 85% GPU allocation, max 8K, max 8 sequences,
  4,096 batched tokens, no CPU/KV offload.

The image imported vLLM, torch, Triton, AITER, and r4d successfully; `pip check`
passed. Startup resolved all 14 expected libr4d kernels.

### External upstream baseline

- Source: DeadCode `vllm-radiance` commit
  `759b9eea529` (release 0.7.4)
- Image: `stilldeadcode/vllm-radiance:0.7.4`
- Immutable digest:
  `sha256:2ab4dfc999203441d0b330876df2b41422132576159a25aa5d85d398e83fbc49`
- Stack: vLLM 0.27.1, PyTorch 2.11, Triton 3.6, AITER 0.1.17-era,
  ROCm 7.14, libr4d 0.4.0.

This replaces 0.5.8 as the external Radiance baseline going forward. The fork
does not copy the upstream dependency stack: it ports the R4D work onto the
already-qualified pinned-main compiler/runtime foundation so DFlash2 remains
available.

## What was ported

- R4D attention for target prefill and decode, adapted to pinned main's LBHNC
  KV-cache layout API.
- R4D GDN prefill, decode, and speculative-state paths.
- R4D vision flash attention.
- Exact R4D TP2 P2P all-reduce and rotated six-bit compressed all-reduce.
- R4D router integration.
- `RADIANCE_FAST_DRAFT`: INT2-g128 MTP-head copy with exact top-32 reranking.
- Current entrypoint reporting and Qwen chat-template updates.

The old broad gfx1201 AITER gate was not restored because pinned vLLM main has
separate safe RDNA4 routing. Radiance's preshuffled FP8 dispatcher, split-K
alignment, fused quantization, and the known-good dependency pins were retained.
A per-operator `RADIANCE_USE_R4D_GDN` switch was added so correctness and
performance can be isolated without disabling the complete R4D family.

## BetterBench contract

- BetterBench v0.2.2, exact commit
  `575cc3925bac922d6ad4a39e62502673799979d9`.
- Corpus v1.0; code/reasoning/prose/JSON/file-edit/summarization/math/chat.
- 10 measured passes per category after warmup.
- Greedy sampling, temperature 0, seed 20260823.
- Unique nonce on every request; prefix caching explicitly disabled across
  images as a second protection against reuse.
- 24 requests at each concurrency level c1/c2/c4/c8.
- Cold prefill at target depths 2K/4K/7K.
- One-second GPU/host telemetry and exact manifests/checksums.

The BetterBench self-test passed before GPU use, including its timing calibration
and null A/B gate. A five-pass shortcut was not used for published numbers.

## BetterBench scoreboard

Output rates are tokens/second. “Weighted” is BetterBench's category-weighted
single-stream decode median. c1-c8 are aggregate throughput from 24 requests per
level.

| Configuration | Weighted | c1 | c2 | c4 | c8 | Min free VRAM/card |
|---|---:|---:|---:|---:|---:|---:|
| DeadCode 0.7.4 R4D non-spec | 35.7 | 35.6 | 66.9 | 117.4 | 190.7 | 5.54 GiB |
| Fork R4D non-spec | 35.8 | 35.5 | 67.0 | 118.4 | 187.6 | 6.00 GiB |
| Fork R4D MTP K8, INT2 exact rerank | 93.7 | 82.8 | 149.0 | 242.7 | 316.3 | 8.88 GiB |
| Fork R4D DFlash2 K5 | 99.4 | 93.3 | 169.6 | 293.6 | 451.6 | 6.52 GiB |
| Fork R4D DFlash2 K7 | **112.6** | **102.1** | **189.0** | **305.1** | **496.1** | 6.47 GiB |

### Relative results

- Fork non-spec versus DeadCode 0.7.4: +0.3% weighted, -0.3% c1,
  +0.1% c2, +0.9% c4, and -1.6% c8. This is decode parity, not a broad win.
- Fork MTP versus fork non-spec: 2.62x weighted; 2.33x/2.22x/2.05x/1.69x
  at c1/c2/c4/c8.
- DFlash K5 versus fork non-spec: 2.78x weighted and 2.41x at c8.
- DFlash K7 versus fork non-spec: 3.15x weighted and 2.64x at c8.
- DFlash K7 versus MTP: +20.2% weighted and +23.3%/+26.8%/+25.7%/+56.8%
  at c1/c2/c4/c8.
- DFlash K7 versus K5: +13.3% weighted and +9.4%/+11.4%/+3.9%/+9.9%
  at c1/c2/c4/c8.

The cumulative server counters explain why K7 wins despite a lower raw draft
acceptance percentage:

| Mode | Accepted draft tokens | Draft tokens | Acceptance | Accepted tokens/round |
|---|---:|---:|---:|---:|
| MTP INT2 | 56,500 | 106,900 | 52.85% | 2.445 |
| DFlash K5 | 57,067 | 111,425 | 51.22% | 2.561 |
| DFlash K7 | 59,143 | 138,201 | 42.79% | 2.996 |

K7 accepts a smaller fraction of a larger proposal but advances farther per
target verification.

### Cold prefill

| Configuration | 2K | 4K | 7K prompt tok/s |
|---|---:|---:|---:|
| DeadCode 0.7.4 non-spec | 4,200 | 4,381 | 4,344 |
| Fork non-spec | 4,061 | 4,263 | 4,227 |
| Fork MTP | 4,015 | 4,261 | 4,241 |
| Fork DFlash K5 | 4,278 | 4,467 | 4,386 |
| Fork DFlash K7 | 4,286 | 4,462 | 4,356 |

The fork's ordinary non-spec prefill is 2.7-3.3% below stock DeadCode 0.7.4.
That is a real negative result and remains follow-up work. DFlash's compatible
runner/R4D combination recovers the gap and slightly exceeds upstream during
these tiny-decode prefill probes, but that does not erase the non-spec result.

## Correctness characterization

The first live port failed safely because pinned vLLM main selected LBNHC while
the R4D kernels require LBHNC. The adapter now declares only LBHNC through the
current `KVCacheLayout` API; the corrected smoke passed.

Fixed-prompt isolation then found multiple deterministic numerical contributors:

- R4D attention changes some greedy streams versus the previous AITER path.
- R4D GDN changes additional streams versus the previous FLA/Triton path.
- Rotated six-bit all-reduce adds a separate lossy numerical difference versus
  exact R4D all-reduce.
- Every repeated candidate was internally deterministic; no recurrent-state
  corruption or request-to-request randomness was observed.

For the new R4D baseline, DFlash K5 and K7 each strictly matched matched
non-spec on 3/8 meaningful fixed prompts. MTP and both DFlash depths produced
the same fixed output payload byte-for-byte, and K5/K7 repeated identically.
This strongly characterizes a shared speculative/runner execution-order or
numerical-path difference rather than a DFlash-only random failure. It does not
turn the strict gate into a pass. DFlash and the speculative runner therefore
remain experimental and are not enabled in the portable Compose default.

## Negative and rejected experiments

- `20260823T161238Z_libr4d-port_smoke`: failed at first request due the R4D
  KV-layout mismatch. Retained as immutable failure evidence.
- `20260823T162001Z_libr4d-quick`: invalidated and marked failed because the
  executing harness was edited during the run, causing cases to restart. No
  result from it is used.
- WHT6 versus exact all-reduce changed 6/8 fixed prompt streams. WHT6 remains
  the upstream performance default but is documented as numerically lossy.
- Disabling R4D GDN or swapping only attention did not restore the previous
  output stream, proving divergence is not attributable to one kernel alone.
- vLLM PR #53383's long-context ROCm DFlash verification partitioner was not
  backported. It is an unmerged 692-line MI210-tested change aimed primarily at
  41K-100K verification; this qualification is capped at 8K and uses R4D target
  attention. Architecture applicability is not established for gfx1201.
- The pinned vLLM commit is only ten commits behind audited `main` at the time
  of this work. No later merged DFlash/hybrid-GDN correctness fix applied to
  this target. The one nearby DFlash-related commit was Qwen3 Omni/DSpark work
  and was not pulled into the dense Qwen3.8 lane.

## Immutable runs

- Port smoke failure: `20260823T161238Z_libr4d-port_smoke`
- Corrected smoke: `20260823T161630Z_libr4d-lbhnc_smoke`
- Valid R4D quick: `20260823T162825Z_libr4d-quick-valid`
- Correctness isolation: `20260823T163657Z_r4d-correctness-isolation`
- MTP BF16-head 2,048 control: `20260823T164815Z_libr4d-mtp`
- MTP BF16-head 4,096 control: `20260823T165800Z_libr4d-mtp-4096`
- MTP INT2-head quick: `20260823T170400Z_libr4d-mtp-int2`
- DFlash K5 quick: `20260823T170900Z_libr4d-dflash-k5`
- DFlash K7 quick: `20260823T171500Z_libr4d-dflash-k7`
- Fork non-spec BetterBench:
  `20260823T172400Z_betterbench-standard-r4d-nonspec`
- Fork MTP INT2 BetterBench:
  `20260823T175500Z_betterbench-standard-r4d-mtp-int2`
- Fork DFlash K5 BetterBench:
  `20260823T181100Z_betterbench-standard-r4d-dflash-k5`
- Fork DFlash K7 BetterBench:
  `20260823T182500Z_betterbench-standard-r4d-dflash-k7`
- DeadCode 0.7.4 smoke: `20260823T172100Z_deadcode-074-smoke`
- DeadCode 0.7.4 BetterBench:
  `20260823T183800Z_betterbench-standard-deadcode074-nonspec`

Each completed BetterBench directory contains its profile, exact tool commit,
raw JSON, standalone HTML, Markdown report, server log, manifest, telemetry,
and SHA256 manifest.

## Deployment decision

The safe generic default is R4D non-spec with native FP8 target weights,
mandatory FP8 KV, TP2, 16K, 85% allocation, eight admitted sequences, and a
4,096-token scheduler budget. R4D attention and R4D GDN are on by default.
Prefix caching/aligned GDN state remains enabled in the public Compose for
real shared-prefix serving; it was forced off only in the cross-image benchmark.

For an explicitly experimental high-throughput deployment, DFlash K7 is the
measured winner. Use the selective-FP8 drafter, draft TP2, `TRITON_ATTN` for the
drafter, max 8K, and piecewise graphs. It preserves healthy headroom and passed
smoke, repeated requests, c1-c8, and 7K prefill, but it must be labeled
experimental until strict cross-mode greedy equivalence is resolved or a
documented numerical qualification policy is accepted.
