# Stable vLLM v0.28 upgrade and qualification

## Decision

Radiance now pins stable vLLM v0.28.0 at commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The release contains native
Qwen3.8 DFlash2, so the former v0.27.1 DFlash source-copy backport is retired.
The qualified AMD compiler and kernel stack remains ROCm 7.14, AMD PyTorch
2.12, AMD Triton 3.7.1, AITER 0.1.20, Transformers 5.15.0, XGrammar 0.2.3,
and libr4d 0.5.0.

The v0.28 release itself has a material required-tool regression for the
qualified `qwen3` reasoning plus `qwen3_coder` tool-parser combination. This
change carries the smallest upstream fix and retains the two speculative
XGrammar fixes from the preceding release. It also carries two directly
relevant post-release fixes for DFlash logits-cache addressing and ROCm graph
capture. No unrelated post-release vLLM main changes are floated into the
image.

## Reproducibility

- Branch base / merged `main`: `09dd025aa3b8355ed37a2dc13547f50812b5ea94`
- vLLM tag commit: `2cf0a6915ce544dc493a0990f2ea38d81601128a`
- AMD PyTorch: 2.12 branch commit
  `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`
- AMD Triton: `f0b55c07da61c71775bef6d1a15ebf846430ac75`
- AITER: `fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`
- libr4d: `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`
- ROCm userspace: 7.14.0; HIP runtime 7.14.60850
- Host kernel: 7.0.0-30-generic
- GPUs: two Radeon AI PRO R9700, gfx1201, VBIOS
  `113-APM107573-101`, SMC firmware `00.104.76.00`

The final matched benchmark target was AMD's Quark MXFP4 checkpoint at Hugging
Face revision `156be69f9cac862a41d8b32e773ea2d2754341e8`; its 19,798,196,184-byte
weight has content OID
`be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272`.
The tcclaviger FP8 DFlash2 drafter was revision
`ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`; its 2,118,882,784-byte weight
has content OID
`7dbb99a8d0120f502e66b256aa7c0866d933ceeee4a02463d9db591811e8404e`.
Both are now recorded independently in each speculative-run manifest.

Final clean-source handoff image:

- Tag: `vllm-radiance:v028-release-candidate`
- OCI manifest-list digest:
  `sha256:d948325639106aeee3d6b56860d1a4b3c89ae921e27877aa64caba53748df2cf`
- Image config digest:
  `sha256:f0952f16595faed6f21d57176ccf52d4129c9489a3db3bc80dca031bf1c335f7`

The immutable BetterBench run used the equivalent final patch overlay
`vllm-radiance:v028-final-candidate` (manifest-list
`sha256:2ccbde7470c5c500628e16019a9a0fa2609f5f08ba072014a2dbd37ae60c0b9c`).
The clean-source image was rebuilt afterward; the parser, rejection sampler,
ROCm graph wrapper, `r4d.so`, and MXFP4 extension are byte-identical between
the two images.

## Patch audit and upstream follow-ups

Every active Radiance patch was applied to a clean copy of the exact v0.28
source. Twenty-four groups applied without changes. The Quark MXFP4 registration
patch required one guarded anchor update because v0.28 removed a now-unused
`linear_backend` local; behavior was unchanged. All modified Python sources
passed `ast.parse`.

The old `patch_dflash2_v0271_backport.py` remains only as historical source
archaeology and is no longer in any image build loop. The v0.28 Qwen3.8
DFlash2 source files match the reviewed upstream implementation already used
by this fork.

Four focused follow-ups are intentionally applied:

1. vLLM #52830, merge `46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3`,
   preserves registered reasoning/tool adapters when they share one parser
   engine. This restores the structural tag required to enforce a strict tool
   union.
2. vLLM #52805 and #53046 stop speculative token batches at grammar
   termination and validate drafts generated before the post-reasoning grammar
   mask becomes active.
3. vLLM #53017, commit
   `d4f4d3f40fc5350a71777fcb0e5eb8a57bda631f`, addresses DFlash's logits cache
   with the tensor's actual column stride and rejects a cache narrower than the
   sampled vocabulary.
4. vLLM #53818, commit
   `080a66a69c6fd1fe464756f88ab958baad66ce69`, captures general, encoder, and
   Gemma4 graphs on vLLM's current ROCm stream.

The DFlash decoder-class loader fix and empty-schedule GDN-state fix are already
present in the release source and were not duplicated. Wider-draft FlashInfer
all-reduce workspace sizing, speculators-format option merging, and the new
batch-sharded sampler were reviewed but not backported: the qualified path does
not select those code paths, target and draft hidden sizes match, or the change
is a new feature rather than a release regression fix.

## Required-tool regression and fix

The first v0.28 candidate passed only 29/30 sampled multi-tool requests. A
larger run passed **94/100**; all six failures selected `click` correctly but
omitted required `ref`. Server logs had no XGrammar FSM errors. The target and
v0.27.1 controls rendered byte-identical prompts, which excluded the
Transformers 5.15 chat-template update.

Root cause is the v0.28 `ParserManager` shared-engine shortcut. It returns the
raw shared parser engine for `qwen3` plus `qwen3_coder`, discarding the
registered Qwen tool adapter and its `qwen_3_coder` structural tag. Therefore
`tool_choice=required` selects a tool but does not constrain its required JSON
properties.

After the #52830 backport:

- the GPU-free regression check confirms a `DelegatingParser`, the
  `Qwen3EngineToolParser`, and a non-empty structural tag;
- the same live sampled fixture passed **100/100**;
- the server log contained zero FSM failures, grammar-overrun warnings,
  tracebacks, or engine errors.

The strict sampled gate is retained. It was not weakened to make v0.28 pass.

## Matched BetterBench qualification

Both images use BetterBench v0.2.2 commit
`575cc3925bac922d6ad4a39e62502673799979d9`, corpus v1, ten passes per
category, greedy temperature-zero decoding, unique cold nonces, FP8 KV, TP2,
8K/C8, 85% allocation, 4,096 batched tokens, MXFP4/W4A8 target, fast DFlash
K7, disabled dynamic drafting, and PIECEWISE graphs. Prefix caching is off only
for this cold benchmark contract.

| Metric | Merged-main v0.27.1 | Final v0.28 | Delta |
|---|---:|---:|---:|
| Weighted single-stream TPS | 145.5 | 152.5 | +4.81% |
| c1 aggregate TPS | 132.4 | 135.0 | +1.94% |
| c2 aggregate TPS | 234.6 | 222.3 | -5.24% |
| c4 aggregate TPS | 349.8 | 348.8 | -0.27% |
| c8 aggregate TPS | 418.8 | 474.0 | +13.19% |
| 2K prefill TPS | 3,629.6 | 3,622.7 | -0.19% |
| 4K prefill TPS | 4,129.1 | 4,109.6 | -0.47% |
| 7K prefill TPS | 3,950.5 | 3,937.6 | -0.33% |

The v0.27.1 control completed 24/24 requests at every concurrency, passed the
eight-prompt meaningful fixture, and passed 30/30 sampled required-tool calls.
Its immutable run is
`20260826T1914Z_v028_upgrade/v0271-main-c8-betterbench-standard`.

The final v0.28 run completed 24/24 requests at every concurrency, completed
all eight meaningful correctness prompts, and passed 100/100 sampled
multi-tool required-schema calls. Its K7 metrics covered 152,786 drafted and
68,868 accepted tokens: 45.07% token acceptance and 4.111 mean accepted length.
The run retained 5.12 GiB minimum physical VRAM headroom per GPU; peak sampled
power was 319 W and 328 W. No XGrammar FSM failures, OOMs, tracebacks, or
engine errors occurred.

The weighted result and especially c8 improve, while c2 is a real-looking
borderline regression and is reported as such. The prefill deltas are within
half a percent. No claim is made that every workload or concurrency is faster.
Per-category output TPS was 101.9 chat, 158.1 code, 173.1 file edit, 198.5
JSON, 193.7 math, 118.1 prose, 118.9 reasoning, and 169.6 summarization. The
immutable run is
`20260826T1914Z_v028_upgrade/v028-final-betterbench-standard`; the exact
direction-normalized table is `compare-v0271-main-v028-final.md` beside it.

## Negative and diagnostic results

- The first attempted control resolved to the older local 1.0.9 image rather
  than merged main. It was stopped before publication and remains marked
  `failed`; the digest-correct 1.0.10 control was then run from scratch.
- The first v0.28 shared-parser candidate failed the tool gate (29/30, then
  94/100). Those raw failures remain stored; they are not excluded from the
  experiment history.
- A two-repetition synthetic quick check showed a nominal -27% c2 result, but
  individual repetitions had very high acceptance-driven variance. It was not
  published as a regression; the matched 10-pass BetterBench run is the
  decision gate.
- The first `Dockerfile.dev` attempt reused the pruned release image and could
  not link vLLM because `librocshmem.a` is intentionally absent at runtime.
  The developer builder now restores only that 113 MiB link-time archive from
  the exact full ROCm 7.14 base in a throwaway stage.
- Transformers 5.15.0 prints two upstream doc-validation messages for
  undocumented Qwen3-VL `min_frames`/`max_frames`. They are noisy but do not
  disable multimodal loading or fail a request; the version remains the exact
  v0.28 ROCm lock rather than silently substituting another release.

## Validation checklist

- All guarded patch groups apply to the pinned v0.28 wheel.
- Python parse and import validation pass.
- `pip check` reports no broken requirements.
- The GPU runtime reports vLLM 0.28.0, PyTorch 2.12 ROCm 7.14, Triton 3.7.1,
  AITER 0.1.20, Transformers 5.15.0, XGrammar 0.2.3, and libr4d 0.5.0.
- PIECEWISE graph capture completes on both R9700s.
- Radiance MXFP4 selects its kernel for 304/304 target linears.
- R4D attention/GDN and the Radiance TP2 custom all-reduce remain active.
- The independent native-FP8, non-spec TP2 smoke passed with FP8 KV, confirming
  that qualification is not limited to MXFP4 plus DFlash.
- Final BetterBench, correctness, and the 100-request tool-schema gate pass.
- The clean-source image passed the actual 128K/C4 production-shaped MXFP4 +
  DFlash K7 profile with FP8 KV, prefix caching, aligned GDN state, PIECEWISE
  graphs, and request-ID headers. Text and base64-image chat requests completed;
  a further clean-image multi-tool gate passed 30/30.
- Both release and fast developer Dockerfiles build from their exact pins,
  `pip check` passes in both, and the public plus private Compose merge validates.
- After qualification, the local service was restored to the published
  merged-main image rather than leaving an unpublished candidate in production.

## Deployment

The upgrade does not change the public model-neutral Compose envelope or make
DFlash a default. Continue using the measured MXFP4 128K/C4 production profile
with FP8 KV, prefix caching, `MAMBA_CACHE_MODE=align`, DFlash K7, fast draft,
and PIECEWISE graphs when choosing the experimental high-throughput lane.
Checkpoint-native chat templates remain the portable default.

The v0.28 candidate is suitable for review and further deployment testing. It
is not made the repository or local production default in this branch: DFlash
remains opt-in because its established strict speculative/non-spec output
equivalence status has not changed, and the matched c2 lane needs to remain a
visible performance caveat.
