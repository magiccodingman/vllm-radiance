# vLLM0.33 bounded serving evidence

Candidate image `sha256:0544917159f9d858ac74408ee3afa0a189104b4970b234f92e504495860034f4`, source `1d485b3c5f18c5db15b58a0b26ab27fac685ed6a`. All GPU processes expose only GPU0 (PCI0000:03:00.0/renderD128, count1 verified through HIP); production and GPU1 were untouched.

## Exact candidate CPU and native-payload identity

`cpu-05/` passes all 19 registered installed CPU/image checks with return0 on the stable clean1d checkout. The earlier cpu04 run advanced its checkout during execution and is not used as exact-source evidence.

`unchanged-native-image-payload-01.json` records byte-identical LFM model/convolution source and r4d/MXFP4 shared objects between this image and b2/source 451, which passed all eight native gates (including326 unchanged LFM and138 verify-head tests). Those unchanged numerical suites were not needlessly repeated; this is explicit artifact-identity evidence, not a claim of a separate native run. Changed Python integration is protected by cpu05 and the real model runs.

## Explicit Qwen integration controls

Both Quark live profiles set `RADIANCE_QWEN_AUTO_TOOL_GRAMMAR=1` and
`RADIANCE_QWEN_INT64_BOUND_COMPAT=1` explicitly, as recorded in their manifests.
The reusable global defaults remain0. These results qualify that client-shaped
integration lane; they do not claim equivalent wire behavior with those controls
disabled. V1 below identifies the preserved runner default, not an assertion
that every portable setting was used unchanged (the tests are TP1 and bounded).

## Quark27B TP1 V2

`qwen-v2/` records native Quark/W4A8+FP8KV/R4D, full target head, non-speculative, graphs, APC,4K / 2 sequences/70% budget. Resident owners prove304 W4A8 layers with zero reported fallback. Native runner/model owners, exact fresh/repeat/independent token trajectories, actual1,568-token prefix-cache reuse, one-token recompute, nested JSON, named tool, bounded prefill and a synthetic red-image repeat pass. All 30 sampled tool-schema attempts and40 open-object streaming/non-streaming cases pass. Raw requests/responses/SSE/metrics and commands are retained. Recorded timings are incidental; no performance comparison is claimed.

The first `qwen-startup-failure/` preserves the successful weight load followed by the old R4D block-query API TypeError. The repair accepts v0.33's cache-spec argument without changing the exact16token kernel-block contract. A real upstream cache-planner CPU check passes for16/1568/2240token framework pages, and the subsequent V2 run reaches serving.

## LFM negative controls

The original LFM live results and exact batch-equivalence failure remain in the adjacent `20261107-v033-lfm3-live/` report. The official-convolution control here removes only the width-7 FP32-products adaptation. It passes that particular single/batch token fixture, though logprobs still differ between batch members. This narrows the investigation; it does not demonstrate universal upstream batch invariance or justify reverting the correction required by the unchanged numerical suite. Upstream's batch-invariant mode rejects SHORT_CONV_ATTN. LFM remains an optional experimental profile with the failure visible.

## Remaining scope

V1 and the bounded FP8 DFlash smoke pass as recorded below. TP2, matched performance, capacity and release qualification remain deferred. This MR does not merge release or deploy production.

## Quark27B TP1 V1 (preserved portable runner default)

`qwen-v1/` records the same native Quark/W4A8+FP8KV/R4D and full target head with V1, APC,8K / 2 sequences/70% budget. The8K context admits the canonical6144completion-token client budget plus up to1,311 prompt tokens; the fixture/caps were not reduced. All 40 client-shaped streaming/non-streaming cases pass, including independent parallel calls, nested packages, strict/non-strict schemas and the Int64 exporter boundary. The full resident/cache/recompute/structured/tool/vision check passes with expected_runner=v1,304 W4A8 layers and zero reported fallback, and measured1,568-token prefix reuse. All 30 sampled tool-schema attempts and40 open-object wire cases pass. Own test server stopped after completion; no production configuration was touched.

## Bounded native FP8 DFlash2 K7 smoke

`dflash-smoke/` passes on the same054/source 1d image with Quark W4A8 target,
FP8 KV/R4D, native FP8 DFlash2 drafter (pinned fixture revisionf02593d0), TP1
for both models, PIECEWISE graphs,4K / 2 sequences, full target head (`FAST_DRAFT=0`)
and explicit83% GPU budget. Worker inspection confirms the native
DFlash2Qwen3ForCausalLM, FP8 draft linears and seven speculative steps. Real
short completion/SSE token parity and a strict named lookup call pass. Native
metrics record148 draft and80 accepted tokens, demonstrating actual drafting
rather than a disabled path. This is text-only smoke evidence, not throughput,
matched performance, serial-versus-speculative equivalence or TP2 qualification.
Speculation remains opt-in.

`dflash-memory-failure/` preserves the first80%-budget startup refusal:
4K requires1.58 GiB KV while1.29 GiB was available after normal graph profiling.
The retry changes only budget to83% (with the same GPU0 HIP+headroom guard),
port/run identity and reuse of that stopped run's exact-source JIT cache.
Context, graphs, source pins, quality and gate assertions are unchanged. The
passing helper stops only its own server after collecting immutable evidence.

Both dev Dockerfile BuildKit preflights pass without warnings against the built
candidate. The final nine CPU PR checks pass. All test servers were stopped;
no merge into main/release, publication or production deployment was performed.
