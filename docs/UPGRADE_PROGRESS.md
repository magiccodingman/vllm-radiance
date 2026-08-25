# RDNA4 upgrade progress

> Continuation: the post-v0.27 development pin was replaced with stable vLLM
> v0.27.1 after a structured-output regression was reproduced and isolated.
> See [V0271_TOOL_SCHEMA_STABILITY.md](V0271_TOOL_SCHEMA_STABILITY.md) for the
> 30-request tool-schema gate, retained compiler/R4D stack, compatibility work,
> and BetterBench qualification. Sections below remain the immutable history of
> the earlier upgrade and DFlash2 investigation.

This document is the durable checkpoint log for merge request !5. All performance
numbers use native FP8 weights and mandatory FP8 KV cache with the reusable
`benchmarks/` workload contract. Positive deltas mean improvement.

## Reproducibility pins

- Model: `/nvme/lexar-2/ai/models/Qwen3.8-27B-heretic-ara-fp8-magiccodingman`
  (28.75 GiB safetensors; checksum validation passed).
- vLLM main: `a014e35f38c80fb0652387740193ad2147fed6a3` (2026-08-22).
- Radiance 0.4.0 image: `sha256:f233e3e071653adac6821f9582070a941430a6b795482f6ff04115b17df37047`.
- Radiance 0.5.8 image: `sha256:2e788f7475907cc86d82acddb5f1e50360b1ffd39e1ffbd2d00ac705b728ccc1`.

## Benchmark contract

- TP2: 85% GPU-memory limit, 16K server envelope, routine workloads stop at 8K,
  decode concurrency 1/2/4/8. This reserves about 4.8 GiB outside vLLM's
  allocation on each card for a DFlash2 drafter and runtime variation.
- TP1 reference: native/no-offload, eager, 95%, 8K, decode concurrency 1/2/4.
  It is explicitly selected for this 27B baseline and is not part of the
  model-neutral default, so larger checkpoints do not inherit a capacity-edge
  single-card run.
- Decode: 256 input + 256 forced output tokens, two repetitions and at least two
  request waves.
- Prefill/mixed: 2048 input + 64 output; TP2 concurrency 1/4/8, TP1 concurrency 1.
- Near-envelope context and longer repetitions are qualification-only. An
  explicitly raised 32K envelope adds a 32K capacity case, but the routine gate
  is not a maximum-throughput or VRAM-saturation search.
- FP8 KV is a hard precondition checked from the resolved container command.
- Each shape receives a disjoint-seed warmup wave, and every measured repetition
  has its own deterministic seed. This heats lazy kernels without allowing
  prefix-cache state to leak into a measurement.
- The normalized contract explicitly uses greedy `temperature=0`; sampling
  policy is a comparison key rather than an inherited server default.

This normalized 16K/85% contract supersedes the early 32K/90% server envelope;
the older runs remain immutable historical evidence and the retained software
checkpoints are rerun under the normalized contract before final comparison.

The checkpoint lacks calibrated attention q/k/v/prob scale tensors, so vLLM
records that FP8 KV attention uses scale 1.0. Dynamic KV scale calculation is
not forced on this hybrid GDN model because the current vLLM path disables it
for hybrid recurrent-state correctness. This remains a documented accuracy
risk to revisit on pinned main.

## Baselines

### Radiance 0.4.0

The corrected matrix completed TP2 non-spec, TP1 non-spec, and TP2 MTP. TP1 MTP
is unsupported natively: creation of the draft LM head requested 2.37 GiB with
only 1.06 GiB free. It is no longer in the routine matrix and is not forced with
CPU offload.

Key non-spec output TPS:

| Lane | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| TP2 | 34.24 | 61.62 | 117.12 | 215.53 |
| TP1 | 17.08 | 32.01 | 61.85 | excluded going forward |

TP2 prompt-token throughput (total TPS) at 2K input was 893.83 / 1557.12 /
2217.96 for concurrency 1/4/8. TP2 MTP output TPS was 45.62 / 66.33 / 135.73 /
239.03 at concurrency 1/2/4/8, with workload-dependent acceptance recorded in
the raw result.

### Radiance 0.5.8 source-head checkpoint

The non-spec quick gate completed in about 16 minutes. Against 0.4.0, TP2 decode
was -0.66%, -0.24%, +1.18%, and +1.22% at concurrency 1/2/4/8. TP2 2K prefill
total TPS improved +2.28%, +1.83%, and +1.01% at concurrency 1/4/8, while median
TTFT improved roughly 4.6-5.1%.

TP1 c1/c2/c4 stayed within +0.6-1.3%. TP1 c8 was capacity-distorted: 0.5.8
retained 28,216 KV tokens (3.44 full 8K requests), produced ~18 second tail
TTFTs, and lost 29.8% aggregate TPS while per-token latency stayed flat. This is
why TP1 now stops at c4; TP2 remains the c8 concurrency lane.

## Pinned-main patch audit

- Removed the old broad AITER enablement patch. Current vLLM deliberately routes
  RDNA4 Triton kernels separately from CDNA CK/MFMA/ASM kernels.
- Removed the sampler workaround now owned upstream.
- Removed Radiance's RMS/group-FP8 fusion patch because pinned main selects the
  native quant matcher on RDNA4 itself.
- Retained deterministic gfx1201 discovery, FP8 preshuffle/dispatcher and split-K
  alignment, unified-attention LDS fit/tuning, GDN triangular-solve WMMA, TP2
  custom all-reduce, router GEMM, and MTP correctness/controller hooks.
- Every retained vLLM/AITER patch was applied to a scratch copy of the exact
  pinned sources with its anchor-count and Python-parse guards enabled.

## Pinned-main smoke checkpoint

The fast iteration image built successfully as
`vllm-radiance:dev-a014e35` (local digest
`sha256:067f249f15ea6b2f63672c4bd706d80d084aaaaee65b24ed2023bc6b98e87d67`).
The exact stack is vLLM `0.28.0.dev0+a014e35`, PyTorch 2.11.0 ROCm 7.14,
Triton 3.6.0, and AITER 0.1.17.

TP2 non-spec smoke passed with native FP8 weights and mandatory FP8 KV. Startup
confirmed the Radiance preshuffle hook, unified-attention tuning, custom TP2
all-reduce plus FP8 payload path, Triton/FLA GDN prefill, and Triton GDN decode.
At the bounded 0.90 memory setting, each GPU used 13.98 GiB for model weights,
reserved 0.75 GiB for graphs, and retained 10.59 GiB for KV (525,501 tokens,
16.04 full 32K requests). This leaves the intended non-engine headroom instead
of tuning to the VRAM edge.

## Canonical pinned-main performance checkpoint

Early exploratory runs were retained but rejected after two benchmark defects
were observed directly: lazy Triton JIT inside measured cases, then built-in
warmups reusing a measured prompt through prefix cache. The canonical runs use
the corrected disjoint-seed contract:

- Radiance 0.5.8 TP2 baseline: `20260822T222623Z_..._quick`
- Pinned-main/AITER 0.1.17 TP2: `20260822T223332Z_..._quick`

Pinned main is effectively neutral to mildly positive for decode: +0.51%,
+0.55%, +0.29%, and +0.52% output TPS at concurrency 1/2/4/8. The 8K context
case was -1.55%. Aggregate 2K prefill throughput was -0.46%, -0.80%, and -1.48%
at concurrency 1/4/8; median TTFT improved at concurrent prefill, but TPOT became
worse enough that no broad prefill win is claimed. Decode CV was at most 0.05%,
which is a strong stable-rebase checkpoint.

## AITER 0.1.20 checkpoint

AITER was upgraded in isolation from exact commit
`fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`, built for gfx1201 with
`AITER_USE_SYSTEM_TRITON=1`. Image
`vllm-radiance:dev-a014e35-aiter0.1.20` has local digest
`sha256:b1e303dc0b6bbc7b0c6eac1c727b4a5e1c0035b9c4f1afb74718ac69afb57e40`.
It passed smoke and the canonical TP2 gate.

Against AITER 0.1.17, all measured output-throughput deltas were within -0.24%
to +0.29%. That is expected: Radiance's dense preshuffle dispatcher is retained,
and vLLM still selects Triton/FLA GDN prefill plus Triton GDN decode. AITER 0.1.20
is therefore a safe foundation, but its new GDN prefill work requires an explicit
vLLM bridge before it can produce a material gain.

The constrained TP1 lane was also normalized and compared against Radiance
0.5.8. Pinned main plus AITER 0.1.20 was within -0.21% to -0.25% output TPS for
decode c1/c2/c4, -0.23% for 2K prefill, and -0.21% for the near-8K context case.
Decode CV was at most 0.06%. This confirms that the rebase did not trade away
single-card performance, and TP1 c8 remains excluded as a capacity confounder.

## Experimental AITER GDN prefill bridge

An opt-in vLLM bridge now passes host-derived sequence lengths into AITER
0.1.20's reusable GDN schedule and selects its gfx1201 HIP/WMMA K5 recurrence.
A synthetic variable-length, TP2-shaped correctness test against AITER's Triton
reference passed: output max absolute error was `4.8828125e-4`, final-state max
absolute error was `2.962425e-4`, and all values were finite. A full TP2 server
smoke then passed with the real FP8 model, FP8 KV, and the AITER backend.

The first bounded end-to-end gate did not justify enabling it by default.
Against the exact AITER 0.1.20 control, 2K prefill output TPS ranged from -1.49%
to -0.38%, the 8K case was -0.73%, and decode was consistently -1.37% to
-1.63%. This is below the hoped-for model-level improvement even though the K5
kernel itself is correct. The backend therefore remains explicit-only through
`--additional-config={"gdn_prefill_backend":"aiter"}` for future profiling;
`auto` retains the established Triton/FLA path.

The upstream audit also corrected one research assumption before it could
affect the build: as of 2026-08-23, AITER PR #4732 (the newer 24-commit FlyDSL
K5/fp32-snapshot series) is still open, not merged, and its published timings
are on gfx942. It is not silently pulled into this stability branch. Likewise,
AITER PR #4868's generic RDNA LDS guard is still open; Radiance's broader,
already-tested 2D/3D LDS guard remains in place. vLLM PR #52816 is merged and
present in the pinned tree, so DFlash2 itself is available without an extra
cherry-pick.

## Official AMD compiler-pair checkpoint

The final build moves the compiler stack atomically to AMD PyTorch commit
`6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5` (2.12.0+rocm7.14), AMD Triton
commit `f0b55c07da61c71775bef6d1a15ebf846430ac75` (3.7.1), torchvision 0.27.1,
and AITER 0.1.20. AITER uses the system Triton. The first image audit caught
that a no-dependencies AITER install omitted its pinned `flydsl==0.3.1`
runtime requirement; the release Dockerfile now installs that dependency and
requires `pip check` to pass in the final image.

The corrected image is
`vllm-radiance:dev-a014e35-amd212-fixed` (local digest
`sha256:92594dd0596400a4db49de61d1530341282ef20d74932db69ace821df0beaa6b`).
Its cold TP2 smoke passed with native FP8 weights and FP8 KV. Peak measured
VRAM use was 80.61%, leaving 6.18 GiB physical headroom on each R9700.

Normalized quick checkpoints:

- Radiance 0.5.8: `20260823T015503Z_..._quick`
- pinned main + AITER 0.1.20 on the established compiler:
  `20260823T021006Z_..._quick`
- final AMD compiler pair: `20260823T022446Z_..._quick`

Against Radiance 0.5.8, the final pair is effectively neutral for TP2 decode:
-0.20%, -0.05%, -0.25%, and -0.02% at c1/c2/c4/c8. Its 2K prefill output
throughput is -0.61%, +0.79%, and +0.39% at c1/c4/c8; the 8K context case is
about -0.93%. TP1 decode is roughly -0.5% to -0.8%. The compiler upgrade is
therefore selected as a current upstream foundation, not claimed as a broad
speedup.

`HIP_FORCE_DEV_KERNARG=1` was neutral/noisy (-0.49% to +0.18%) and remains
off. Forcing `TRITON_ATTN` was a clear regression: decode lost 0.6-1.6%, 2K
prefill lost 8.4-18.9%, and the 8K context case lost 45.6%. Unified AITER
attention remains the default. The explicit AITER GDN prefill bridge became a
small positive signal under the new compiler (+0.5-1.0% output TPS and roughly
1-2.6% median TTFT), but that focused lane has one prefill sample per shape;
it remains opt-in until a repeated milestone establishes the margin.

## Radiance-path control experiments

The rebase retains its custom paths because matched controls show they are
still material on dual R9700s:

- Disabling Radiance weight preshuffle and using upstream RDNA4 AITER linear
  regressed every decode and prefill case by 8.6-11.4%.
- Disabling the Radiance TP2 custom all-reduce regressed decode by 7.6-8.3%.

The immutable evidence is stored in `20260823T030410Z_..._quick` and
`20260823T031241Z_..._quick`, including comparison reports. These two paths,
the split-K alignment fix, and the broader 2D/3D attention LDS guard remain
enabled.

## DFlash2 qualification

The stable BF16 target
`Qwen3.8-27B-heretic-ara-heretic-org` and the 3.58 GiB
`Qwen3.8-27B-DFlash2-z-lab` draft load and serve successfully with the DFlash2
V2 runner, TP2 draft sharding, `TRITON_ATTN` for the draft, eager execution,
and mandatory FP8 KV. A 90% cap had no cache blocks after the combined model
load (`27.45 GiB` per GPU); the smallest useful constrained lane is 2K at 92%,
which supplies 2,595 KV tokens and leaves about 2.45 GiB physical VRAM per GPU.

On deterministic random-token decode, DFlash2 improved BF16 output TPS by
+174% at c1 and +60% at c2, but lost 28% at c4 and 60% at c8 because this
capacity-constrained lane cannot sustain concurrent draft work. It is not a
recommended deployment profile. Exact greedy output matched the smoke but
failed the broader gate. A forced-V2 non-spec control reduced, but did not
eliminate, the difference. Eight fixed meaningful prompts remained coherent,
but four diverged after 124-411 matching characters. Eager mode rules out the
open FULL-cudagraph prefill-dispatch bug; the strict byte-equivalence claim is
not accepted under FP8 KV.

The requested experimental native-FP8 target also loads and runs DFlash2. In
an 8K/85% eager lane it retained 4.34-4.38 GiB free VRAM per GPU and delivered
48.50 / 76.08 / 175.74 / 247.88 output TPS at c1/c2/c4/c8, or +186% / +151% /
+179% / +101% against the matched V2 non-spec control. Acceptance length was
4.15-4.56. However, all eight fixed meaningful greedy outputs differed from
the matched non-spec server, sometimes near the beginning. Per the experiment
scope, this path is recorded but not fixed or qualified as a default.

The DFlash evidence is in `20260823T033102Z_..._quick`,
`20260823T035820Z_..._quick`, and `20260823T040506Z_..._quick`, including the
machine-readable failed correctness reports. Non-spec serving remains the
compose default.

## Selected state

- Release identifier: `0.6.0-dev.a014e35`.
- Default serving/portable benchmark envelope: TP2, native FP8 weights,
  mandatory FP8 KV, 16K, 85% GPU allocation, maximum 8 sequences, 2,048
  batched tokens, unified AITER attention, speculative decoding off.
- The portable gate is intentionally model-neutral and does not search for the
  VRAM limit. TP1 is an explicit fit-specific reference, so future 35B models
  can retain the same TP2 c1/c2/c4/c8 contract without inheriting an unsafe
  single-card lane.
- DFlash2 is available for continued experimental work but is not enabled by
  default until its FP8-KV output-equivalence and concurrent behavior are
  resolved.

Final non-spec qualification `20260823T042255Z_..._qualification` completed in
about 13 minutes with zero failed requests. Three-sample decode CV was
0.13% / 0.16% / 0.14% / 0.30% at c1/c2/c4/c8; output TPS was 35.25 / 66.20 /
125.79 / 222.84. Sustained 512-token output reached 35.39 / 67.22 / 127.58 /
229.10 TPS. The concurrent 16,128-token context case completed both requests.
Peak qualification VRAM was 83.44%, leaving at least 5.28 GiB free per R9700;
peak junction temperature was 79 C and peak board power was 249 W.

## DFlash2 continuation (!6)

The follow-up optimization recovered 98.1-99.8% of ordinary Radiance base
decode performance under the DFlash-compatible V2 runner by replacing eager
execution with `PIECEWISE` graphs. A selective-FP8 K5 drafter then qualified
operationally at 93.78 / 138.18 / 225.96 / 399.81 output TPS for c1/c2/c4/c8,
with about 4.41 GiB physical headroom per R9700. Sustained c8 decode reached
541.74 TPS.

Strict greedy equivalence still fails deterministically (1/8 exact prompts in
the matched 128-token gate), so DFlash remains disabled by default. The controls
attribute the result to speculative target-verification shape numerics rather
than FP8 KV, V2 itself, prefix cache, hybrid-GDN state corruption, the drafter,
or custom TP2 reduction. The complete experiment matrix, exact run IDs,
checksums, upstream audit, negative results, and deployment recommendation are
in [DFLASH2_OPTIMIZATION.md](DFLASH2_OPTIMIZATION.md).

## libr4d 0.4.0 / DeadCode 0.7.4 continuation (!9)

DeadCode's Radiance 0.7.4 operator work was ported without reverting the pinned
vLLM-main/AMD compiler foundation. R4D attention, GDN, vision, TP2 collectives,
router, and the optional INT2 MTP head now build from exact libr4d v0.4.0 commit
`000d5f91d0e47ee9faf3b5466f0a12995f0cbfd6`. R4D attention and GDN replace the
older AITER/FLA defaults; the older paths remain measured fallbacks.

The portable default is now TP2, native FP8 weights, mandatory FP8 KV, R4D,
16K, 85% allocation, maximum eight sequences, 4,096 batched tokens, and
speculative decoding off. This supersedes the 2,048/AITER selected-state text
above while retaining it as historical evidence.

BetterBench v0.2.2 (10 passes/category) measured 35.8 weighted non-spec decode
tok/s and 35.5/67.0/118.4/187.6 aggregate tok/s at c1/c2/c4/c8. That is decode
parity with immutable DeadCode 0.7.4 (35.7 weighted and
35.6/66.9/117.4/190.7), although fork non-spec cold prefill is 2.7-3.3% lower.

The opt-in INT2 MTP head reached 93.7 weighted tok/s. Selective-FP8 DFlash K5
reached 99.4, and K7 won at 112.6 weighted plus
102.1/189.0/305.1/496.1 c1/c2/c4/c8. K7 retained at least 6.47 GiB physical
VRAM headroom per card. Strict speculative versus non-spec equivalence remains
failed (3/8 exact prompts), so the performance winner is not promoted to the
generic default. Full pins, run IDs, acceptance counters, correctness
isolation, and negative results are in
[LIBR4D_BETTERBENCH.md](LIBR4D_BETTERBENCH.md).
