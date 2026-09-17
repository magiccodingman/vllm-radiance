# Target-only global INT2 candidate selection

The block shortlist retains at most eight tokens per 64-token vocabulary tile.
For sampled `top_k=20`, that can discard required tokens before BF16 reranking.
The legacy capacity fix therefore uses the full target head when
`top_k > min(RADIANCE_DRAFT_RERANK // 4, 8)`.

The default target method now uses a global-256 shortlist, removing that
per-tile quota:

```text
complete INT2 vocabulary projection
    → global top-256 (or top-128)
    → rescore those rows using the original BF16 weights
    → mask other logits to -inf
    → existing sampler
```

The target runs the existing coarse kernel with `KC=0`, retaining its complete
BF16 score vector without emitting per-tile candidates. Selection runs over that
complete vector. The drafter's `KCAND=8`, rerank setting and packed weights are
unchanged. The target reuses the packing and allocates its own temporary output
and candidate buffers.

## Configuration and fallback

```text
RADIANCE_FAST_DRAFT=1
RADIANCE_VERIFY_HEAD=1
RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=256
```

`RADIANCE_VERIFY_HEAD_GLOBAL_TOPK` defaults to `256` in Python and Compose.
Set `128` for a smaller global shortlist or `0` to restore the legacy block
shortlist with its capacity gate. The target optimization still requires
`RADIANCE_VERIFY_HEAD=1` and `RADIANCE_FAST_DRAFT=1`.
Global selection requires TP1, BF16 inputs and
weights, a supported layout and no more than 32 target rows per invocation.
Sampled requests need `0 < top_k <= GLOBAL_TOPK // 4`. This retains an empirical
margin; it is not a completeness certificate. Greedy requests ignore top-k.

The full head handles unsupported shapes/dtypes, embedding bias, grammar masks,
logprobs, sampled min-p, larger/unbounded sampled top-k, penalties, logit bias
(including allowed-token/min-token masking), bad words and thinking-budget
interventions. Unknown sampler layouts and trace-replay mode also fall back.
Set `RADIANCE_VERIFY_HEAD=0` to use the full target head for every request.

**The global path remains approximate.** It can omit a true winner or a member
of the true top-k set. Reading original BF16 weights during reranking does not
guarantee the full GEMM's reduction order and rounding. There is no runtime
certificate to detect these approximation errors. Full fallback is selected by
eligibility checks, not by a claim that global selection verified its recall.

## Measured real-workload results: 60K+ output tokens per method

Eleven intact private Pi coding request boundaries contain **57,008–65,527 input
tokens**, 42–62 messages and 11 tool definitions each. No tools were executed and
no private text, token IDs or tensors are included in this repository. Hardware:
one AMD Radeon AI PRO R9700, ROCm 7.14, PyTorch 2.12.0 and Triton 3.7.1. The TP1
head has 248,320 vocabulary rows and 5,120 hidden dimensions; its BF16 weights
occupy about 2.37 GiB. The pinned model/drafter, fixed D7 speculation, graph
configuration and cache format remain constant across methods.

| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained | Measured tok/s | Estimated tok/s |
|---|---:|---:|---:|---:|---:|
| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) | 63.9 | 63.9 |
| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) | 67.2 | 67.2 |
| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) | 67.5 | 67.2 |
| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) | 66.8 | 67.1 |

### Generated output and throughput

Each method completed **115 natural responses**, with temperature 1, top-p 0.95
and top-k 20. The four methods receive the same prompt/seed in each comparison,
and their order rotates. Each cycle visits all eleven request boundaries, then
uses new seeds. Generation may use all remaining model context, with natural
EOS enabled; no response was length-truncated and no tool call was executed.

| Target path | Measured output tokens | Request time, including first-output wait | Time after first output |
|---|---:|---:|---:|
| Full BF16 | 60,598 | 1,143.453 s | 946.230 s |
| Block-80 | 60,075 | 1,096.052 s | 892.548 s |
| Global-128 | 60,348 | 1,084.678 s | 892.266 s |
| Global-256 | 60,675 | 1,103.268 s | 906.873 s |

Across the 460 measured requests: **241,696 output tokens**, 4,427.450 seconds of
request time and 3,637.917 seconds after first output. These totals exclude the
separate capture pass, warmup and isolated head replay. Throughput divides the
sum of output tokens after each request's first chunk by the summed time after
first output; it is not the arithmetic mean of request rates.

The complete first four-method comparison is uniformly excluded because it
exposed first-use compilation. A separate extension warmed all four methods
before adding matched natural completions until every measured budget still
exceeded 60,000 tokens. An extension bookkeeping failure occurred before GPU
acquisition and was corrected without changing the original measurements.

Full-reference repeat requests reproduced their earlier output on **54/111**
prompt/seed pairs; after excluding the initial pair, this is 53/110. Fifty-five
of the differing pairs have the same reported cached-input count. This remains
an unresolved repeatability difference: it does not establish a head, cache,
sampler or scheduling cause. Within the timed comparisons, block/global-128/
global-256 outputs match the corresponding full-head output on **100/105/102 of
115** pairs. These whole-output counts are separate from head recall and are
not a quality score.

Global-256 measured 4.5% faster than full BF16. The estimate holds full-head
output and observed verification counts fixed, replacing only measured M8 head
cost. It predicts about 67.1–67.2 tok/s for all fast methods. The measured
end-to-end differences also include changed natural continuations, speculative
acceptance and execution variation; they cannot be attributed solely to head
cost. No TP2 or certified-adaptive timing is claimed.

### Identical-hidden-state accuracy and head latency

A separate full-reference pass generated **61,561 tokens across 111 natural
completions**. Every consecutive head invocation was captured: **15,191 calls,
119,988 prediction rows**, comprising 220 single-row calls and 14,971 eight-row
calls. This includes prefill and rejected speculative candidates. All calls
were replayed; none were selected or discarded based on success. The standalone
full head reproduced every saved in-model logit digest, dtype and shape exactly.

Top-1 match compares the final argmax token ID with the full head. Complete
top-20 retention requires every reference token at or above its twentieth score
to remain finite, including ties. It does not imply identical retained scores,
ranking or sampling probabilities. These percentages describe the observed
sample; they are not general reliability estimates.

Global-256 reduced complete-top-20 misses from **21,536 to 202**, adding
**0.042 ms** to median head time versus block-8. Both global methods retained the
reference argmax in every row, but each produced two different final argmax IDs.
The block/global-128/global-256 methods produced **2,378/3,550/6,295 retained-logit
differences**, respectively, with maximum absolute difference 0.25. The extra
logit count also reflects the larger number of rescored candidates; it is not
directly a per-method error probability. The result does not certify computation
upstream of the captured hidden vectors.

Latency uses interleaved GPU-event timings after warming every method/shape.
Up to 256 deterministic capture indices receive five timing repetitions per
method; **1,265 timings per method** have M8 shape and enter the headline table.
Capture files are selected by index in their stored filename ordering, not by
an assertion of evenly spaced chronological samples. The benchmark exercises
the numeric prototype used by the global path; public integration eligibility
checks are separate. M8 head latency is neither a whole-model step nor one
emitted token's latency.

Aggregate results: [long-run summary](../benchmarks/results/20260916-verify-head-global-topk-long/summary.json),
[table](../benchmarks/results/20260916-verify-head-global-topk-long/table.md) and
[validation](../benchmarks/results/20260916-verify-head-global-topk-long/validation.json).
The [earlier 650-row / 433-output-token-per-method pilot](../benchmarks/results/20260916-verify-head-global-topk/summary.json)
is retained as historical evidence; the table above uses the longer run.

## Regression checks

```text
python -m pytest -q tests
RADIANCE_TEST_NATIVE=1 python -m pytest -q tests
```

The implementation passed **135 tests with no skips** on the R9700 at revision
`dc3487c`, before global-256 became the default. The default-selection change
passes 117 CPU cases and explicitly skips the 19 GPU cases; it changes no
numeric kernel or fallback predicate. Native tests were not repeated for this
configuration change. Syntax checks and focused lint checks pass.

CPU checks execute the actual dispatch functions, including mixed/reordered
requests, unset-option global-256 dispatch, explicit legacy selection and
global-specific restrictions. Native checks use a
public synthetic fixture with twenty strictly positive winners in one tile. The
old block path loses eleven strictly required tokens; global selection retains
the complete top-20 and matches reference scores on that fixture. This fixture
isolates selection capacity rather than claiming INT2 is exact in general.

Native cases cover target row counts 1/2/3/8/16/32, both candidate depths,
unchanged drafter results, complete fallback for unsupported inputs, and the
public hook's global → full → global transition. Tied zero-score candidates
outside the final top-20 may occupy different shortlist slots; tests compare the
specified top-20 sampling support and scores rather than requiring an arbitrary
tie ordering from `torch.topk`.
