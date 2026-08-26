# XGrammar speculative-decoding correctness backport

## Decision

Radiance's stable vLLM v0.27.1 base now carries two isolated upstream
structured-output fixes:

- vLLM PR [#52805](https://github.com/vllm-project/vllm/pull/52805), merge
  `12f64b39d29282437e35be9aa5db432fb2a1a6e6`: stop speculative token batches
  when the XGrammar matcher reaches termination and clear cached termination
  state on reset.
- vLLM PR [#53046](https://github.com/vllm-project/vllm/pull/53046), commit
  `c6e19b3be24338759a443e03c8325d76da9ee202`: validate drafts generated before
  the reasoning-end grammar bitmask became active, and advance the grammar only
  through drafts that are valid in its current state.

The patches are source-guarded overlays. They do not float the vLLM pin or
change model execution, sampling, R4D kernels, target verification, or draft
generation.

## Production incident correlation

The reported WC-019 attempt `20260826T180129-077Z-b1fd89` produced a malformed,
truncated `doubleClick` call. At `2026-08-26T18:01:33Z`, the production engine
logged `Failed to advance FSM` for request
`chatcmpl-8ee81bbd7568dc83-b5e67108`; it returned HTTP 200 one second later.
The surrounding workload repeatedly logged both failed FSM advancement and
attempts to feed tokens to a matcher that had already accepted its stop token.
The timestamp and failure mode make the grammar/speculative path the likely
cause for this incident, while not claiming that every malformed client result
must have the same cause.

## Why both upstream changes are required

PR #52805 alone passes its focused state-machine regression: a batch such as
`[content, EOS, trailing]` stops at EOS, subsequent advances are benign, and
`reset()` clears both the matcher and its cached termination flag.

It did **not** eliminate the live failure on Qwen3.8 with the Qwen reasoning
parser and DFlash K7. The first candidate returned 30/30 valid multi-tool calls
but emitted **84** `Failed to advance FSM` errors. Those failures came from
post-reasoning drafts that had been proposed before the structured-output
bitmask activated. Feeding them directly to `accept_tokens()` was invalid even
though the grammar had not yet terminated.

Commit `c6e19b3` addresses that distinct boundary. The combined candidate first
uses the non-mutating validation path, then advances only through a valid draft.

Upstream issue [#53181](https://github.com/vllm-project/vllm/issues/53181)
records rare residual XGrammar failures even after #52805. It had no linked fix
when this backport was qualified. The live results below cover the high-rate
Qwen reasoning-boundary failure reproduced on this deployment; they do not
claim that every possible XGrammar/speculative failure is solved.

## Validation

Candidate image:

- tag: `vllm-radiance:xgrammar-spec-candidate`
- local image digest:
  `sha256:c962bcac79386c6226bbfa69ae03eb20197166ec475ba95b6d998b5676e3484a`
- version:
  `0.9.3-dev.vllm0.27.1-r4d0.5.0-mxfp4.dflash2.xgrammar`

Image checks:

- `pip check`: pass
- vLLM/PyTorch/XGrammar imports: pass (`0.27.1`, `2.12.0+rocm7.14`,
  `0.2.3`)
- Compose validation: pass
- focused GPU-free state/patch check: pass
- live health: pass

The sampled gate used the tracked three-tool fixture, `tool_choice=required`,
temperature 1.0, top-p 0.95, top-k 20, reasoning effort medium, DFlash K7,
FP8 KV, TP2, and the production MXFP4 target/drafter pair.

| Run ID | Candidate | Valid calls | FSM errors | Matcher-after-stop warnings | HTTP failures |
|---|---|---:|---:|---:|---:|
| `20260826T182300Z_xgrammar52805-tool-schema` | #52805 only | 30/30 | 84 | 0 | 0 |
| `20260826T182800Z_xgrammar-combined-tool-schema` | combined | 30/30 | 0 | 0 | 0 |
| `20260826T182900Z_xgrammar-combined-tool-schema-100` | combined sustained | 100/100 | 0 | 0 | 0 |

The immutable local run directories retain each request/response, fixture
checksum, summary, and server log.

A bounded c4 smoke used eight fixed-size random requests after one warmup, each
with 512 input and 128 output tokens. All 8 completed, producing 243.4 aggregate
output tokens/s, 391.6 ms mean TTFT, 11.6 ms mean TPOT, 55.8% DFlash acceptance,
and 4.91 mean acceptance length. This is a health/performance sanity check, not
a replacement BetterBench publication run.

## Reproduction

The source-level regression check is intentionally GPU-free and first failed
against the unpatched image:

```bash
python benchmarks/bin/check_xgrammar_spec_termination.py
```

The live sampled gate is:

```bash
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=Qwen3.8-27B \
ATTEMPTS=100 \
benchmarks/bin/run_tool_schema_gate.sh
```

Qualification requires both valid response JSON and no matching server-log
entries for `Failed to advance FSM`, `matcher has terminated`, or
`Unexpected: grammar rejected` during the gate window.
