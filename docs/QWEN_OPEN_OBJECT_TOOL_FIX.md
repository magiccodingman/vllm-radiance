# Qwen open nested-object tool-call fix

## Outcome

Radiance now preserves generic deferred-tool arguments whose outer schema uses
the valid JSON Schema shape `{"type":"object"}` without explicitly spelling
`additionalProperties: true`.

Before the fix, both streaming and non-streaming Chat Completions returned an
empty nested `arguments` string and moved generated fields beside it. The
failure reproduced directly on the provider wire, before any Hermes, MCP, or
application processing. It was distinct from the earlier speculative
XGrammar termination/state bugs.

## Root cause

JSON Schema defines omitted `additionalProperties` as allowed. XGrammar 0.2.3
nevertheless selects different Qwen-XML representations for these semantically
equivalent nested schemas:

```json
{"type":"object"}
```

compiled to recursively nested Qwen parameter tags, while:

```json
{"type":"object","additionalProperties":true}
```

compiled to a JSON-braced object inside the parent parameter. The raw model
tokens from the failing control proved the former path:

```xml
<parameter=arguments>
<parameter=command>
acknowledge_popup
</parameter>
</parameter>
```

vLLM 0.28's `_qwen3_arg_converter` is intentionally flat. It interpreted the
inner `command` parameter as a sibling and serialized the parent as an empty
string. The provider response became equivalent to:

```json
{
  "name": "mcp__smacx__smac_command",
  "arguments": "",
  "command": "acknowledge_popup"
}
```

## Implementation

`patch_qwen_open_object_schema.py` normalizes only plain, semantically open
*nested* objects for the `qwen_3`, `qwen_3_5`, and `qwen_3_coder` structural-tag
models. It adds explicit `additionalProperties: true` to a nested object only
when all of the following are true:

- the schema type is `object`;
- it is not the function's root parameter object;
- `additionalProperties` is omitted;
- no declared or pattern properties are present;
- no `$ref`, `allOf`, `anyOf`, or `oneOf` composition is present.

The transformation preserves standard JSON Schema semantics while selecting
the representation vLLM's existing Qwen parser can round-trip. Explicitly
closed objects, declared objects, composition, and root schemas are unchanged.
The patch is exact-anchor guarded and idempotent.

Named tool choices intentionally retain vLLM's `finish_reason: "stop"` policy.
Required/automatic tool calls return `tool_calls`; that policy was not part of
the argument-corruption defect.

## Qualification

The candidate was an overlay on published production digest
`sha256:4d5474f4b82f8382020f02330e751712aa3cf5cb2b89207dfe52cd02e9247696`.
The local candidate image digest was
`sha256:46f28a71abcd6421a9ebc67c37ec5a4aedc653ed1bdc8e6f2bd4a7918ae84e6f`.
It ran under the exact production MXFP4 target, DFlash K7/fast-draft, TP2,
FP8-KV, PIECEWISE graph, parser, and 256K/C4 configuration. Experimental RX4
quant/stream features remained disabled.

The committed live fixture covers command, LAN, parallel two-call, and memory
shapes in both streaming and non-streaming modes.

| Gate | Result |
|---|---:|
| Unpatched exact production control | 0/8 |
| Candidate first pass | 8/8 |
| Candidate, temperature 1, preserve thinking | 40/40 |
| Candidate, temperature 1, do not preserve thinking | 40/40 |
| Existing sampled required multi-tool gate | 30/30 |

Across the final live qualification window the candidate served 118 requests
with zero XGrammar FSM/termination diagnostics, zero tracebacks/runtime
failures, zero restarts, and a healthy final state.

Immutable synthetic wire captures are stored at:

- `benchmarks/results/qwen-open-object-preserve-thinking/20260901T013349Z_open-object-tool-gate/`
- `benchmarks/results/qwen-open-object-no-preserve-thinking/20260901T013423Z_open-object-tool-gate/`

GPU-free qualification also passed:

```bash
python benchmarks/bin/check_qwen_open_object_schema.py
python benchmarks/bin/check_parser_shared_engine.py --tokenizer /path/to/model
python benchmarks/bin/check_xgrammar_spec_termination.py
pip check
```

Both public and local-example Compose combinations parsed successfully. After
qualification, the machine was restored to the published production digest;
the candidate remains a local image until the merge and publication workflow
produces a release image.
