# Qwen tool grammar interoperability

## Observed failure and scope

The vLLM 0.30.0 / XGrammar 0.2.3 runtime skips automatic tool grammar generation
when no tool explicitly sets `strict=true`. Hermes's MCP schema conversion does
not currently add that field. This is separate from the older open-object Qwen
XML defect: that earlier normalization was already installed during diagnosis.

A synthetic direct request with the reporting input schema and `strict=true`
failed before model generation. The schema exposed an integer `maximum` as
`9.223372036854776e+18`. XGrammar rejected the floating-point Int64 boundary;
vLLM's automatic fallback then sent the modern structural tag to Guidance's
legacy-only parser, hiding the compiler failure behind `Invalid grammar
specification: 'triggers'`. A CPU-only control without that maximum compiled.
Removing the maximum was diagnostic only, not the implemented repair.

## Overlay and explicit operator choices

`patch_qwen_tool_grammar.py` runs after `patch_qwen_open_object_schema.py` in
all three image builds. It preserves schema copies (including Responses API
parameters), converts whole floating-point integer bounds to exact integers
only when representable in Int64, and prevents the incompatible modern-tag
Guidance fallback. Other fallback paths and legacy structural tags remain.
Failure messages contain no raw tool schema or model prompt.

Two optional environment switches default to **off**:

- `RADIANCE_QWEN_AUTO_TOOL_GRAMMAR=1`: builds the Qwen grammar for automatic
  tool choice even without a tool's strict flag. Applies only to the three
  Qwen XML formats. It does not force a tool call, alter `tool_choice=none`,
  or override vLLM's global `VLLM_ENFORCE_STRICT_TOOL_CALLING` opt-out.
  On vLLM 0.33 it restores the legacy parameter-schema default for omitted
  `strict` flags; an explicit `strict=false` still leaves arguments free.
  With this switch off, native vLLM strictness rules apply. The independent
  server floor `--tool-strict-level=parameter` forces parameter schemas for
  all tools according to upstream policy, including explicitly non-strict tools.
- `RADIANCE_QWEN_INT64_BOUND_COMPAT=1`: for schema exporters that round an
  Int64 maximum to the floating-point value `2**63`, explicitly narrows only
  that inclusive maximum to exact `9223372036854775807`. This is an operator
  compatibility decision, **not** a mathematically identical rewrite for
  arbitrary schemas intentionally admitting `2**63`. Exact integer bounds
  are preferable. Unsupported other out-of-range floating bounds reject
  with guidance; no bounds are silently dropped. Number schemas, annotations,
  examples, defaults, and instance data are not rewritten.

Publishing an image enables neither switch. Enabling them on a shared server
requires an authorized rollout and actual wire-level validation. Do not patch
the installed production package in place or change a global Hermes profile.
Rollback is the retained immutable previous image plus its exact environment
and launch settings; do not remove its image or caches before acceptance.

## Tests and evidence boundaries

`python tools/radiance.py test pr` includes GPU-free helper/patch contracts.
`python benchmarks/bin/check_qwen_tool_grammar_image.py` additionally requires
the installed patched image, but does not load model weights or use GPUs.
It checks auto/required/named, reasoning on/off, Chat/Responses nonmutation,
and actual grammar compilation plus fail-closed modern fallback.

The existing sampled tool-schema and open-object wire gates remain required
before production qualification. Neither a successful compiler check nor a
single successful generation establishes full autonomous agent success,
model-quality equivalence, throughput, or DFlash correctness. The observed
failure does not establish that all earlier malformed calls share one cause.

## Operations and observability

No new service, credentials, persistent state, ports, GPU kernels or backup
requirements. The existing inference endpoint and availability monitoring own
the service; expected schema rejection remains a client-visible validation
failure, not evidence of a healthy completed tool call. Check request failure
rates and actual synthetic tool outputs after authorized rollout, without
capturing customer prompts or tool data. No live monitoring configuration or
production runtime is changed by this source patch.
