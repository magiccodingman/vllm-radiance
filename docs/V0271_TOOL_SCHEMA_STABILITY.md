# vLLM v0.27.1 tool-schema stability qualification

## Decision

Radiance now pins stable vLLM v0.27.1 at commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. DFlash2 is present in this
release, so the post-release development pin is no longer required for the
fork's measured speculative lane.

The rest of the qualified performance stack is retained: AMD PyTorch 2.12,
AMD Triton 3.7.1, AITER 0.1.20, ROCm 7.14, libr4d 0.4.0, the Radiance FP8
dispatcher/preshuffle path, and custom TP2 all-reduce. Transformers is pinned
to 5.14.1, matching stable Radiance's compatibility boundary for Gemma-4;
XGrammar is pinned to the qualified 0.2.3 structured-output runtime.

Final local image:

- Tag: `vllm-radiance:v0271-tool-schema-candidate`
- Image ID / local manifest-list digest:
  `sha256:b3c0b006e1afdb8ec3838eaa8a9841bf670584783af4adfdf92d77f0302fbbbd`

## Reproduced failure

The former vLLM pin was
`a014e35f38c80fb0652387740193ad2147fed6a3`, reported as
`0.28.0.dev0+a014e35`. With the ARA native-FP8 target, FP8 KV, Qwen3 Coder
tool parser, and a sampled three-tool `tool_choice=required` request, required
arguments were intermittently omitted. The initial production-shaped run
passed only 20/30 requests. Controls showed that the failure:

- persisted without DFlash2 (20/30);
- persisted under V1 eager (24/30) and V2 eager (24/30);
- occurred with a named tool (16/30) and even a single tool (26/30);
- did not depend on the custom Radiance chat template;
- did not reproduce in stable Radiance/vLLM v0.27.1 under either V1 eager or
  V2 piecewise execution (30/30 for each control).

The invalid token trace closed `</function>` before the required `ref` field.
A standalone XGrammar test rejected that closing token and the ROCm GPU mask
set its logit to negative infinity. The live post-release runner nevertheless
emitted it. This localizes the defect to the post-release vLLM structured-output
integration rather than the model, Qwen parser, schema, sampler seed, R4D
attention, speculative decoder, or XGrammar grammar itself.

The complete diagnostic control matrix is retained locally under immutable run
ID `20260824T220045Z_tool-schema-radiance-investigation`, including raw
responses, extracted token traces, and the standalone grammar-mask control.

The investigation did not identify one proven upstream source commit as the
sole cause because the stable comparison image also differed in compiler
dependencies. The controlled candidate in this change narrows that further:
only vLLM was returned to v0.27.1 while AMD PyTorch 2.12, Triton 3.7.1, AITER
0.1.20, ROCm 7.14, and libr4d 0.4.0 were retained, and the gate passed. The
release pin is therefore a verified mitigation and the appropriate stable
foundation, without claiming an unproven line-level root cause.

## Final regression gate

The tracked fixture is
`benchmarks/fixtures/tool-schema-multitool.json`; the reusable runner is
`benchmarks/bin/run_tool_schema_gate.sh`. It sends 30 sampled requests at
temperature 1.0, top-p 0.95, top-k 20, with `click`, `keypress`, and `assert`
schemas available and `tool_choice=required`. A pass requires `click` plus the
exact required `ref=e12` and all other required arguments on every request.

Final exact-image run:

- Run ID: `20260825T001000Z_v0271-final-image-ara-tool-schema`
- Target: `Qwen3.8-27B-heretic-ara-fp8-magiccodingman`
- Chat template: checkpoint-native (no server override)
- Result: **30/30 passed**

The same dependency set also passed 30/30 before the final reproducibility
rebuild (`20260824T233000Z_v0271-transformers5141-ara-tool-schema`). An earlier
candidate with the same vLLM pin and Transformers 5.15.1 passed 30/30 as well
(`20260824T232000Z_v0271-ara-tool-schema`), before Transformers was pinned to
the stable 5.14.1 compatibility release.

## Chat-template policy

The portable Compose uses the checkpoint-native template. This is intentional:
the image serves multiple model families, and a Qwen-specific template must
not be imposed globally. The enhanced Radiance Qwen template remains in the
repository for explicit deployment overrides. The template was not the cause
of this regression, but making it opt-in avoids unrelated cross-model behavior
changes and matches the configuration qualified above.

## Performance qualification

The final ARA FP8 non-spec candidate was exercised with the established
BetterBench v0.2.2 standard contract (10 passes/category, c1/c2/c4/c8, 8K,
FP8 KV, TP2, R4D, 85% allocation, 4,096 batched-token budget).

- Run ID: `20260824T233300Z_v0271-final-ara-betterbench`
- Weighted single-stream decode: **35.3 output tok/s**
- Concurrency: **35.3 / 66.8 / 116.3 / 191.6 output tok/s** at
  c1/c2/c4/c8, with 24/24 successful requests at every level
- Weighted TTFT p50: **58 ms**
- Cold prefill: **4,090 / 4,262 / 4,218 prompt tok/s** at the
  2K/4K/7K target depths
- Minimum physical VRAM headroom: **6.65 GiB per target GPU**
- Peak observed board power: **263 W**

The performance run used the immediately preceding candidate image, which
already resolved and reported XGrammar 0.2.3. The final rebuild only made that
resolved version an explicit Dockerfile pin; it did not change the executable
dependency set. The 30/30 exact-image schema run above was repeated after that
rebuild.

The historical apples-to-apples reference is the merged R4D non-spec result in
`docs/LIBR4D_BETTERBENCH.md`: weighted single-stream 35.8 output tok/s, c1
35.5, c2 67.0, c4 118.4, and c8 187.6 aggregate output tok/s.
Relative to that reference, the stable vLLM candidate changed weighted
single-stream by -1.4%, c1 by -0.6%, c2 by -0.3%, c4 by -1.8%, and c8 by
+2.1%. This is performance-equivalent within the intended qualification
threshold while restoring reliable required-field enforcement.

The preliminary run `20260824T232300Z_v0271-ara-betterbench` was deliberately
stopped and remains marked failed: it began before the final Transformers
5.14.1 reproducibility pin was applied, so no partial result is used.

## Compatibility work required by the stable pin

- The opt-in AITER GDN-prefill bridge now supports both the inline v0.27.1
  metadata builder and the post-release helper layout while preserving strict
  source-drift guards.
- The R4D attention wrapper supports v0.27.1's legacy `HND` cache-layout API
  and the post-release `KVCacheLayout.LBHNC` API. Both names describe the same
  head-major, slot-contiguous physical layout required by the R4D kernels.
- The public Compose no longer forces `qwen3.8-enhanced.jinja`; checkpoint-native
  templates are the default and custom templates remain explicit overrides.
