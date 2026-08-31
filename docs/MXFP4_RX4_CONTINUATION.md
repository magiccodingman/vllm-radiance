# RX4 MXFP4 decode continuation

## Scope and provenance

This continuation starts from merged fork `main` at
`10a55730efc281c7548ed7d7a30fa6fa82428715` and selectively ports Brian's RX4
work from
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4).
The audited upstream point is main commit
`b9d7ecf8bed3d4be5e99ad7582ad0d64bb27fc99`; its final authored head is
`5950b383dfe9026df638e8f6bbd69bae968ea71b`. Original authorship is retained in
the four imported Git commits.

The sequence matters:

1. `db9dba680f77` adds traced activation quantization, the FP8 residual stream,
   fused TP2 all-reduce epilogues, shared speculative GDN metadata, and the
   composite small-k sampler.
2. `3d159921edc8` restores new-file hunks that were accidentally omitted when
   the libr4d patch was regenerated. A source build without this correction is
   not an RX4 build.
3. `eafcac951d4f` adds an epilogue prefetch experiment that regressed real
   prefill. It is retained in history for provenance.
4. `5950b383dfe9` reverts that regression, selects 512 threads for large
   prefill epilogues, and adds an inert DFlash selector-width control.

The final tree therefore contains the corrected state, not the briefly shipped
prefetch regression. The upstream launcher is not copied: this fork keeps its
portable Compose, immutable benchmark laboratory, pinned vLLM/libr4d builds,
and deployment-specific environment overlays.

## Integrated operators

- `RADIANCE_NORMQUANT_FUSION=1` atomically hoists MXFP4 activation
  quantization into the traced graph, selects the plain-Torch FP8 quant math,
  and merges vLLM's norm/activation fusion passes into the existing compiler
  JSON.
- `RADIANCE_FP8_STREAM=1` lets adjacent decoder layers exchange an FP8
  activation plus scale while a fused libr4d kernel performs TP2 all-reduce,
  residual add, RMSNorm, and quantization. Unsupported layers retain the stock
  contract.
- `RADIANCE_GDN_SHARED_BUILD=1` builds identical speculative GDN metadata once
  per step and updates only each KV group's block-table-derived state indices.
- `RADIANCE_TOPK_COMPOSITE=1` replaces a serial whole-vocabulary scan with
  multi-block `torch.topk` and a bounded mask when every active top-k is at or
  below `RADIANCE_TOPK_COMPOSITE_KCAP` (64 by default).
- `RADIANCE_DFLASH_SELECTOR_TOPK` can raise DFlash's selector truncation for a
  labeled experiment. Unset or zero preserves the checkpoint and remains the
  default.

The old AR-overlap experiment remains excluded. RX4 composes with this fork's
already-qualified WHT6 large-message all-reduce, GDN merge/fused update,
dynamic verify width, exact rerank/verify head, R4D attention, and safe
M<=64 decode boundary.

## Safety policy

The generic native-FP8 profile is unchanged. Traced quant and FP8 stream are
off by default because they require the Quark MXFP4 W4A8 TP2 geometry. Enabling
FP8 stream without all prerequisites exits with status 64 instead of silently
running an unintended graph.

Radiance environment flags are not included in vLLM's persistent AOT cache
key. The entrypoint therefore namespaces graph caches by GDN merge, traced
quant, FP8 stream, and fast-draft state. It also rejects a separate explicit
`--compilation-config` when the RX4 aggregate profile is active; callers must
provide `RADIANCE_COMPILATION_CONFIG` so the required pass settings can be
merged into the same JSON object.

## GPU-free validation

Before production was touched, the following passed against the exact merged
RX3 production image and pinned sources:

- all three new vLLM patch scripts applied cleanly to vLLM 0.28.0;
- the corrected `r4d_radiance_extras.patch` applied to libr4d commit
  `e8de4bc1f3dbd608dcb8d3ffceb6b48acdf83bb7`;
- the overlay image compiled `r4d.so` and `radiance_mxfp4_fp8.so` for gfx1201;
- `pip check`, native imports, exact version checks, and required RX4 symbol
  checks passed;
- the entrypoint derived both traced-quant flags, merged PIECEWISE plus both
  compiler passes, and selected an isolated
  `radiance-gdnm-nqft-fp8s-fast-draft` cache namespace;
- an invalid FP8-stream profile failed closed with status 64;
- the composite top-k/top-p implementation matched vLLM's full-sort reference
  on 21 CPU cases, including top-k boundary ties;
- the GPU-free speculative XGrammar termination test and shared
  reasoning/tool-parser structural test passed;
- public and benchmark Compose files parsed successfully.

## Live qualification

Live GPU results, exact image digest, immutable run IDs, correctness status,
BetterBench category rows, acceptance, telemetry, and the final deployment
decision are recorded here after the production quiet-window gate.
