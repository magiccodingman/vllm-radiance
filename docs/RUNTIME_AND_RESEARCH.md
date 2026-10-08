# Runtime defaults and optional research


The portable defaults remain non-speculative: `RADIANCE_FAST_DRAFT=0` and no
`RADIANCE_SPECULATIVE_CONFIG`. The GDN extreme-decay correction is part of the
native build and runs only for affected sequence/head pairs. State persisted
under the old scan must be rebuilt. See [numerical corrections](NUMERICAL_CORRECTIONS.md).

`RADIANCE_VERIFY_HEAD=1` is configured but inert unless `RADIANCE_FAST_DRAFT=1`.
When explicitly enabled, TP2 retains the full target head; TP1's global-256
shortlist is approximate and can omit the true winner. Set
`RADIANCE_VERIFY_HEAD=0` for the full head. See [verify-head configuration and
negative results](VERIFY_HEAD_GLOBAL_TOPK.md).

The [2026-10-08 qualification](../benchmarks/results/20261008-pr8-11-production-qualification/README.md)
covers the saved dual-R9700 Quark27B/DFlash-7 profile, not every speculative
configuration. All eight model responses matched an independent corrected FP64
reference; historical-main responses changed because the old scan corrupted state.
Tool/wire gates and capacity through one 131K request passed. Decode throughput
was essentially unchanged; the longest measured prefill workload cost about 8%.
This evidence does not establish speculative/non-speculative cross-mode equivalence
or change the portable non-speculative default.

Optional research lives outside the serving configuration:

| Path | Purpose | Qualification boundary |
|---|---|---|
| [Conformance support](../benchmarks/conformance/README.md) | Numerical/state comparison, replay and evidence contracts | Separate locked CPU environment; no automatic model loading |
| [D7 arithmetic research](../benchmarks/d7-repair/README.md) | Source-bound arithmetic, eager/compiled replay and native probes | Native adapters pinned to vLLM 0.28; fail closed on 0.30 |

The combined research CPU suite passed 428 tests with one explicit retained-artifact
skip. These packages are not installed by the serving build and introduce no Compose
switch or production dependency. Their profile JSON files define research reference
contracts, not serving defaults. Start with `python tools/radiance.py test conformance`
or `python tools/radiance.py test d7-repair`; native work needs matching source/build
receipts and an explicit GPU maintenance run. See [benchmark lab usage](../benchmarks/README.md).

