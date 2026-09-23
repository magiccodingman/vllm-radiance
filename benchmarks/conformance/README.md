# Numerical and state conformance support

This is the portable support used in the
[M1/M8 and eager/compiled investigation](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).
It compares implementations against a declared reference; passing finite samples
does not prove arbitrary-input equivalence or model task accuracy.

```text
conformance/
├── src/qwen_r9700_lab/   # preserved module names and source identities
├── tests/               # synthetic CPU regressions and negative controls
├── configs/profiles/    # explicit serial GDN arithmetic contract
├── SOURCE_PROVENANCE.json
├── pyproject.toml
└── uv.lock
```

Install and run the CPU suite without loading a model:

```sh
uv sync --project benchmarks/conformance --group dev
uv run --project benchmarks/conformance pytest benchmarks/conformance/tests -q
```

The top-k comparator consumes aligned full-logit summaries and reports top-1,
top-10 and top-20 set agreement, order agreement, overlap, retained-score equality,
boundary ties and full-row digests separately. Empty domains, mismatched positions,
changed manifests and incomplete evidence are errors. It does not decode tokens.

The other modules provide logical-state comparisons, transition invariants,
tentative-output checking, fault injection, source/binary binding, exclusive GPU
leases and checkpointed worker supervision. Abstract proofs and CPU simulations
remain distinct from native GPU qualification. Private raw rows and public
aggregate receipts have different storage rules.

For native replay, the companion `benchmarks/d7-repair` submission supplies the
actual source-bound workers and numerical repair adapters. Build and qualification
receipts must match before those adapters can be installed. Production defaults
and container builds are unaffected by this support package.

The Python namespace is retained to preserve compatibility with existing sealed
workers. Renaming and narrowing this dependency closure can be reviewed separately
from the initial import. The standalone CPU suite is the publication-time check;
the report describes the separate completed pinned-stack GPU campaign.

The companion D7 repair bundle also supplies the source-preserving native call
tape and four-column isolated-stage study. Its evidence auditor rejects missing
layers, changed fixtures, unchecked reference output and failed state recovery.
The combined support/repair CPU suite passes **412 tests**, with one retained
compiler-artifact check skipped. Native qualification is reported separately in
the [public report](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).
