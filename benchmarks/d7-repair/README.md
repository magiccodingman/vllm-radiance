# Pinned D7 arithmetic and eager/compiled rounding repairs

These are the source-bound workers, kernels and adapters used for the completed
[compiled M1/M8 and eager/compiled experiments](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).
This is an experimental qualification path. It does not change the container's
default inference configuration or automatically accept a different source build.

```text
d7-repair/
├── stock_gdn_*                # convolution, recurrence, causal prefill and runtime binding
├── stock_m1_*                 # serial arithmetic for normalization, attention and BF16 head
├── build_*                    # CPU-side HIP compilation and source/binary receipts
├── probe_*                    # small native correctness/state/graph/performance gates
├── optimized_d7_*             # admitted optimized paths and pre-capture installation
├── benchmark_*                # forced-token replay and separate unprofiled speed controls
├── execution_mode_d7_worker.py # controlled eager/compiled replay using the same repairs
├── rotary_mode_d7_worker.py   # pinned native RoPE nearest-even product intervention
├── isolated_d7_capture.py     # actual native boundaries, checked against release outputs
├── native_d7_*                # native call tape and isolated historical/stateful stage adapters
├── analyze_native_d7_stages.py # complete-layer and release-output evidence audit
├── compare_*                 # CPU-only admission and comparison of retained evidence
├── trace_private_d7_rows.py   # owner-only aligned logit capture
├── tests/                    # CPU adapter, admission and replay regressions
├── configs/profiles/          # pinned finite-precision reference contract
└── SOURCE_PROVENANCE.json
```

Apply the companion conformance-support PR first. Its `qwen_r9700_lab` package
provides the shared evidence, state and process contracts.

```sh
uv run --project benchmarks/conformance --extra cpu-tests \
  pytest benchmarks/d7-repair/tests -q
```

One compiler-identity test additionally requires the preserved, hashed reference
HSACO and metadata; set `RADIANCE_GDN_REFERENCE_ARTIFACTS` to that artifact directory.
A skip is not native qualification. The other CPU tests use synthetic data and do
not access a GPU. No private Pi fixture or generated text is included here.

Native execution order:

1. Select the exact backend/model contract and source binding. Supply your own
   owner-only token fixture, native specification and isolated output directory.
2. Use `build_stock_m1_norm.py`, `build_stock_m1_head_pair.py`,
   `build_stock_m1_attention_shared.py` and `build_packed_gdn_transport.py` to create
   authenticated builds. Their inputs are explicit source/build directories;
   their outputs are shared libraries, compiler logs and JSON build receipts.
3. Run the matching `probe_*` entry points under the shared exclusive GPU lease.
   These compare outputs and persistent state, test graph replay and injected
   faults, and record the qualified build identities. Consult each script's
   `--help` for its admitted geometry and required native spec.
4. `qualify_stock_gdn_model.py` binds the numerical repair. The optimized workers
   bind the stage receipts before tracing/capture; unsupported shapes retain the
   repaired fallback. Never reseal a changed build as if old evidence covered it.
5. `benchmark_compiled_d7_corpus.py` consumes a sealed corpus and repair/performance
   manifests, checkpoints each continuation, audits actual compiled graph use and
   produces separate private aligned rows and public aggregate comparisons.
   `benchmark_optimized_d7.py` measures speed separately without correctness tracing.
6. `benchmark_crossmode_d7_corpus.py run --revision before|final` measures eager M1
   versus compiled M8 on that same sealed corpus. Run both revisions with distinct
   output/private directories. The final eager reference includes the RoPE repair;
   final compiled M8 preserves BF16 intermediates. Each continuation is checkpointed.
   `audit_crossmode_d7_corpus.py` authenticates all four runs, checks execution modes,
   repair identities, graph replays and aligned positions, then recomputes the public
   before/after counts. Raw tokens and ranked IDs stay in private tmpfs. The
   benchmark retains the shared GPU lease and explicit `--allow-gpu` requirement;
   the auditor runs on the CPU.

There are two major repairs:

1. **M1/M8 arithmetic alignment.** The optimized compiled pair matched all
   10,000 decode positions and 23 initial-prefill predictions, including full
   vocabulary vectors and top-1/10/20 membership, ordering and scores.
2. **Eager/compiled rounding alignment.** Preserve intermediate casts with
   `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` before starting a compiled worker.
   For the pinned eager control, `benchmark_rotary_contract_d7.py` installs the
   source-checked BF16 nearest-even RoPE product correction before model load.
   The pair matched all 320 decode vectors and the prefill prediction. This
   smaller result is separate from Fix 1's 10K qualification.

The combined repair also matches **eager M1 versus compiled M8** at all **10,000
decode positions and 23 initial-prefill predictions**, including top-1/10/20
sets, ranking, retained scores, boundary ties and full-vocabulary hashes. The
report compares original and final pairs with fresh eager references on the same
23-continuation Pi corpus. This is a forced, all-seven-accepted replay on the
pinned backend, not an arbitrary-input or independent mathematical proof.

The RoPE intervention addresses the installed compiler's BF16 multiply lowering;
the corresponding [Triton repair is already merged](https://github.com/triton-lang/triton/pull/11227).
Use the pinned intervention only with its admitted source identity. A newer
compiler requires fresh qualification, not another copy of the workaround.

`compare_execution_modes_d7.py` checks source/configuration identities before
comparing retained rows. `compare_rotary_intervention_d7.py` admits the explicit
rounding change. Boundary capture is an untimed diagnostic; it must reproduce
the graph-enabled release outputs, and its timings are never release timings.
The public report supplies the stage timings, evidence and qualification scope.

`benchmark_native_catalog_d7.py` records the actual generated callable identities
and argument layouts. `benchmark_native_tape_d7.py` captures the current compiled
calls and private state, checks its full output against the ordinary forward,
then substitutes one stage/layer at a time. `analyze_native_d7_stages.py` requires
complete layer and position coverage, state restoration, negative controls and
an exact bridge to the graph-enabled release before exporting aggregate counts.
Equal local outputs/state may reuse the validated reference result; unequal
ones run through the actual remaining model and full vocabulary head.

Set `QWEN_D7_TAPE_GROUPS=40` for the 320-position isolated sweep. The default
captures one eight-position pilot group; `--tokens 320` controls the forced
continuation length, not the number of instrumented groups. The aggregate audit
rejects a run unless all 40 groups and 320 positions are present.

The isolated matrix has four comparisons: original compiled M1/M8; Fix 1
compiled M1/M8; Fix 1 compiled/eager M8; final compiled/eager M8. GDN/attention
adapters preserve private cache state between substitutions. The original fused
Q/K-normalization/rotation boundary is treated as a composite, and attention
decode/merge share one correctness boundary despite separate GPU timings.
The updated combined support/repair CPU suite passes **428 tests**, with one
retained-compiler-artifact check skipped.

The code intentionally rejects source or geometry drift. Porting these adapters
to newer dependencies needs fresh qualification; the published result is tied
to the pinned Radiance 1.0.16 / vLLM 0.28 environment. This PR provides the complete
experimental integration for review, while small core changes are submitted to
vLLM and libr4d separately.
