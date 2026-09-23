# Contributing to vllm-radiance

Radiance is experimental, but its production-facing paths are deliberately qualified. Contributions are
welcome even if you do not own the primary dual-R9700 test host.

The maintenance CLI/validator uses PyYAML (`pip install PyYAML` if your environment does not already provide it).

Start with `AGENTS.md` (the policy applies equally well to human and automated maintenance), then use:

```bash
python tools/radiance.py test changed
```

to run the automatically runnable checks selected for your changes. `python tools/radiance.py test <area> --dry-run`
shows the broader area-specific plan.

## Qualification levels

Radiance distinguishes evidence levels rather than pretending every contributor has identical hardware:

1. **CPU-qualified** — deterministic static/unit/contract checks pass.
2. **GPU-smoke-qualified** — a compatible GPU runtime starts and serves the changed path.
3. **R9700-TP2-qualified** — the primary physical topology passes the relevant correctness/smoke gates.
4. **Performance-qualified** — matched benchmark evidence supports the performance claim.
5. **Release-qualified** — the required production-profile and release gates have passed.

A default-off experiment can be useful before it reaches the final levels. A change to a qualified production
default generally needs the physical evidence selected by `.radiance/tests.yaml`.

## External work and attribution

Radiance intentionally tracks other developers and forks. Preserve original authorship when importing commits.
For new imported/adapted work, record provenance in `.radiance/upstreams.yaml` / `.radiance/patches.yaml` as
appropriate and prefer these Git trailers when they apply:

```text
Radiance-Upstream: <repository or developer/fork>
Radiance-Upstream-Commit: <exact upstream commit>
Radiance-Import: cherry-pick|selective-port|conceptual-port
```

Do not rewrite old history simply to add trailers.

## Tests versus benchmarks

`tests`/contract checks answer whether an invariant holds. The benchmark laboratory answers what changed under
a controlled experiment. They share a selection front door but are not interchangeable. Do not replace a
sampled/live correctness gate with an easier deterministic smoke just to get green.

See `docs/maintenance/TESTING_POLICY.md` and `benchmarks/README.md` for the detailed contracts.
