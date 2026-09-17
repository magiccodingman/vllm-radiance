# Dependency and upstream upgrade policy

Radiance follows stable upstream releases where practical, pins exact revisions, and selectively carries
reviewed fixes or optimizations when the qualified stack needs them. An upgrade is a migration/qualification
event, not a version-number edit.

## Required sequence

1. **Resolve exact old and target identities.** Record release/tag and immutable commit.
2. **Read release semantics first.** Read release notes, migration notes, deprecations, breaking changes, and
   directly relevant upstream PRs before editing the Radiance tree.
3. **Build an expected-impact map.** Identify affected runners, schedulers, speculative paths, graph capture,
   quantization dispatch, GDN/recurrent state, parsers, KV/cache systems, collectives, and compiler contracts.
4. **Audit every relevant local overlay.** Classify each as:
   - upstream-owned/equivalent now;
   - still required unchanged;
   - mechanical source drift with unchanged semantics;
   - semantic migration required;
   - obsolete;
   - deprecated-architecture debt;
   - genuinely ambiguous / Tier 3.
5. **Prefer upstream ownership.** Remove redundant local code once equivalence is demonstrated. Do not retain a
   patch merely because it still applies.
6. **Preserve Radiance behavior deliberately.** Upgrading may not silently discard a correctness fix, ROCm
   behavior, supported production capability, or qualified optimization.
7. **Reconcile the stack/build.** Treat the ROCm/PyTorch/Triton/AITER/vLLM relationship as a compatibility stack
   where current evidence requires it.
8. **Select validation from affected areas.** Use `.radiance/tests.yaml` / `python tools/radiance.py test changed`.
9. **Run candidate-image and physical gates when selected.** Do not claim them from source inspection.
10. **Use matched performance controls.** Preserve variance and negative results rather than publishing only wins.
11. **Write a qualification report.** Record exact pins, overlay decisions, failed experiments, test evidence,
    performance/correctness results, and the final deployment decision.
12. **Update the control plane.** Change `stack.yaml`, patch ownership/status, upstream audit refs, and tests when
    the upgrade changes reality.

The governing order is:

> **release semantics → expected impact map → source diff/migration**

not:

> **diff → make it compile**

## Model-runner and architecture migrations

When upstream changes a default execution architecture, treat runner ownership as a first-class migration
dimension. `.radiance/patches.yaml` has a `runner` field specifically so an upgrade can inventory overlays as
`v1`, `v2`, `both`, `independent`, or `unknown`.

An upgrade agent should explicitly determine whether critical paths remain on an old fallback, moved to the
new runner, or need migration. A deprecation/removal deadline turns remaining old-runner entries into visible
technical debt rather than a future surprise.

## Escalation

A source-drift repair is not Tier 3 merely because it is tedious. A deep kernel or compiler investigation is
not Tier 3 merely because it is difficult. Escalate when evidence leaves multiple materially different valid
choices or the upgrade would change the project contract.

## Evidence to preserve

Never erase a failed candidate from the experimental story merely because a later repair passed. Radiance's
v0.28 qualification is the reference example: it preserved the failing tool-schema candidate, fixed the
actual regression, reran the live gate, retained performance caveats, and removed an obsolete DFlash source
backport once vLLM owned that implementation.
