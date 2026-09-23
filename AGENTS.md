# Radiance agent policy

Radiance is a **qualified downstream inference distribution**, not merely a collection of vLLM patches. Its
primary qualified environment is AMD Radeon AI PRO R9700 / gfx1201, especially dual-GPU TP2. Runtime
changes must preserve the distinction between qualified production defaults and explicit experiments.

## Decision authority

### Tier 1 — policy/evidence determines the action

When current policy and evidence uniquely determine the implementation or maintenance action, perform it,
test it, document it when appropriate, commit it, and continue. Typical examples are mechanical compatibility
updates, obvious wiring defects, source-drift anchor changes with unchanged semantics, and removing a local
overlay after equivalent upstream ownership is demonstrated.

### Tier 2 — investigation is required, but evidence can determine the action

Difficult work is still agent-owned when investigation can establish a uniquely justified action within the
existing Radiance contract. Debug, trace, benchmark, inspect kernels/compiler behavior, compare upstream
implementations, or run hardware experiments as necessary. Once evidence identifies the cause and the repair
or migration is uniquely justified, implement it, validate it, document it, and continue.

### Tier 3 — a genuine decision remains

Escalate when materially different valid choices remain or proceeding would change architecture, supported
behavior, correctness standards, production defaults, upstream ownership boundaries, or create a substantial
new long-term maintenance obligation. Do not YOLO an architectural choice.

A Tier-3 handoff must contain:

- observed facts;
- what has been ruled out;
- remaining viable explanations/options;
- consequences and tradeoffs of each;
- the smallest useful next discriminator, if one exists;
- the exact human decision required.

**Do not involve the human merely because work is difficult, lengthy, unfamiliar, or experimental. Involve
the human when unresolved judgment actually requires a human decision.**

## Non-negotiable maintenance rules

- Prefer exact reproducible pins and preserve the qualified compiler/runtime stack.
- Read release semantics before performing a dependency upgrade; do not reduce an upgrade to “make the diff compile.”
- Prefer upstream ownership once equivalent upstream behavior is present and qualified.
- Never silently lose a Radiance correctness fix or qualified optimization during an upgrade.
- Preserve negative results and failed gates as evidence.
- Do not weaken, skip, or redefine a required qualification gate merely to make a candidate pass.
- Performance claims require matched controls.
- Hardware qualification must only be claimed when it actually ran.
- Use real upstream extension seams when they exist; do not invent abstraction layers that merely hide unavoidable vLLM coupling.

## Where to look

- Dependency/upstream upgrades: `docs/maintenance/UPGRADE_POLICY.md`
- External imports/provenance: `docs/maintenance/IMPORT_POLICY.md` and `.radiance/upstreams.yaml`
- Active overlays: `.radiance/patches.yaml`
- Test selection/qualification: `docs/maintenance/TESTING_POLICY.md` and `.radiance/tests.yaml`
- Production/default promotion: `docs/maintenance/RELEASE_POLICY.md`
- Maintenance architecture: `docs/maintenance/ARCHITECTURE.md`
- Exact stack identity: `.radiance/stack.yaml`
- Historical evidence: `docs/`, `benchmarks/results/`, and immutable benchmark manifests

## Canonical commands

```bash
python tools/radiance.py test pr
python tools/radiance.py test changed
python tools/radiance.py test <area>
python tools/radiance.py qualify upgrade
python tools/radiance.py qualify release
python tools/validate_control_plane.py
```

Qualification commands print the complete plan. Use `--run-auto` only for the automatically runnable CPU
subset; physical-GPU and maintenance-window gates remain explicit.
