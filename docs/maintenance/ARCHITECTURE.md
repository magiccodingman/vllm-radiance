# Radiance maintenance architecture

## Why this exists

Radiance grew from a specialized RDNA4 vLLM derivative into a qualified downstream distribution. The project
now combines an exact compiler/runtime foundation, guarded source overlays, native kernels, external kernel
libraries, selective imports from several upstream developers, and a substantial qualification laboratory.

The runtime architecture is not being replaced by this control plane. The control plane makes the maintenance
knowledge around that runtime explicit and machine-checkable.

## The five layers

### 1. Foundation

The exact vLLM / ROCm / AMD PyTorch / AMD Triton / AITER stack is recorded in `.radiance/stack.yaml`.
The Dockerfile remains the build authority; validation prevents the manifest from silently drifting away from it.

### 2. Overlay

Radiance modifies the pinned foundation through guarded `patch_*.py` scripts, runtime modules, HIP kernels,
and a pinned libr4d build plus `r4d_radiance_extras.patch`. `.radiance/patches.yaml` records why each source
overlay exists, how it is activated, what it affects, its known provenance, and the gates protecting it.

This does **not** imply that every vLLM interaction should be abstracted away. When vLLM provides a genuine
stable extension seam, prefer it. Otherwise an explicit guarded patch is often more honest and maintainable
than another compatibility layer hiding the same coupling.

### 3. Provenance

`.radiance/upstreams.yaml` records the repositories/forks that Radiance builds from or repeatedly audits.
Tracked work can be a build dependency, selective import source, or conceptual watch; those relationships are
intentionally different.

### 4. Qualification

`.radiance/tests.yaml` indexes deterministic checks, candidate-image contracts, physical R9700 gates,
performance experiments, capacity qualification, and maintenance-only probes. The existing benchmark harness
and immutable result history remain authoritative evidence.

### 5. Distribution

The release branch publishes exact-source-built images. Production/default promotion follows
`docs/maintenance/RELEASE_POLICY.md`; experimental availability and production qualification are not synonyms.

## Current policy versus history

Maintenance policy lives in `docs/maintenance/`. Historical reports remain immutable evidence under `docs/`
and `benchmarks/results/`. Agents should use history to understand evidence without having to reverse-engineer
today's policy from old experimental states.

## Control-plane invariants

`tools/validate_control_plane.py` checks the relationships that should never become aspirational prose:

- active Dockerfile-applied overlays must be registered;
- every root `patch_*.py` file must be represented;
- referenced test IDs must exist;
- registry IDs are unique;
- declared stack pins agree with Dockerfile arguments;
- registry paths and policy documents exist.

The goal is deliberately modest: make repository knowledge dependable, not build a new framework that becomes
harder to maintain than Radiance itself.
