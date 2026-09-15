# Radiance maintenance metadata

These files are the machine-readable side of the maintenance control plane:

- `stack.yaml` — exact qualified foundation and atomic compatibility groups;
- `patches.yaml` — source-overlay ownership, activation, provenance, runner scope, and protecting gates;
- `upstreams.yaml` — repositories/forks Radiance builds from or repeatedly audits;
- `tests.yaml` — test/qualification catalog, hardware levels, profiles, and change-to-area mapping.

Human policy lives in `docs/maintenance/`. Historical qualification evidence remains in the existing `docs/`
and `benchmarks/results/` trees.

Run `python tools/validate_control_plane.py` after changing these files. The validator deliberately checks
the metadata against the actual Dockerfile patch loop, root patch files, stack pins, and referenced test IDs
so the registry cannot quietly become aspirational documentation.
