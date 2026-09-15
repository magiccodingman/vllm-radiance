# Testing and qualification policy

Radiance already has a mature benchmark laboratory. This policy gives contributors and agents a coherent
front door without collapsing fundamentally different evidence into one bucket.

## Registry and commands

`.radiance/tests.yaml` catalogs checks with:

- stable IDs;
- canonical commands;
- test class;
- hardware/environment requirements;
- affected subsystem tags;
- CI/automatic eligibility;
- qualification profiles.

Use:

```bash
python tools/radiance.py test pr
python tools/radiance.py test changed
python tools/radiance.py test <area>
python tools/radiance.py qualify upgrade
python tools/radiance.py qualify release
```

`test` executes only entries explicitly marked automatic unless `--dry-run` is used for planning.
`qualify` prints the entire qualification plan by default; `--run-auto` executes the safe automatic subset
while leaving candidate-image, physical-GPU, and maintenance-window gates explicit.

## Evidence classes

- **Static/unit/contract** — deterministic source or pure-Python invariants.
- **Candidate-image** — GPU-free logic that still needs the exact installed vLLM/Transformers/runtime image.
- **Live correctness** — real server behavior, including sampled regressions that cannot be replaced by easier
  deterministic smokes.
- **Performance** — matched controlled experiments with immutable manifests/telemetry.
- **Capacity** — explicit full-wave or long-context pressure tests.
- **Maintenance-only** — probes that require declared downtime or special safety conditions.

The class is part of the meaning of the evidence. A benchmark is not a unit test, and a unit test does not
substitute for an end-to-end sampled/live gate.

## Change-to-test selection

`change_map` maps repository paths into areas. `test changed` discovers changed files from Git, maps them to
areas, and runs the automatically runnable tests protecting those areas plus the baseline PR invariants.

Patch entries also reference protecting test IDs. Agents may add stronger validation beyond this minimum.
They may **not** silently remove, weaken, skip, or redefine a required gate just to make a candidate pass.
Changing a meaningful correctness/qualification standard is normally Tier 3.

## Hardware scarcity

The dual-R9700 host is a laboratory resource, not an assumption about every contributor's workstation. MR CI
therefore emphasizes deterministic CPU checks. When a change needs R9700 evidence, the MR should say
“CPU-qualified; R9700-TP2 qualification required before production promotion” rather than either blocking all
contributions or pretending the hardware test happened.

## Negative results

Keep failed qualification runs and control results when they are informative. Do not publish only aggregate
wins. Inspect CV/variance, acceptance, TTFT/ITL/TPOT, correctness, and the workload-specific metrics relevant
to the change. Existing `benchmarks/README.md` remains the detailed experimental contract.
