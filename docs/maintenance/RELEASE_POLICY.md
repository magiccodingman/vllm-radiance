# Release and default-promotion policy

Radiance separates “code exists” from “production default is qualified.”

## Promotion principles

A change may be merged as an explicit/default-off experiment with less evidence than is required to change a
qualified production default, provided its isolation is real and documented.

Changing a production default requires the qualification selected for the affected areas and profile. In
particular:

- correctness gates may not be weakened to justify promotion;
- a performance default needs matched performance evidence;
- hardware-specific claims need the hardware run;
- the production profile must retain documented VRAM/headroom and supported behavior;
- experimental flags with known correctness failures remain explicit experiments.

## Evidence levels

The repository uses these descriptive states:

- `CPU-qualified`
- `GPU-smoke-qualified`
- `R9700-TP2-qualified`
- `performance-qualified`
- `release-qualified`

State the highest level actually demonstrated; do not imply the later levels.

## Release qualification

`python tools/radiance.py qualify release` prints the current registry-defined release plan. It is a planner, not an
automatic license to run maintenance-window or physical-GPU tests. Run and retain the required evidence using
the benchmark harness, then update the release/qualification report.

The `release` branch remains the Docker publication path. This control-plane work does not change the policy
that stable vLLM releases and exact source pins form the production foundation.
