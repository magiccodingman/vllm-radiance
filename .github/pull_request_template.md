## Problem and fix

Explain the concrete root cause, resulting behavior and why this fix is justified.
Link the issue if one exists (Refs #...).

## Regression coverage and verification

Name the regression test, the commands actually executed and their results.
If a regression test cannot be added, explain why and provide the alternative evidence.
Record the commit tested and preserve relevant failures as well as successful results.

## Qualification still required

Consult AGENTS.md and .radiance/tests.yaml for the affected areas. List the specific
candidate-image, GPU, R9700 TP2, correctness, performance or release gates still
required, and mark checks that did not run as NOT RUN. Do not infer hardware
qualification from a passing CPU job.

## Maintainer decision

Describe any remaining Tier-3 decision using the AGENTS.md handoff requirements.
Changes to qualified production defaults require their existing qualification.
