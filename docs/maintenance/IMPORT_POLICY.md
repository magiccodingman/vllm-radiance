# External upstream and import policy

Radiance intentionally learns from and tracks multiple upstream developers/forks. This is a strength, but the
relationship must stay auditable as the watch list grows.

## Ledger

`.radiance/upstreams.yaml` records meaningful tracked repositories and whether Radiance:

- builds the source directly;
- selectively imports/cherry-picks it;
- adapts concepts to a different base;
- watches it for future work.

`last_audited_ref` is the checkpoint for the next audit. When an exact commit can be established, record it.
When the historical evidence only establishes a report/release checkpoint, say that instead of inventing a SHA.

## Audit workflow

For each tracked upstream, inspect changes since the last audited point and classify useful work as:

- **ignore** — unrelated or incompatible;
- **research-only** — informative, but not an import candidate;
- **direct-import** — suitable with preserved authorship;
- **selective-adaptation** — useful implementation, but Radiance needs compatibility changes;
- **conceptual-port** — import the idea, not the code;
- **supersedes-local** — upstream now owns behavior Radiance should stop maintaining;
- **tier3-decision** — materially different valid strategies remain.

Do not assume a result transfers across a different vLLM base, GPU topology, compiler stack, or benchmark
contract. Attractive upstream numbers are hypotheses until Radiance's relevant gates pass.

## Attribution

Preserve original authorship for imported commits. For future imports/adaptations, use Git trailers when useful:

```text
Radiance-Upstream: <repository/fork>
Radiance-Upstream-Commit: <exact commit>
Radiance-Import: cherry-pick|selective-port|conceptual-port
```

The patch registry may also record upstream PRs/commits when an overlay is a focused backport. Unknown
provenance should be recorded as unknown or lineage-only rather than guessed.

## Dependency boundary

Tracking a repository does not mean turning it into a submodule or permanent runtime dependency. libr4d and
AITER already have explicit source-build boundaries. Other watched forks can remain audit sources until a real
dependency is justified.
