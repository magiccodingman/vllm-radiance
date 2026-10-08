"""Executable control/decision formulae used by the conformance checker.

These small expressions are also called with symbolic arguments by Z3. Their
proofs cover these functions, not a GPU implementation that merely resembles
them. Numeric certificates are conditional on independently justified bounds.
"""

from fractions import Fraction

from qwen_r9700_lab.conformance_gate import publication_allowed
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, integer, seal


def transaction_ready(output, state, identity, coverage, completed, cancelled, base, current):
    return (
        publication_allowed(output, state, identity, coverage)
        & completed
        & (cancelled == False)  # noqa: E712 -- works on Python and symbolic Booleans
        & (base == current)
    )


def version_matches(position, kv, gdn, conv):
    return (position == kv) & (position == gdn) & (position == conv)


def writable_exclusively(references, external_references, immutable):
    return (
        (references == 1) & (external_references == 0) & (immutable == False)  # noqa: E712
    )


def restore_matches(
    contract, restored_contract, generation, restored_generation, size, restored_size
):
    return (
        (contract == restored_contract)
        & (generation == restored_generation)
        & (size == restored_size)
    )


def snapshot_publishable(verified, durable, current, candidate):
    return verified & durable & (current == candidate)


def separated_intervals(winner_lower, competitor_upper):
    return winner_lower > competitor_upper


def unresolved_frontiers_separated(
    cutoff, local_upper, global_upper, local_unresolved, global_unresolved
):
    """Compare already conservative bounds for both unrescored partitions.

    Local means dropped within a vocabulary block; global means emitted by
    that block but omitted from exact reranking. Empty partitions are ignored.
    This predicate does not establish coverage, score identity or bounds.
    No rounding-sensitive arithmetic is performed here. Native bound
    construction, cutoff selection and invocation binding remain unproved.
    """
    return (
        (local_unresolved == False) | (cutoff > local_upper)  # noqa: E712
    ) & (
        (global_unresolved == False) | (cutoff > global_upper)  # noqa: E712
    )


def margin_preserved(margin, error):
    return (error >= 0) & (margin > 2 * error)


def interval_certificate(lower, upper, winner: int, *, bound_origin: str) -> dict:
    """Exact rational inequalities; never turn empirical errors into a proof.

    All vocabulary entries must be present. Missing/excluded logits cannot be
    assumed harmless merely because an INT2 shortlist was exactly reranked.
    The caller supplies sound intervals including upstream state/hidden error.
    """
    integer(winner)
    if not lower or len(lower) != len(upper) or winner >= len(lower) or not bound_origin:
        raise DiagnosticError("incomplete full-vocabulary interval certificate")
    lo, hi = [Fraction(v) for v in lower], [Fraction(v) for v in upper]
    if any(a > b for a, b in zip(lo, hi, strict=True)):
        raise DiagnosticError("reversed logit interval")
    certified = all(separated_intervals(lo[winner], hi[i]) for i in range(len(lo)) if i != winner)
    return seal(
        {
            "schema": "urn:qwen:conditional-argmax-certificate:v1",
            "vocabulary": len(lo),
            "winner": winner,
            "certified_under_bounds": certified,
            "lower": [str(v) for v in lo],
            "upper": [str(v) for v in hi],
            "bound_origin": bound_origin,
            "bounds_soundness": "ASSUMED",
            "status": "RUNTIME-CHECKED",
            "backend_equivalence": "UNPROVED",
        }
    )


def rejection_distribution(target, draft):
    """Exact finite-distribution reference for one speculative rejection step.

    P(token) = min(p,q) + Z * max(p-q,0)/Z = p. Z=0 means
    all proposals are accepted; no residual distribution is sampled.
    Does not specify RNG consumption or imply equal seeded sample paths.
    """
    p, q = [Fraction(v) for v in target], [Fraction(v) for v in draft]
    if not p or len(p) != len(q) or sum(p) != 1 or sum(q) != 1 or any(v < 0 for v in p + q):
        raise DiagnosticError("invalid complete target/draft probability distribution")
    accepted = [min(a, b) for a, b in zip(p, q, strict=True)]
    residual = [a - b for a, b in zip(p, accepted, strict=True)]
    rejected_mass = sum(residual)
    conditional = [v / rejected_mass for v in residual] if rejected_mass else None
    final = [a + r for a, r in zip(accepted, residual, strict=True)]
    return {
        "accepted_mass": accepted,
        "rejected_mass": rejected_mass,
        "residual": conditional,
        "output": final,
    }
