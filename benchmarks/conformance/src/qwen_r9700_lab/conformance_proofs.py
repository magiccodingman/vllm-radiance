"""Small, scoped SMT obligations over actual prototype helper expressions.

This does not prove Radiance, the checker implementation, Python or GPU code.
The solver, Python/Z3 symbolic-expression translation and machine are trusted.
Every counterexample query, implementation hash and result is preserved.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from qwen_r9700_lab.conformance_gate import copy_accepted_prefix, publication_allowed
from qwen_r9700_lab.diagnostic_contract import integer, remaining_materialized, seal, write_private


def check_obligation(
    output_root, name, counterexample, *, expected, assumptions=(), timeout_ms=10_000
):
    """Preserve the query and witness; inconsistent domains never prove a claim."""
    import z3

    integer(timeout_ms, minimum=1)
    domain = z3.Solver()
    domain.set(timeout=timeout_ms)
    domain.add(*assumptions)
    domain_query = domain.to_smt2()
    (output_root / (name + ".domain.smt2")).write_text(domain_query)
    domain_result = domain.check()
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*assumptions, counterexample)
    query = solver.to_smt2()
    (output_root / (name + ".smt2")).write_text(query)
    actual = solver.check()
    row = {
        "name": name,
        "expected": str(expected),
        "result": str(actual),
        "domain_result": str(domain_result),
        "domain_sha256": hashlib.sha256(domain_query.encode()).hexdigest(),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "status": ("PROVED" if expected == z3.unsat else "TESTED")
        if actual == expected and domain_result == z3.sat
        else "UNPROVED",
    }
    if domain_result != z3.sat:
        row["domain_error"] = "inconsistent or unproved preconditions"
    if actual == z3.unknown:
        row["unknown_reason"] = solver.reason_unknown()
    if actual == z3.sat:
        witness = solver.model().sexpr()
        (output_root / (name + ".counterexample.smt2")).write_text(witness + "\n")
        row["counterexample_sha256"] = hashlib.sha256((witness + "\n").encode()).hexdigest()
    return row


def run_obligations(output_root: Path, timeout_ms: int = 10_000):
    import z3

    from qwen_r9700_lab.conformance_invariants import (
        margin_preserved,
        restore_matches,
        separated_intervals,
        snapshot_publishable,
        transaction_ready,
        unresolved_frontiers_separated,
        version_matches,
        writable_exclusively,
    )
    from qwen_r9700_lab.conformance_radiance import convolution_offset, temporal_column
    from qwen_r9700_lab.conformance_reference import (
        bf16_round_bits,
        canonical_slot,
        high_nibble,
        low_nibble,
    )

    integer(timeout_ms, minimum=1)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []

    def check(name, counterexample, *, expected=z3.unsat, assumptions=()):
        rows.append(
            check_obligation(
                output_root,
                name,
                counterexample,
                expected=expected,
                assumptions=assumptions,
                timeout_ms=timeout_ms,
            )
        )

    out, state, identity, complete = z3.Bools("output_equal state_equal identity_equal complete")
    # The implementation function executes on symbolic values: its Boolean
    # expression is not manually copied into the verification condition.
    allowed = publication_allowed(out, state, identity, complete)
    check("gate_requires_state", z3.And(allowed, z3.Not(state)))
    check("gate_requires_output", z3.And(allowed, z3.Not(out)))
    check("gate_requires_identity", z3.And(allowed, z3.Not(identity)))
    check("gate_requires_coverage", z3.And(allowed, z3.Not(complete)))
    check("mutant_output_only_gate", z3.And(out, z3.Not(state)), expected=z3.sat)

    current = tuple(z3.BitVec(f"old_{i}", 32) for i in range(8))
    proposed = tuple(z3.BitVec(f"candidate_{i}", 32) for i in range(8))
    alternative = tuple(z3.BitVec(f"perturbed_{i}", 32) for i in range(8))
    for count in range(9):
        committed = copy_accepted_prefix(current, proposed, count)
        changed = copy_accepted_prefix(current, alternative, count)
        check(
            f"prefix_selection_{count}",
            z3.Or(*[committed[i] != (proposed[i] if i < count else current[i]) for i in range(8)]),
        )
        check(
            f"rejected_suffix_noninterference_{count}",
            z3.Or(*[committed[i] != changed[i] for i in range(8)]),
            assumptions=[proposed[i] == alternative[i] for i in range(count)],
        )
    check(
        "mutant_commit_all",
        z3.Or(*[proposed[i] != current[i] for i in range(1, 8)]),
        expected=z3.sat,
    )

    before, old_pending, emitted, new_pending = z3.Ints("before old_pending emitted new_pending")
    after = remaining_materialized(before, old_pending, emitted, new_pending)
    check("pending_token_conservation", after + new_pending != before + old_pending + emitted)
    check(
        "materialized_nonnegative",
        after < 0,
        assumptions=[
            before >= 0,
            old_pending >= 0,
            old_pending <= 1,
            emitted >= 1,
            new_pending >= 0,
            new_pending <= 1,
        ],
    )

    fp = z3.Float32()
    large, one, negative = (z3.FPVal(v, fp) for v in (16777216, 1, -16777216))
    rnd = z3.RNE()
    left = z3.fpAdd(rnd, z3.fpAdd(rnd, large, one), negative)
    right = z3.fpAdd(rnd, large, z3.fpAdd(rnd, one, negative))
    check("float32_reassociation_counterexample", z3.Not(z3.fpEQ(left, right)), expected=z3.sat)

    # Execute the actual NumPy bit expression symbolically. The independent
    # oracle splits retained/discarded fields and applies the RNE rule. This
    # proves the helper on its finite/Inf input domain, not GPU cast instructions.
    bits = z3.BitVec("float32_bits", 32)
    rounded = bf16_round_bits(bits)
    high, low = z3.Extract(31, 16, bits), z3.Extract(15, 0, bits)
    finite_or_inf = z3.Or((bits & 0x7F800000) != 0x7F800000, (bits & 0x007FFFFF) == 0)
    increment = z3.If(
        z3.Or(z3.UGT(low, 0x8000), z3.And(low == 0x8000, (high & 1) == 1)),
        z3.BitVecVal(1, 16),
        z3.BitVecVal(0, 16),
    )
    independent = high + increment
    check(
        "bf16_rounding_matches_rne",
        rounded != z3.Concat(independent, z3.BitVecVal(0, 16)),
        assumptions=[finite_or_inf],
    )
    check(
        "bf16_rounding_clears_discarded_bits", (rounded & 0xFFFF) != 0, assumptions=[finite_or_inf]
    )
    check(
        "bf16_representable_values_unchanged",
        rounded != bits,
        assumptions=[finite_or_inf, low == 0],
    )
    check(
        "bf16_halfway_ties_even",
        (rounded & 0x10000) != 0,
        assumptions=[finite_or_inf, low == 0x8000],
    )
    check(
        "mutant_bf16_truncation",
        (bits & 0xFFFF0000) != rounded,
        expected=z3.sat,
        assumptions=[finite_or_inf],
    )

    packed = z3.BitVec("packed_byte", 8)
    check("mxfp4_nibble_roundtrip", ((high_nibble(packed) << 4) | low_nibble(packed)) != packed)
    check("mxfp4_low_nibble_unsigned_range", z3.UGT(low_nibble(packed), 15))
    check("mxfp4_high_nibble_unsigned_range", z3.UGT(high_nibble(packed), 15))
    block, offset, size = z3.Ints("logical_block logical_offset logical_block_size")
    slot = canonical_slot(block, offset, size)
    domain = [block >= 0, size > 0, offset >= 0, offset < size]
    check("canonical_slot_inverse_block", slot / size != block, assumptions=domain)
    check("canonical_slot_inverse_offset", slot % size != offset, assumptions=domain)
    running, accepted = z3.Ints("running_state_column accepted_count")
    domain = [running >= 0, accepted >= 1, accepted <= 8]
    check(
        "accepted_state_column",
        temporal_column(running, accepted) - running != convolution_offset(accepted),
        assumptions=domain,
    )
    check(
        "accepted_conv_window_range",
        z3.Or(convolution_offset(accepted) < 0, convolution_offset(accepted) > 7),
        assumptions=domain,
    )
    check("mutant_unshifted_conv_window", accepted > 1, expected=z3.sat, assumptions=domain)

    completed, cancelled, verified, durable = z3.Bools("completed cancelled verified durable")
    base, epoch = z3.Ints("base_revision current_revision")
    ready = transaction_ready(out, state, identity, complete, completed, cancelled, base, epoch)
    for name, violation in (
        ("transaction_requires_completed_writes", z3.Not(completed)),
        ("transaction_rejects_cancelled_work", cancelled),
        ("transaction_rejects_stale_revision", base != epoch),
        ("transaction_requires_full_equality", z3.Not(z3.And(out, state, identity, complete))),
    ):
        check(name, z3.And(ready, violation))
    check("mutant_unfenced_publication", z3.And(allowed, z3.Not(completed)), expected=z3.sat)
    position, kv, gdn, conv = z3.Ints("position kv_position gdn_position conv_position")
    check(
        "all_state_versions_match",
        z3.And(
            version_matches(position, kv, gdn, conv),
            z3.Or(kv != position, gdn != position, conv != position),
        ),
    )
    check("mutant_kv_only_version", z3.And(kv == position, gdn != position), expected=z3.sat)
    refs, pins = z3.Ints("owner_references external_pins")
    immutable = z3.Bool("immutable")
    write = writable_exclusively(refs, pins, immutable)
    check(
        "shared_or_pinned_state_is_not_writable",
        z3.And(write, z3.Or(refs != 1, pins != 0, immutable)),
    )
    publish = snapshot_publishable(verified, durable, epoch, base)
    check(
        "snapshot_requires_verified_durable_current",
        z3.And(publish, z3.Or(z3.Not(verified), z3.Not(durable), epoch != base)),
    )
    check("mutant_publish_before_durability", z3.And(verified, z3.Not(durable)), expected=z3.sat)
    ca, cb, ga, gb, sa, sb = z3.Ints(
        "contract_a contract_b generation_a generation_b size_a size_b"
    )
    check(
        "restore_requires_same_contract_generation_position",
        z3.And(restore_matches(ca, cb, ga, gb, sa, sb), z3.Or(ca != cb, ga != gb, sa != sb)),
    )

    # Conditional *real/rational* interval theorem. It does not infer these
    # bounds from samples, and does not certify any GPU transcendental/kernel.
    a, b, da, db, error = z3.Reals("winner runner_up winner_error runner_error certified_bound")
    check(
        "argmax_margin_bound",
        a + da <= b + db,
        assumptions=[
            margin_preserved(a - b, error),
            da >= -error,
            da <= error,
            db >= -error,
            db <= error,
        ],
    )
    lower, upper, actual_a, actual_b = z3.Reals(
        "winner_lower other_upper actual_winner actual_other"
    )
    check(
        "full_vocabulary_interval_winner",
        actual_a <= actual_b,
        assumptions=[
            separated_intervals(lower, upper),
            actual_a >= lower,
            actual_b <= upper,
        ],
    )
    check("mutant_exact_shortlist_omits_winner", z3.And(a > b, actual_b > a), expected=z3.sat)

    # An arbitrary unresolved token must belong to one of the two covered
    # partitions. This proves the comparison predicate under sound bounds;
    # it does not derive those bounds or prove the native shortlist/partition.
    cutoff, local_upper, global_upper, omitted_score = z3.Reals(
        "topk_cutoff local_discard_upper global_discard_upper omitted_reference_score"
    )
    local_unresolved, global_unresolved = z3.Bools("local_unresolved global_unresolved")
    frontiers = unresolved_frontiers_separated(
        cutoff, local_upper, global_upper, local_unresolved, global_unresolved
    )
    covered = z3.Or(
        z3.And(local_unresolved, omitted_score <= local_upper),
        z3.And(global_unresolved, omitted_score <= global_upper),
    )
    check(
        "two_frontier_topk_bound",
        omitted_score >= cutoff,
        assumptions=[frontiers, covered],
    )
    check(
        "mutant_local_only_omits_global_candidate",
        omitted_score > cutoff,
        expected=z3.sat,
        assumptions=[
            local_unresolved,
            global_unresolved,
            cutoff > local_upper,
            omitted_score <= global_upper,
        ],
    )
    check(
        "mutant_nonstrict_cutoff_accepts_omitted_tie",
        omitted_score == cutoff,
        expected=z3.sat,
        assumptions=[cutoff >= global_upper, omitted_score <= global_upper],
    )
    check(
        "mutant_frontier_coverage_gap",
        omitted_score > cutoff,
        expected=z3.sat,
        assumptions=[frontiers, z3.Not(covered)],
    )

    # A single induction step for a fail-closed published prefix. There is no
    # availability claim when the gate refuses publication. Reference, isolation
    # and event delivery remain explicit trusted preconditions.
    reference_state, candidate_state = z3.BitVecs("reference_state candidate_state", 64)
    reference_output, candidate_output = z3.BitVecs("reference_output candidate_output", 64)
    step = transaction_ready(
        reference_output == candidate_output,
        reference_state == candidate_state,
        identity,
        complete,
        completed,
        cancelled,
        base,
        epoch,
    )
    check(
        "published_prefix_induction_step",
        z3.And(
            step, z3.Or(reference_state != candidate_state, reference_output != candidate_output)
        ),
    )

    functions = {
        f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
        for f in (
            copy_accepted_prefix,
            publication_allowed,
            remaining_materialized,
            low_nibble,
            high_nibble,
            canonical_slot,
            bf16_round_bits,
            temporal_column,
            convolution_offset,
            transaction_ready,
            version_matches,
            writable_exclusively,
            snapshot_publishable,
            restore_matches,
            margin_preserved,
            separated_intervals,
            unresolved_frontiers_separated,
        )
    }
    report = seal(
        {
            "schema": "urn:qwen:conformance-helper-obligations:v1",
            "solver": z3.get_full_version(),
            "timeout_ms": timeout_ms,
            "functions_sha256": functions,
            "cases": rows,
            "all_expected_results": all(r["status"] != "UNPROVED" for r in rows),
            "proved_scope": (
                "actual pure helper expressions: 8-slot prefix copy; byte nibble decoding; "
                "unbounded-integer logical indexing and accepted state offsets; "
                "control/publication predicates, finite/Inf FP32-to-BF16 RNE bit expression "
                "and conditional rational interval/two-frontier cutoff inequalities; "
                "not native GPU kernels"
            ),
            "trusted": [
                "Z3",
                "Python execution and slicing",
                "Z3 expression translation",
                "hardware",
            ],
            "whole_gate_status": "TESTED",
            "bounds_soundness": "ASSUMED; empirical errors cannot establish universal bounds",
            "native_event_mapping": "UNPROVED",
            "radiance_status": "UNPROVED",
            "gpu_kernel_status": "UNPROVED",
            "compiler_status": "ASSUMED",
        }
    )
    write_private(output_root / "results.json", report)
    return report
