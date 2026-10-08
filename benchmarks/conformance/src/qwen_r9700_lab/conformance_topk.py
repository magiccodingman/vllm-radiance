"""Aligned M1/M8 logit comparisons; no inference or token decoding at import.

Ordering is score descending, token ID ascending. Exact membership, ordering,
overlap and tied cutoffs are different measurements and are reported separately.
This is observed equivalence, never a proof for unobserved model inputs.
"""

from __future__ import annotations

import hashlib

import numpy as np

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, digest, seal

KS = (1, 10, 20)


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def summarize_logits(logits):
    """Keep exact leading scores and every boundary tie, without serializing text."""
    values = np.asarray(logits)
    require(values.ndim == 1 and values.size >= 21, "incomplete vocabulary row")
    require(values.dtype == np.float32, "logits must preserve the native FP32 output")
    require(np.all(np.isfinite(values)), "non-finite full-head logits")
    threshold = np.partition(values, values.size - 21)[-21]
    ids = np.flatnonzero(values >= threshold)
    order = np.lexsort((ids, -values[ids]))
    ids = ids[order]
    scores = values[ids]
    return {
        "vocabulary": int(values.size),
        "logits_sha256": hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest(),
        "ids": ids.tolist(),
        "scores": scores.tolist(),
        "boundary_ties": {str(k): int(np.count_nonzero(values == scores[k - 1])) for k in KS},
    }


def validate_row(row):
    ids, scores = row["ids"], row["scores"]
    require(len(ids) == len(scores) >= 21, "incomplete top-k evidence")
    require(len(set(ids)) == len(ids), "duplicate vocabulary IDs")
    require(all(type(i) is int and 0 <= i < row["vocabulary"] for i in ids), "invalid ID")
    require(all(np.isfinite(x) for x in scores), "non-finite retained score")
    require(
        list(zip(ids, scores, strict=True))
        == sorted(zip(ids, scores, strict=True), key=lambda x: (-x[1], x[0])),
        "noncanonical ranking",
    )
    for k in KS:
        cutoff = scores[k - 1]
        require(
            row["boundary_ties"][str(k)] == sum(x == cutoff for x in scores),
            "missing tied boundary candidates",
        )


def compare_rows(reference, candidate):
    validate_row(reference)
    validate_row(candidate)
    require(reference["vocabulary"] == candidate["vocabulary"], "vocabulary changed")
    result = {"full_logits_exact": reference["logits_sha256"] == candidate["logits_sha256"]}
    for k in KS:
        left, right = reference["ids"][:k], candidate["ids"][:k]
        left_ties = {
            i
            for i, score in zip(reference["ids"], reference["scores"], strict=True)
            if score >= reference["scores"][k - 1]
        }
        right_ties = {
            i
            for i, score in zip(candidate["ids"], candidate["scores"], strict=True)
            if score >= candidate["scores"][k - 1]
        }
        result[str(k)] = {
            "set_exact": set(left) == set(right),
            "ranked_exact": left == right,
            "overlap": len(set(left) & set(right)),
            "inclusive_tie_set_exact": left_ties == right_ties,
            "reference_boundary_tied": reference["boundary_ties"][str(k)] > 1,
            "candidate_boundary_tied": candidate["boundary_ties"][str(k)] > 1,
            "retained_scores_exact": left == right
            and reference["scores"][:k] == candidate["scores"][:k],
        }
    return result


def aggregate(comparisons):
    require(bool(comparisons), "empty equivalence measurement")
    total = len(comparisons)
    result = {
        "positions": total,
        "full_logits_exact": sum(r["full_logits_exact"] for r in comparisons),
    }
    for k in KS:
        rows = [r[str(k)] for r in comparisons]
        result[str(k)] = {
            **{
                key: sum(row[key] for row in rows)
                for key in (
                    "set_exact",
                    "ranked_exact",
                    "inclusive_tie_set_exact",
                    "reference_boundary_tied",
                    "candidate_boundary_tied",
                    "retained_scores_exact",
                )
            },
            "set_exact_percent": 100 * sum(row["set_exact"] for row in rows) / total,
            "ranked_exact_percent": 100 * sum(row["ranked_exact"] for row in rows) / total,
            "mean_overlap_tokens": sum(row["overlap"] for row in rows) / total,
            "mean_overlap_percent": 100 * sum(row["overlap"] for row in rows) / (total * k),
        }
    return result


def compare_saved_rows(before, after, *, target_rows):
    """Compare one arm across revisions without assuming its reference is unchanged."""
    for report in (before, after):
        authenticate(report)
        require(
            report.get("schema") == "urn:qwen:d7-equivalence-private-rows:v1",
            "wrong saved replay evidence",
        )
    require(type(target_rows) is int and target_rows in (1, 8), "unsupported target row count")
    require(before["continuation"] == after["continuation"], "saved replay histories differ")
    require(
        len(before["rows"]) == len(after["rows"]) > 0,
        "saved replay lengths differ or are empty",
    )
    compared = []
    for position, (left, right) in enumerate(zip(before["rows"], after["rows"], strict=True)):
        require(
            left["position"] == right["position"] == position
            and left["absolute_position"] == right["absolute_position"],
            "saved replay positions differ",
        )
        require(
            left["target_rows"] == right["target_rows"] == target_rows,
            "saved replay execution arms differ",
        )
        compared.append(compare_rows(left["logits"], right["logits"]))
    return compared, compare_rows(before["prefill"], after["prefill"])


def compare_measurements(before, after):
    """Join completed before/after evidence only for the identical frozen corpus."""
    for report in (before, after):
        authenticate(report)
        require(
            report.get("schema") == "urn:qwen:d7-equivalence-summary:v1"
            and report.get("status") == "MEASURED",
            "before/after comparison requires completed native measurements",
        )
    for key in ("corpus", "ordering", "scope"):
        require(before[key] == after[key], "before/after measurement contract differs")
    require(before["revision"] != after["revision"], "before and after are the same revision")
    require(
        before["metrics"]["positions"] == after["metrics"]["positions"] == 10000,
        "before/after comparison requires all 10,000 positions in each measurement",
    )
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-before-after:v1",
            "before": before["sha256"],
            "after": after["sha256"],
            "corpus": before["corpus"],
            "positions": 10000,
            "top_k": {
                str(k): {
                    "before": before["metrics"][str(k)],
                    "after": after["metrics"][str(k)],
                    "set_agreement_percentage_point_change": (
                        after["metrics"][str(k)]["set_exact_percent"]
                        - before["metrics"][str(k)]["set_exact_percent"]
                    ),
                }
                for k in KS
            },
            "scope": "observed native agreement; not a universal proof or answer-quality score",
        }
    )


def choose_continuations(records, target):
    """Select a fixed chronological corpus before inspecting any M1/M8 result.

    The prefill prediction is excluded. N saved output tokens provide N-1 decode
    positions with a real pending successor; the last response may be truncated
    in the replay corpus to make the evaluated-position budget exact.
    """
    require(
        type(target) is int and target > 0 and target % 8 == 0,
        "position budget must contain complete M8 groups",
    )
    selected, remaining, seen = [], target, set()
    for row in records:
        require(
            row["mode"] == "full" and row["finish_reason"] == "stop",
            "not a natural full-head response",
        )
        require(row["trial"] not in seen, "duplicate saved response")
        seen.add(row["trial"])
        # Never count a scheduler-shortened final batch as an M8 observation.
        count = min(remaining, 8 * ((row["output_tokens"] - 1) // 8))
        if count > 0:
            selected.append({**row, "evaluate_positions": count})
            remaining -= count
        if remaining == 0:
            break
    require(remaining == 0, "saved natural responses do not cover the position budget")
    return selected


class ReplaySchedule:
    """Pure controller for the source-bound runner adapter; tests use no GPU."""

    def __init__(self, prefix, output, *, speculation):
        require(bool(prefix) and len(output) >= 2, "empty forced replay")
        require(all(type(t) is int and t >= 0 for t in (*prefix, *output)), "invalid forced token")
        self.prefix, self.output, self.speculation = prefix, output, speculation
        self.cursor = 0
        self.prefill_done = False
        self.prefill_cursor = 0

    @property
    def done(self):
        return self.prefill_done and self.cursor == len(self.output) - 1

    def expected_inputs(self, positions):
        history = self.prefix + self.output
        return [history[p] if p < len(history) else 0 for p in positions]

    def check_inputs(self, positions, inputs):
        require(positions and len(positions) == len(inputs), "empty or incomplete model input")
        require(
            positions == list(range(positions[0], positions[0] + len(positions))),
            "nonconsecutive model positions",
        )
        require(
            inputs == self.expected_inputs(positions),
            "model did not consume the frozen token history",
        )
        if self.prefill_done:
            require(positions[0] == len(self.prefix) + self.cursor, "decode history is misaligned")
            require(
                len(positions) == (8 if self.speculation else 1), "wrong target execution width"
            )
        else:
            require(positions[0] == self.prefill_cursor, "prefill skipped or repeated a token")
            require(positions[-1] < len(self.prefix), "prefill crossed into the output suffix")
            self.prefill_cursor = positions[-1] + 1

    def commit(self, drafts):
        require(not self.done, "replay exceeded its exact position budget")
        if not self.prefill_done:
            require(drafts == 0, "speculation appeared during initial prefill")
            require(
                self.prefill_cursor == len(self.prefix), "prefill did not consume the full prompt"
            )
            self.prefill_done = True
            return {
                "prefill": True,
                "start": -1,
                "count": 1,
                "tokens": self.output[:1],
                "reject": 0,
            }
        require(drafts == (7 if self.speculation else 0), "target width changed during replay")
        count = min(drafts + 1, len(self.output) - 1 - self.cursor)
        start = self.cursor
        tokens = self.output[start + 1 : start + count + 1]
        self.cursor += count
        return {
            "prefill": False,
            "start": start,
            "count": count,
            "tokens": tokens,
            "reject": drafts + 1 - count,
        }

    def proposals(self):
        return [
            self.output[p] if p < len(self.output) else 0
            for p in range(self.cursor + 1, self.cursor + 8)
        ]

    def history_digest(self, position):
        return digest(self.prefix + self.output[: position + 1])
