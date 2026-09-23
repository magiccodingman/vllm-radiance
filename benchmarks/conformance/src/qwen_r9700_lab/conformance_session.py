"""Durable fail-closed publication of independently checked private frames.

The SQLite commit is the authoritative publication point. Worker frame files
remain tentative and no output token is returned before the commit. An external
consumer resumes by revision; exactly-once external side effects require that
consumer to acknowledge/idempotently apply those revisions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import uuid
from pathlib import Path

from qwen_r9700_lab.conformance_gate import publication_allowed
from qwen_r9700_lab.conformance_state import (
    archive_frame,
    compare_frames,
    load_arrays,
    private_directory,
    read_frame,
)
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    digest,
    integer,
    remaining_materialized,
    seal,
    write_private,
)


class SessionMismatchError(DiagnosticError):
    def __init__(self, receipt: dict):
        super().__init__("unchecked transition rejected; no output published")
        self.receipt = receipt


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CheckedSession:
    def __init__(self, root: Path, *, contract: str, required_components: list[str], create=False):
        if create:
            root.mkdir(mode=0o700)
        private_directory(root)
        self.root, self.contract, self.required = root, contract, tuple(required_components)
        if not self.required or len(set(self.required)) != len(self.required):
            raise DiagnosticError("publication gate requires complete unique state coverage")
        database = root / "authority.sqlite3"
        if create:
            fd = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        if database.is_symlink() or database.stat().st_mode & 0o077:
            raise DiagnosticError("unsafe authority database")
        self.db = sqlite3.connect(database, isolation_level=None, timeout=10)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        if create:
            self.db.executescript("""
                CREATE TABLE metadata (contract TEXT NOT NULL, coverage TEXT NOT NULL);
                CREATE TABLE commits (
                    revision INTEGER PRIMARY KEY, operation TEXT NOT NULL,
                    input_digest TEXT NOT NULL, reference_frame TEXT NOT NULL,
                    candidate_frame TEXT NOT NULL, receipt TEXT NOT NULL,
                    token_bytes BLOB NOT NULL, stop_reason TEXT);
                CREATE TABLE failures (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL,
                    receipt TEXT NOT NULL);
            """)
            self.db.execute("INSERT INTO metadata VALUES (?, ?)", (contract, digest(self.required)))
            (root / "frames").mkdir(mode=0o700)
            sync_directory(root)
            sync_directory(root.parent)
        row = self.db.execute("SELECT contract,coverage FROM metadata").fetchone()
        if row != (contract, digest(self.required)):
            self.close()
            raise DiagnosticError("authority reference contract changed")

    @property
    def revision(self):
        return self.db.execute("SELECT COALESCE(MAX(revision),0) FROM commits").fetchone()[0]

    def commit(
        self,
        reference: Path,
        candidate: Path,
        *,
        base_revision: int,
        reference_tokens: tuple[int, ...],
        candidate_tokens: tuple[int, ...],
        reference_stop: str | None,
        candidate_stop: str | None,
    ) -> dict:
        integer(base_revision)
        for tokens in (reference_tokens, candidate_tokens):
            if type(tokens) is not tuple:
                raise DiagnosticError("tentative output must be immutable")
            for token in tokens:
                integer(token)
                if token >= 2**31:
                    raise DiagnosticError("invalid output token")
        allowed = {None, "eos", "tool_call", "complete"}
        if reference_stop not in allowed or candidate_stop not in allowed:
            raise DiagnosticError("errors and truncation cannot become successful EOS")
        # Workers may reuse their buffers after handing over a transition. The
        # authority holds its own verified copies, never a path into a worker.
        attempt = self.root / "frames" / uuid.uuid4().hex
        attempt.mkdir(mode=0o700)
        a = archive_frame(reference, attempt / "reference")
        b = archive_frame(candidate, attempt / "candidate")
        # Frame files and their own directory entries are synced by the writer.
        # Persist both parent links before SQLite can reference this attempt.
        sync_directory(attempt)
        sync_directory(attempt.parent)
        reference, candidate = attempt / "reference", attempt / "candidate"
        if any(f["logical"].get("execution_mode") == "forced_token_replay" for f in (a, b)):
            raise DiagnosticError("forced diagnostic tokens cannot be published to a session")
        required = list(self.required)
        if a["coverage"] != required or b["coverage"] != required:
            raise DiagnosticError("candidate did not expose the required complete state")
        if a["contract"] != self.contract or b["contract"] != self.contract:
            raise DiagnosticError("candidate uses a different model contract")
        if "sequence.tokens" in required:
            _, sequence = load_arrays(reference)
            tokens = sequence["sequence.tokens"]
            position = sequence.get("sequence.position")
            if tokens.dtype.str != "<i4" or tokens.ndim != 1 or len(tokens) != a["consumed"]:
                raise DiagnosticError("reference token state has an invalid representation")
            if (
                position is None
                or position.dtype.str != "<i8"
                or position.tolist() != [len(tokens)]
            ):
                raise DiagnosticError("reference position does not describe its token state")
            if digest(tokens.tolist()) != a["input_digest"]:
                raise DiagnosticError(
                    "reference consumed prefix identity does not match its tokens"
                )
        comparison = compare_frames(reference, candidate)
        output_equal = reference_tokens == candidate_tokens and reference_stop == candidate_stop
        receipt = seal(
            {
                "schema": "urn:qwen:checked-session-receipt:v1",
                "base_revision": base_revision,
                "comparison": comparison["sha256"],
                "state_equal": comparison["equal"],
                "output_equal": output_equal,
                "reference_frame": a["sha256"],
                "candidate_frame": b["sha256"],
                "reference_consumed": a["consumed"],
                "reference_pending": a["pending"],
                "output_count": len(reference_tokens),
                "first_difference": comparison["first_difference"],
                "formal_backend_equivalence": "UNPROVED",
            }
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.revision != base_revision:
                raise DiagnosticError("stale transition cannot replace the authoritative state")
            previous = self.db.execute(
                "SELECT reference_frame,receipt FROM commits ORDER BY revision DESC LIMIT 1"
            ).fetchone()
            if previous is not None and a["phase"] == "prefill":
                raise DiagnosticError("initial prefill cannot overwrite a live authority")
            if previous is None and a["phase"] != "prefill":
                raise DiagnosticError("an independently computed initial prefill is required")
            if previous is not None:
                old, old_arrays = load_arrays(Path(previous[0]))
                if old["sha256"] != json.loads(previous[1])["reference_frame"]:
                    raise DiagnosticError("authoritative state changed after commit")
                expected_position = remaining_materialized(
                    old["consumed"],
                    int(old["pending"] is not None),
                    len(reference_tokens),
                    int(a["pending"] is not None),
                )
                if a["consumed"] != expected_position:
                    raise DiagnosticError("materialized/pending token conservation failed")
                if "sequence.tokens" in old_arrays:
                    _, current_arrays = load_arrays(reference)
                    old_tokens = old_arrays["sequence.tokens"].tolist()
                    new_tokens = current_arrays["sequence.tokens"].tolist()
                    emitted = (
                        old_tokens
                        + ([old["pending"]] if old["pending"] is not None else [])
                        + list(reference_tokens)
                    )
                    if new_tokens + ([a["pending"]] if a["pending"] is not None else []) != emitted:
                        raise DiagnosticError(
                            "transition did not preserve the emitted token prefix"
                        )
            if a["pending"] is not None and (
                not reference_tokens or reference_tokens[-1] != a["pending"]
            ):
                raise DiagnosticError("pending token does not match the emitted suffix")
            if reference_stop == "tool_call" and "protocol.events" not in self.required:
                raise DiagnosticError(
                    "tool publication requires independently checked parser events"
                )
            if not publication_allowed(
                output_equal,
                comparison["equal"],
                a["contract"] == b["contract"] == self.contract,
                set(a["coverage"]) == set(b["coverage"]) == set(self.required),
            ):
                self.db.execute(
                    "INSERT INTO failures(revision,receipt) VALUES (?,?)",
                    (base_revision, json.dumps(receipt)),
                )
                self.db.execute("COMMIT")
                raise SessionMismatchError(receipt)
            payload = b"".join(struct.pack("<i", v) for v in reference_tokens)
            # Only reference-owned state is retained as authoritative. Candidate
            # files and its mutable state are never reused by a fallback worker.
            self.db.execute(
                "INSERT INTO commits VALUES (?,?,?,?,?,?,?,?)",
                (
                    base_revision + 1,
                    a["phase"],
                    a["input_digest"],
                    str(reference),
                    str(candidate),
                    json.dumps(receipt),
                    payload,
                    reference_stop,
                ),
            )
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return {
            "revision": base_revision + 1,
            "tokens": reference_tokens,
            "stop_reason": reference_stop,
            "receipt": receipt["sha256"],
        }

    def outputs_since(self, revision: int):
        integer(revision)
        return [
            {
                "revision": row[0],
                "tokens": tuple(v[0] for v in struct.iter_unpack("<i", row[1])),
                "stop_reason": row[2],
            }
            for row in self.db.execute(
                "SELECT revision,token_bytes,stop_reason FROM commits "
                "WHERE revision>? ORDER BY revision",
                (revision,),
            )
        ]

    def summary(self):
        return {
            "revision": self.revision,
            "rejected_transitions": self.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0],
            "publication": "RUNTIME-CHECKED" if self.revision else "UNPROVED",
            "scope": "submitted_frame_and_output_equality",
            "native_state_completeness": "UNPROVED",
            "formal_backend_equivalence": "UNPROVED",
        }

    def close(self):
        self.db.close()


def compare_campaign(reference: Path, candidate: Path, output: Path) -> dict:
    """Compare a sealed schedule, including initial prefill, in causal order."""
    from qwen_r9700_lab.diagnostic_contract import authenticate, private_json

    schedules = [private_json(p / "schedule.json") for p in (reference, candidate)]
    for schedule in schedules:
        authenticate(schedule)
        if schedule.get("schema") != "urn:qwen:conformance-schedule:v1" or not schedule.get(
            "frames"
        ):
            raise DiagnosticError("missing or empty execution schedule")
    from qwen_r9700_lab.conformance_replay import scheduled_inputs, validate_plan

    plans = [validate_plan(private_json(p / "plan.json")) for p in (reference, candidate)]
    if plans[0] != plans[1] or schedules[0]["contract"] != schedules[1]["contract"]:
        raise DiagnosticError("reference and candidate executed different schedules")
    expected = list(scheduled_inputs(plans[0]))
    for root, schedule in zip((reference, candidate), schedules, strict=True):
        if schedule.get("plan") != plans[0]["sha256"] or len(schedule["frames"]) != len(expected):
            raise DiagnosticError("schedule omitted required observations")
        if (
            schedule.get("initial_state") != "independent_zero_state"
            or schedule.get("published_to_session") is not False
        ):
            raise DiagnosticError("replay did not declare independent initial prefill")
        for actual, required in zip(schedule["frames"], expected, strict=True):
            if {k: v for k, v in actual.items() if k != "sha256"} != required:
                raise DiagnosticError("schedule diverged from its declared plan")
            frame = read_frame(root / required["name"])
            if frame["sha256"] != actual["sha256"] or frame["coverage"] != schedule["coverage"]:
                raise DiagnosticError("schedule frame changed or is incomplete")
    if schedules[0]["coverage"] != schedules[1]["coverage"]:
        raise DiagnosticError("reference and candidate coverage differs")
    output.mkdir(mode=0o700)
    rows, first = [], None
    for index, entry in enumerate(expected):
        name = entry["name"]
        result = compare_frames(reference / name, candidate / name)
        write_private(output / f"comparison-{index:06d}.json", result)
        rows.append({"index": index, "equal": result["equal"], "sha256": result["sha256"]})
        if first is None and not result["equal"]:
            first = {"frame": index, **result["first_difference"]}
    report = seal(
        {
            "schema": "urn:qwen:conformance-campaign-result:v1",
            "frames": rows,
            "first_difference": first,
            "equal": first is None,
            "formal_backend_equivalence": "UNPROVED",
        }
    )
    write_private(output / "report.json", report)
    return report
