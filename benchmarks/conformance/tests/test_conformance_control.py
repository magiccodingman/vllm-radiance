from copy import deepcopy
from fractions import Fraction

import pytest

from qwen_r9700_lab.conformance_control import (
    SCHEMA,
    ControlLedger,
    ControlRecorder,
    audit_control,
    read_control_spool,
)
from qwen_r9700_lab.conformance_invariants import interval_certificate, rejection_distribution
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal

GEN = digest("generation A")
COMPONENTS = ["kv", "conv", "gdn"]


def event(kind, **fields):
    return {"event": kind, **fields}


def transcript(width=0):
    return [
        event("chat", chat="A", generation=GEN, consumed=32, pending=True, components=COMPONENTS),
        event("begin", transaction="t1", chat="A", generation=GEN, drafted=7),
        event("fence", transaction="t1"),
        event(
            "validate",
            transaction="t1",
            output_equal=True,
            state_equal=True,
            identity_equal=True,
            complete=True,
        ),
        event(
            "commit",
            transaction="t1",
            accepted=width,
            emitted=width + 1,
            consumed=33 + width,
            pending=True,
            versions=dict.fromkeys(COMPONENTS, 33 + width),
        ),
    ]


def audit(events):
    return audit_control(
        seal(
            {
                "schema": SCHEMA,
                "execution": digest("test execution"),
                "adapter": digest("test adapter"),
                "events": events,
            }
        )
    )


@pytest.mark.parametrize("width", range(8))
def test_every_d7_width_has_exact_pending_and_state_version_conservation(width):
    result = audit(transcript(width))
    assert result["equal"]
    assert result["counts"]["commits"] == 1
    assert result["native_adapter_qualification"] == "UNPROVED"


@pytest.mark.parametrize("component", COMPONENTS)
def test_wrong_component_version_is_not_a_numerical_tolerance(component):
    events = transcript(3)
    events[-1]["versions"][component] += 1
    result = audit(events)
    assert not result["equal"]
    assert result["first_failure"]["event_index"] == 4


@pytest.mark.parametrize(
    "fault",
    [
        "missing_fence",
        "state",
        "output",
        "identity",
        "coverage",
        "cancelled",
        "stale_revision",
        "stale_generation",
        "pending",
    ],
)
def test_deliberate_publication_faults_fail_closed(fault):
    events = transcript()
    if fault == "missing_fence":
        del events[2]
    elif fault in {"state", "output", "identity", "coverage"}:
        key = "complete" if fault == "coverage" else fault + "_equal"
        events[3][key] = False
    elif fault == "cancelled":
        events.insert(4, event("cancel", transaction="t1"))
    elif fault == "stale_revision":
        second = deepcopy(events[1:])
        for e in second:
            e["transaction"] = "t2"
        events = events[:2] + second + events[2:]
    elif fault == "stale_generation":
        events.insert(
            4, event("generation", chat="A", generation=digest("new"), consumed=3, pending=True)
        )
    else:
        events[-1]["pending"] = False
    assert not audit(events)["equal"]


def test_empty_truncated_duplicate_and_unknown_events_never_pass():
    with pytest.raises(DiagnosticError, match="empty"):
        audit([])
    assert not audit(transcript()[:-1])["equal"]
    assert not audit([*transcript(), event("made_up_event")])["equal"]
    assert not audit([*transcript(), transcript()[-1]])["equal"]


def ledger_with_two_chats():
    ledger = ControlLedger()
    ledger.apply(transcript()[0])
    ledger.apply({**transcript()[0], "chat": "B"})
    return ledger


def test_shared_prefix_copy_on_write_preserves_other_chat_and_references():
    ledger = ledger_with_two_chats()
    ledger.apply(event("allocate", block=917, chat="A", immutable=True))
    ledger.apply(event("share", block=917, chat="B"))
    with pytest.raises(DiagnosticError, match="shared"):
        ledger.apply(event("write", block=917, chat="A"))
    ledger.apply(event("copy_on_write", source=917, replacement=42, chat="A"))
    ledger.apply(event("write", block=42, chat="A"))
    assert ledger.blocks[917]["owners"] == {"B"}
    with pytest.raises(DiagnosticError, match="different"):
        ledger.apply(event("write", block=42, chat="B"))
    ledger.apply(event("release", block=42, chat="A"))
    assert 42 not in ledger.blocks
    assert ledger.blocks[917]["immutable"]


def test_failed_copy_on_write_is_atomic_and_external_pin_prevents_writes():
    ledger = ledger_with_two_chats()
    ledger.apply(event("allocate", block=1, chat="A", immutable=False, external=1))
    before = deepcopy(ledger.blocks)
    with pytest.raises(DiagnosticError):
        ledger.apply(event("copy_on_write", source=1, replacement=1, chat="A"))
    assert ledger.blocks == before
    with pytest.raises(DiagnosticError, match="pinned"):
        ledger.apply(event("write", block=1, chat="A"))
    with pytest.raises(DiagnosticError, match="immutable"):
        ledger.apply(event("share", block=1, chat="B"))


def test_verified_previous_snapshot_survives_failed_replacement_and_compaction():
    ledger = ledger_with_two_chats()
    for name in ("old", "replacement"):
        ledger.apply(
            event(
                "snapshot",
                snapshot=name,
                chat="A",
                generation=GEN,
                consumed=32,
                state_sha256=digest(name),
            )
        )
    ledger.apply(event("verify_snapshot", snapshot="old", state_sha256=digest("old"), durable=True))
    ledger.apply(event("publish_snapshot", snapshot="old"))
    with pytest.raises(DiagnosticError):
        ledger.apply(event("publish_snapshot", snapshot="replacement"))
    assert ledger.chats["A"]["head"] == "old"
    ledger.apply(
        event(
            "restore",
            snapshot="old",
            chat="A",
            generation=GEN,
            consumed=32,
            state_sha256=digest("old"),
        )
    )
    with pytest.raises(DiagnosticError):
        ledger.apply(
            event(
                "restore",
                snapshot="old",
                chat="B",
                generation=GEN,
                consumed=32,
                state_sha256=digest("old"),
            )
        )
    ledger.apply(
        event("generation", chat="A", generation=digest("compacted"), consumed=4, pending=True)
    )
    ledger.apply(
        event(
            "verify_snapshot",
            snapshot="replacement",
            state_sha256=digest("replacement"),
            durable=True,
        )
    )
    with pytest.raises(DiagnosticError):
        ledger.apply(event("publish_snapshot", snapshot="replacement"))
    assert ledger.chats["A"]["head"] == "old"


def test_content_free_spool_records_real_order_and_detects_lost_or_reordered_events(tmp_path):
    recorder = ControlRecorder(
        tmp_path / "events", execution=digest("exec"), adapter=digest("adapter")
    )
    for e in transcript():
        recorder.record(e["event"], **{k: v for k, v in e.items() if k != "event"})
    assert recorder.finish()["equal"]
    first, second = (tmp_path / "events" / f"{n:09d}.json" for n in (0, 1))
    first.write_bytes(second.read_bytes())
    with pytest.raises(DiagnosticError, match="reordered"):
        read_control_spool(tmp_path / "events", count=5)


def test_interval_certificate_rejects_ties_and_missing_true_winner():
    assert interval_certificate(["2", "0"], ["3", "1"], 0, bound_origin="fixture")[
        "certified_under_bounds"
    ]
    assert not interval_certificate(["1", "0"], ["2", "1"], 0, bound_origin="fixture")[
        "certified_under_bounds"
    ]
    result = interval_certificate(
        ["2", "0", "4"], ["3", "1", "5"], 0, bound_origin="shortlist omitted token 2"
    )
    assert not result["certified_under_bounds"]
    assert result["bounds_soundness"] == "ASSUMED"
    with pytest.raises(DiagnosticError):
        interval_certificate([1, 2], [3], 0, bound_origin="fixture")


@pytest.mark.parametrize(
    "p,q",
    [
        (["1/2", "1/2"], ["1/2", "1/2"]),
        (["1", "0"], ["0", "1"]),
        (["0", "1/3", "2/3"], ["1/4", "3/4", "0"]),
    ],
)
def test_exact_rejection_reference_covers_zero_draft_and_full_acceptance(p, q):
    result = rejection_distribution(p, q)
    assert result["output"] == [Fraction(v) for v in p]
    if result["rejected_mass"]:
        assert sum(result["residual"]) == 1
    else:
        assert result["residual"] is None
