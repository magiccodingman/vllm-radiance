"""Strict event replay for concurrent cache and speculative state lifetimes.

The event consumer runs on CPU, with no GPU synchronization. A runtime adapter
must emit events at its real boundaries; an event claim is not proof of device
completion or tensor equality. Complete tensor comparisons provide those checks
separately. A truncated trace or unsupported event is never a successful audit.
"""

from copy import deepcopy
from pathlib import Path

from qwen_r9700_lab.conformance_invariants import (
    snapshot_publishable,
    transaction_ready,
    writable_exclusively,
)
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    integer,
    private_json,
    require_name,
    require_sha,
    seal,
    validate_speculative_commit,
    write_private,
)

SCHEMA = "urn:qwen:control-events:v1"


class ControlLedger:
    """Actual audit implementation; inputs contain identities/counts, no text."""

    def __init__(self):
        self.chats, self.transactions, self.blocks, self.snapshots = {}, {}, {}, {}
        self.committed, self.restored = 0, 0

    def apply(self, event):
        if not isinstance(event, dict) or "event" not in event:
            raise DiagnosticError("malformed control event")
        kind = event["event"]
        method = getattr(self, "event_" + kind, None) if isinstance(kind, str) else None
        if method is None:
            raise DiagnosticError("unsupported control event")
        # A rejected event cannot partially change the checker state.
        saved = deepcopy(self.__dict__)
        try:
            method(**{k: v for k, v in event.items() if k != "event"})
        except (KeyError, TypeError) as exc:
            self.__dict__.update(saved)
            raise DiagnosticError("incomplete or unregistered control state") from exc
        except Exception:
            self.__dict__.update(saved)
            raise

    def event_chat(self, chat, generation, consumed, pending, components):
        require_name(chat)
        require_sha(generation)
        integer(consumed)
        if type(pending) is not bool or not components or len(set(components)) != len(components):
            raise DiagnosticError("incomplete initial control state")
        for name in components:
            require_name(name)
        if chat in self.chats:
            raise DiagnosticError("chat registered twice")
        self.chats[chat] = {
            "generation": generation,
            "consumed": consumed,
            "pending": int(pending),
            "components": set(components),
            "revision": 0,
            "head": None,
        }

    def event_begin(self, transaction, chat, generation, drafted):
        require_name(transaction)
        state = self.chats[chat]
        if transaction in self.transactions or state["generation"] != generation:
            raise DiagnosticError("duplicate or stale transition")
        if integer(drafted) > 7:
            raise DiagnosticError("control profile admits M1 or at most seven drafts")
        self.transactions[transaction] = {
            "chat": chat,
            "generation": generation,
            "drafted": drafted,
            "revision": state["revision"],
            "completed": False,
            "validated": False,
            "cancelled": False,
            "closed": False,
        }

    def _active(self, transaction):
        tx = self.transactions[transaction]
        if tx["closed"]:
            raise DiagnosticError("closed transition cannot change state")
        return tx

    def event_fence(self, transaction):
        tx = self._active(transaction)
        if tx["completed"]:
            raise DiagnosticError("duplicate completion event")
        tx["completed"] = True

    def event_validate(self, transaction, output_equal, state_equal, identity_equal, complete):
        tx = self._active(transaction)
        values = (output_equal, state_equal, identity_equal, complete)
        if any(type(v) is not bool for v in values) or not tx["completed"] or tx["validated"]:
            raise DiagnosticError("validation requires completed writes and one comparison")
        tx["checks"], tx["validated"] = values, True

    def event_cancel(self, transaction):
        tx = self._active(transaction)
        tx["cancelled"], tx["closed"] = True, True

    def event_commit(self, transaction, accepted, emitted, consumed, pending, versions):
        tx = self._active(transaction)
        chat = self.chats[tx["chat"]]
        if not tx["validated"] or not transaction_ready(
            *tx.get("checks", (False,) * 4),
            tx["completed"],
            tx["cancelled"],
            (tx["generation"], tx["revision"]),
            (chat["generation"], chat["revision"]),
        ):
            raise DiagnosticError("unvalidated, unfinished, cancelled or stale publication")
        if type(pending) is not bool:
            raise DiagnosticError("pending state must be explicit")
        validate_speculative_commit(
            before_materialized=chat["consumed"],
            before_pending=chat["pending"],
            drafted=tx["drafted"],
            accepted=accepted,
            emitted=emitted,
            after_materialized=consumed,
            after_pending=int(pending),
            component_versions=versions,
            required_components=chat["components"],
        )
        chat.update(consumed=consumed, pending=int(pending), revision=chat["revision"] + 1)
        tx["closed"] = True
        self.committed += 1

    def event_generation(self, chat, generation, consumed, pending):
        state = self.chats[chat]
        require_sha(generation)
        integer(consumed)
        if generation == state["generation"] or type(pending) is not bool:
            raise DiagnosticError("invalid superseding generation")
        state.update(
            generation=generation,
            consumed=consumed,
            pending=int(pending),
            revision=state["revision"] + 1,
        )

    def event_allocate(self, block, chat, immutable, external=0):
        integer(block)
        integer(external)
        if chat not in self.chats or block in self.blocks or type(immutable) is not bool:
            raise DiagnosticError("invalid allocation or owner")
        self.blocks[block] = {"owners": {chat}, "immutable": immutable, "external": external}

    def event_share(self, block, chat):
        entry = self.blocks[block]
        if chat not in self.chats or chat in entry["owners"] or not entry["immutable"]:
            raise DiagnosticError("shared prefix must be immutable with distinct owners")
        entry["owners"].add(chat)

    def event_write(self, block, chat):
        entry = self.blocks[block]
        if entry["owners"] != {chat} or not writable_exclusively(
            len(entry["owners"]), entry["external"], entry["immutable"]
        ):
            raise DiagnosticError("write would modify a shared, pinned or different chat block")

    def event_copy_on_write(self, source, replacement, chat):
        if chat not in self.blocks[source]["owners"]:
            raise DiagnosticError("copy-on-write source is not owned by chat")
        self.event_allocate(replacement, chat, False)
        self.event_release(source, chat)

    def event_release(self, block, chat):
        entry = self.blocks[block]
        if chat not in entry["owners"]:
            raise DiagnosticError("release from a different owner")
        entry["owners"].remove(chat)
        if not entry["owners"] and not entry["external"]:
            del self.blocks[block]

    def event_snapshot(self, snapshot, chat, generation, consumed, state_sha256):
        require_name(snapshot)
        require_sha(state_sha256)
        state = self.chats[chat]
        if (
            snapshot in self.snapshots
            or generation != state["generation"]
            or consumed != state["consumed"]
        ):
            raise DiagnosticError("snapshot belongs to a stale or different state")
        self.snapshots[snapshot] = {
            "chat": chat,
            "generation": generation,
            "consumed": consumed,
            "state_sha256": state_sha256,
            "verified": False,
            "durable": False,
        }

    def event_verify_snapshot(self, snapshot, state_sha256, durable):
        entry = self.snapshots[snapshot]
        if state_sha256 != entry["state_sha256"] or type(durable) is not bool:
            raise DiagnosticError("snapshot verification changed state or omitted durability")
        entry.update(verified=True, durable=durable)

    def event_publish_snapshot(self, snapshot):
        entry = self.snapshots[snapshot]
        chat = self.chats[entry["chat"]]
        if not snapshot_publishable(
            entry["verified"], entry["durable"], chat["generation"], entry["generation"]
        ):
            raise DiagnosticError("snapshot is unverified, nondurable or superseded")
        if chat["head"] is not None:
            old = self.snapshots[chat["head"]]
            if old["generation"] == entry["generation"] and old["consumed"] > entry["consumed"]:
                raise DiagnosticError("snapshot publication would rewind the durable head")
        chat["head"] = snapshot

    def event_restore(self, snapshot, chat, generation, consumed, state_sha256):
        entry, state = self.snapshots[snapshot], self.chats[chat]
        if (
            state["head"] != snapshot
            or entry["chat"] != chat
            or generation != state["generation"]
            or generation != entry["generation"]
            or consumed != entry["consumed"]
            or state_sha256 != entry["state_sha256"]
        ):
            raise DiagnosticError("restored snapshot identity, version or bytes differ")
        self.restored += 1

    def finish(self):
        if not self.chats or not (self.committed or self.restored):
            raise DiagnosticError("empty control qualification domain")
        if any(not tx["closed"] for tx in self.transactions.values()):
            raise DiagnosticError("incomplete trace has unfinished transitions")
        return {"commits": self.committed, "restores": self.restored, "chats": len(self.chats)}


def audit_control(document: dict, output: Path | None = None):
    authenticate(document)
    if (
        set(document) != {"schema", "execution", "adapter", "events", "sha256"}
        or document["schema"] != SCHEMA
    ):
        raise DiagnosticError("incomplete control trace identity")
    require_sha(document["execution"])
    require_sha(document["adapter"])
    if not isinstance(document["events"], list) or not document["events"]:
        raise DiagnosticError("empty control trace")
    ledger, failure = ControlLedger(), None
    for index, event in enumerate(document["events"]):
        try:
            ledger.apply(event)
        except DiagnosticError as exc:
            failure = {"event_index": index, "reason": str(exc)}
            break
    counts = None
    if failure is None:
        try:
            counts = ledger.finish()
        except DiagnosticError as exc:
            failure = {"event_index": len(document["events"]), "reason": str(exc)}
    report = seal(
        {
            "schema": "urn:qwen:control-audit:v1",
            "trace": document["sha256"],
            "equal": failure is None,
            "first_failure": failure,
            "counts": counts,
            "status": "TESTED",
            "scope": "complete observed control event sequence",
            "device_completion_and_tensor_equality": "ASSUMED: independently check producer events",
            "native_adapter_qualification": "UNPROVED",
        }
    )
    if output is not None:
        write_private(output, report)
    return report


class ControlRecorder:
    """Create-once event spool for adapters; survives a crash at any event boundary."""

    def __init__(self, root: Path, *, execution: str, adapter: str):
        import threading

        require_sha(execution)
        require_sha(adapter)
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.root, self.lock, self.count, self.closed = root, threading.Lock(), 0, False
        self.identity = {"schema": SCHEMA, "execution": execution, "adapter": adapter}
        write_private(root / "identity.json", seal(self.identity))
        self.previous = digest(self.identity)

    def record(self, event: str, **fields):
        with self.lock:
            if self.closed:
                raise DiagnosticError("control recorder already finalized")
            entry = seal(
                {"index": self.count, "previous": self.previous, "data": {"event": event, **fields}}
            )
            write_private(self.root / f"{self.count:09d}.json", entry)
            self.previous, self.count = entry["sha256"], self.count + 1

    def finish(self):
        with self.lock:
            if self.closed:
                raise DiagnosticError("control recorder already finalized")
            self.closed = True
            doc = read_control_spool(self.root, count=self.count)
            write_private(self.root / "events.json", doc)
            return audit_control(doc)


def read_control_spool(root: Path, *, count: int):
    integer(count)
    identity = private_json(root / "identity.json")
    authenticate(identity)
    identity = {k: v for k, v in identity.items() if k != "sha256"}
    events, previous = [], digest(identity)
    for index in range(count):
        entry = private_json(root / f"{index:09d}.json")
        authenticate(entry)
        if entry["index"] != index or entry["previous"] != previous:
            raise DiagnosticError("control event chain is reordered, missing or changed")
        events.append(entry["data"])
        previous = entry["sha256"]
    return seal({**identity, "events": events})
