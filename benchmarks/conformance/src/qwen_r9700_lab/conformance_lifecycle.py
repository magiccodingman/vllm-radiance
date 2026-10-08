"""Recovery comparisons and real compressed-store transport, with no GPU imports.

Transport tests exercise Radiance's ChatStore, not a substitute JSON snapshot.
The canonical-frame envelope is diagnostic: it does not claim to implement the
native vLLM connector's page mapping. Native captures use the same comparison
contract through RecoveryCapture; their extraction must be qualified separately.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from pathlib import Path

from qwen_r9700_lab.conformance_state import (
    FrameWriter,
    archive_frame,
    compare_frames,
    open_blob,
    read_frame,
)
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    integer,
    require_sha,
    seal,
    write_private,
)
from qwen_r9700_lab.radiance_cache import KEY, ChatStore

ENVELOPE = b"QWENFRAME1\0"
RECOVERY_SCHEMA = "urn:qwen:recovery-comparison:v1"
STAGES = ("reference", "live", "restored")


def state_view(source: Path, destination: Path, required: list[str]) -> dict:
    """Compare logical state across phases, excluding outputs and physical layout.

    Required coverage comes from the model contract, never from an intersection
    of whatever two producers happen to emit. All original evidence is retained.
    """
    frame = read_frame(source)
    if not required or len(set(required)) != len(required):
        raise DiagnosticError("recovery comparison requires explicit complete state coverage")
    if any(name not in frame["components"] for name in required):
        raise DiagnosticError("recovery capture omitted required state")
    writer = FrameWriter(
        destination,
        contract=frame["contract"],
        execution=frame["execution"],
        adapter=frame["adapter"],
        input_digest=frame["input_digest"],
        phase="restore",
        consumed=frame["consumed"],
        pending=frame["pending"],
        expected=required,
        # Mode describes the producer. All other logical metadata is compared;
        # dropping sampler/version fields would hide latent state corruption.
        logical={k: v for k, v in frame["logical"].items() if k != "execution_mode"},
    )
    for name in required:
        descriptor = frame["components"][name]
        with open_blob(source, descriptor["file"]) as stream:
            writer.add_stream(
                name,
                iter(lambda: stream.read(1024 * 1024), b""),
                dtype=descriptor["dtype"],
                shape=descriptor["shape"],
            )
        if writer.components[name]["sha256"] != descriptor["sha256"]:
            raise DiagnosticError("recovery source changed during capture")
    return writer.finish()


def recovery_classification(live_equal: bool, restored_equal: bool, unchanged: bool) -> str:
    if live_equal and restored_equal:
        return "all_observed_state_matches"
    if not live_equal and restored_equal:
        return "live_differs_restored_matches_reference"
    if live_equal and not restored_equal:
        return "restore_introduced_difference"
    return "same_difference_survives_restore" if unchanged else "both_differ_from_reference"


def compare_recovery(
    reference: Path, live: Path, restored: Path, output: Path, *, required: list[str]
) -> dict:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sources = dict(zip(STAGES, (reference, live, restored), strict=True))
    identities = {}
    for stage, path in sources.items():
        archived = output / (stage + "-original")
        identities[stage] = archive_frame(path, archived)["sha256"]
        state_view(archived, output / stage, required)
    comparisons = {
        "reference_live": compare_frames(output / "reference", output / "live"),
        "reference_restored": compare_frames(output / "reference", output / "restored"),
        "live_restored": compare_frames(output / "live", output / "restored"),
    }
    report = seal(
        {
            "schema": RECOVERY_SCHEMA,
            "sources": identities,
            "required_components": required,
            "comparisons": comparisons,
            "classification": recovery_classification(
                comparisons["reference_live"]["equal"],
                comparisons["reference_restored"]["equal"],
                comparisons["live_restored"]["equal"],
            ),
            "equal": all(r["equal"] for r in comparisons.values()),
            "status": "TESTED",
            "scope": "captured logical state at identical input prefix and position",
            "corruption_cause": "UNPROVED: a numerical difference is not itself a causal diagnosis",
            "reference_correctness": "ASSUMED: independently qualify the reference producer",
            "native_connector_qualification": "UNPROVED",
        }
    )
    write_private(output / "report.json", report)
    return report


def pack_frame(source: Path) -> bytes:
    frame = read_frame(source)
    if not compare_frames(source, source)["equal"]:
        raise DiagnosticError("nonfinite snapshot cannot enter a checked transport")
    metadata = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    parts = [ENVELOPE, struct.pack("<Q", len(metadata)), metadata]
    for name in frame["coverage"]:
        descriptor = frame["components"][name]
        with open_blob(source, descriptor["file"]) as stream:
            data = stream.read()
        if (
            len(data) != descriptor["nbytes"]
            or hashlib.sha256(data).hexdigest() != descriptor["sha256"]
        ):
            raise DiagnosticError("snapshot changed while serializing")
        parts.append(data)
    return b"".join(parts)


def unpack_frame(payload: bytes, output: Path) -> dict:
    header = len(ENVELOPE) + 8
    if len(payload) < header or payload[: len(ENVELOPE)] != ENVELOPE:
        raise DiagnosticError("invalid diagnostic snapshot envelope")
    length = struct.unpack_from("<Q", payload, len(ENVELOPE))[0]
    if length > len(payload) - header:
        raise DiagnosticError("truncated diagnostic snapshot header")
    try:
        frame = json.loads(payload[header : header + length])
        authenticate(frame)
        coverage = frame["coverage"]
        components = frame["components"]
        if not isinstance(coverage, list) or set(coverage) != set(components):
            raise DiagnosticError("invalid diagnostic snapshot coverage")
        total = sum(integer(components[n]["nbytes"]) for n in coverage)
        if header + length + total != len(payload):
            raise DiagnosticError("snapshot payload has missing or extra bytes")
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticError("invalid diagnostic snapshot metadata") from exc
    writer = FrameWriter(
        output,
        **{k: frame[k] for k in ("contract", "execution", "adapter", "input_digest")},
        phase=frame["phase"],
        consumed=frame["consumed"],
        pending=frame["pending"],
        expected=coverage,
        logical=frame["logical"],
    )
    cursor = header + length
    for name in coverage:
        descriptor = components[name]
        data = payload[cursor : cursor + descriptor["nbytes"]]
        cursor += descriptor["nbytes"]
        if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
            raise DiagnosticError("diagnostic snapshot payload checksum mismatch")
        writer.add(name, data, dtype=descriptor["dtype"], shape=descriptor["shape"])
    result = writer.finish()
    if result != frame:
        raise DiagnosticError("snapshot roundtrip changed its canonical identity")
    return result


class FrameTransport:
    """Immutable RAM copies plus real ChatStore compression/publication/GC.

    A content-addressed receipt is returned only after verified publication.
    Callers keep the previous receipt until then. Restart constructs a new
    transport with an empty RAM bank and restores through ChatStore.read.
    """

    def __init__(self, root: Path, chat: dict, *, block_size=4096):
        if integer(block_size) < 64:
            raise DiagnosticError("diagnostic snapshot blocks are too small")
        self.store = ChatStore(root, chat)
        self.block_size = block_size
        self.ram: bytes | None = None
        self.store.activate()

    def save_ram(self, source: Path) -> None:
        self.ram = pack_frame(source)

    def restore_ram(self, output: Path) -> dict:
        if self.ram is None:
            raise DiagnosticError("RAM snapshot was evicted")
        return unpack_frame(self.ram, output)

    def stage_disk(self, source: Path) -> dict:
        payload = pack_frame(source)
        keys, blocks = [], []
        for offset in range(0, len(payload), self.block_size):
            data = payload[offset : offset + self.block_size].ljust(self.block_size, b"\0")
            key = "g0-" + hashlib.sha256(data).hexdigest() + ".qkv"
            keys.append(key)
            blocks.append((key, memoryview(data)))
        if not self.store.write_many(blocks):
            raise DiagnosticError("snapshot generation was superseded while staging")
        frame = read_frame(source)
        return seal(
            {
                "schema": "urn:qwen:diagnostic-frame-receipt:v1",
                "chat": self.store.chat,
                "keys": keys,
                "block_size": self.block_size,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "frame_sha256": frame["sha256"],
                "consumed": frame["consumed"],
            }
        )

    def publish(self, receipt: dict) -> bool:
        self._validate_receipt(receipt)
        return self.store.publish(receipt["keys"], receipt["consumed"], self.block_size)

    def save_disk(self, source: Path) -> dict:
        receipt = self.stage_disk(source)
        if not self.publish(receipt):
            raise DiagnosticError("incomplete checkpoint: previous disk head retained")
        return receipt

    def _validate_receipt(self, receipt):
        authenticate(receipt)
        if (
            receipt.get("schema") != "urn:qwen:diagnostic-frame-receipt:v1"
            or receipt.get("chat") != self.store.chat
            or receipt.get("block_size") != self.block_size
            or not receipt.get("keys")
        ):
            raise DiagnosticError("snapshot receipt belongs to a different chat or generation")
        for key in receipt["keys"]:
            if not isinstance(key, str) or not KEY.fullmatch(key):
                raise DiagnosticError("invalid diagnostic snapshot object key")
        require_sha(receipt["payload_sha256"])
        require_sha(receipt["frame_sha256"])
        size = integer(receipt["payload_bytes"])
        if (
            not (len(receipt["keys"]) - 1) * self.block_size
            < size
            <= len(receipt["keys"]) * self.block_size
        ):
            raise DiagnosticError("snapshot receipt has inconsistent length")

    def restore_disk(self, receipt: dict, output: Path) -> dict:
        import zstandard

        self._validate_receipt(receipt)
        # Hold one shared lock over the entire read: a new head cannot collect
        # some of these blocks halfway through the restored frame.
        try:
            payload = b"".join(self.store.read_many(receipt["keys"], self.block_size))
        except zstandard.ZstdError as exc:
            raise DiagnosticError("compressed snapshot failed integrity verification") from exc
        data = payload[: receipt["payload_bytes"]]
        if any(payload[receipt["payload_bytes"] :]):
            raise DiagnosticError("snapshot padding changed")
        if hashlib.sha256(data).hexdigest() != receipt["payload_sha256"]:
            raise DiagnosticError("snapshot receipt does not match restored bytes")
        result = unpack_frame(data, output)
        if result["sha256"] != receipt["frame_sha256"]:
            raise DiagnosticError("snapshot receipt does not match restored frame")
        return result

    def evict_ram(self, source: Path) -> dict:
        receipt = self.save_disk(source)
        self.ram = None  # publication must succeed before the last RAM copy goes away
        return receipt


class RecoveryCapture:
    """Adapter-neutral, immutable capture points for real native lifecycle tests.

    A native producer supplies a complete frame at an already committed boundary.
    This class never invokes a GPU operation or tries to guess a page layout.
    """

    def __init__(self, root: Path, *, contract: str, chat: str, generation: str, required):
        for value in (contract, chat, generation):
            require_sha(value)
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.root, self.required = root, list(required)
        self.identity = {"contract": contract, "chat": chat, "generation": generation}
        self.lock = threading.Lock()
        self.frames = {}

    def record(self, stage: str, source: Path, *, chat: str, generation: str, quiescent: bool):
        with self.lock:
            if stage not in STAGES or stage in self.frames:
                raise DiagnosticError("duplicate or unregistered recovery capture stage")
            if chat != self.identity["chat"] or generation != self.identity["generation"]:
                raise DiagnosticError("recovery crossed a chat or compaction generation")
            if quiescent is not True:
                raise DiagnosticError("full state capture needs a completed state transition")
            frame = read_frame(source)
            if frame["contract"] != self.identity["contract"]:
                raise DiagnosticError("recovery capture changed numerical contract")
            if not set(self.required).issubset(frame["coverage"]):
                raise DiagnosticError("recovery capture is missing required state")
            self.frames[stage] = archive_frame(source, self.root / stage)["sha256"]
            write_private(
                self.root / (stage + "-receipt.json"),
                seal({**self.identity, "stage": stage, "frame": self.frames[stage]}),
            )

    def compare(self):
        if set(self.frames) != set(STAGES):
            raise DiagnosticError(
                "recovery comparison missing a capture; never infer a clean state"
            )
        return compare_recovery(
            *(self.root / s for s in STAGES), self.root / "comparison", required=self.required
        )
