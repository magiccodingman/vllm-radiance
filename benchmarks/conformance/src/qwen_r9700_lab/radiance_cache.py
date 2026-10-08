"""Chat-owned, lossless Radiance snapshots. Also runs over SSH using only Python.

The model's cache salt isolates chats and compaction generations. The lock lives
outside the generations: retiring one waits for its readers/writers, and late
writes cannot recreate it. Legacy shared caches are never adopted or deleted.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ROOT = "/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1"
FORMAT = "qwen-chat-cache-v1"
HEADER = struct.Struct(">8sQ32s")
COMPRESSED = b"QWENKV1Z"
RAW = b"QWENKV1R"
ID = re.compile(r"[0-9a-f]{64}\Z")
KEY = re.compile(r"g[0-9]+-[0-9a-f]{16,128}\.qkv\Z")
CONTROL_DIRECTORY = Path("/dev/shm/qwen-radiance-snapshot-control-v1")
CONTROL_SCHEMA = "urn:qwen-r9700:radiance-snapshot-control:v1"
CONTROL_FILE = re.compile(r"([0-9a-f]{32})\.(request|response)\.json\Z")
_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))


class RetiredGenerationError(ValueError):
    """A valid chat identity names a generation durably superseded by compaction."""


class SnapshotIntegrityError(ValueError):
    """Stored bytes fail their own encoding or checksum, independently of the ABI."""


def identity(value: dict) -> dict:
    if not isinstance(value, dict) or any(
        not isinstance(value.get(k), str) or not ID.fullmatch(value[k])
        for k in ("id", "generation")
    ):
        raise ValueError("chat id and generation must be SHA256 identifiers")
    return {
        k: str(value.get(k, ""))[:4096]
        for k in ("id", "generation", "title", "cwd", "session_file")
    }


def cache_salt(chat: dict) -> str:
    chat = identity(chat)
    return f"{FORMAT}:{chat['id']}:{chat['generation']}"


def real_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"not a real directory: {path}")


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, content: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def private_control_directory(path: Path = CONTROL_DIRECTORY, *, create: bool = False) -> Path:
    """Return the owner-only tmpfs control directory used by the live backend."""
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("snapshot control directory must be private and owner-owned")
    return path


def request_tail_flush(
    chat: dict,
    *,
    timeout: float = 120.0,
    control_directory: Path = CONTROL_DIRECTORY,
) -> dict:
    """Ask the live backend to make one in-RAM tail durably publishable."""
    chat = identity(chat)
    if not 0 < timeout <= 600:
        raise ValueError("snapshot flush timeout must be between 0 and 600 seconds")
    directory = private_control_directory(Path(control_directory))
    nonce = uuid.uuid4().hex
    request = directory / f"{nonce}.request.json"
    response = directory / f"{nonce}.response.json"
    payload = {
        "schema": CONTROL_SCHEMA,
        "nonce": nonce,
        "action": "flush",
        "chat": chat,
    }
    atomic_write(request, json.dumps(payload, sort_keys=True).encode())
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                info = response.lstat()
                if response.is_symlink() or not response.is_file() or info.st_size > 65536:
                    raise ValueError("unsafe snapshot flush response")
                result = json.loads(response.read_text())
                if result.get("schema") != CONTROL_SCHEMA or result.get("nonce") != nonce:
                    raise ValueError("snapshot flush response identity mismatch")
                if result.get("status") == "error":
                    raise RuntimeError(result.get("error") or "backend snapshot flush failed")
                if result.get("status") not in ("flushed", "already_durable"):
                    raise RuntimeError("backend rejected the snapshot tail flush")
                return result
            except FileNotFoundError:
                time.sleep(0.05)
        raise TimeoutError("live backend did not complete the snapshot tail flush")
    finally:
        request.unlink(missing_ok=True)
        response.unlink(missing_ok=True)


def encode_block(data: memoryview | bytes) -> bytes:
    import zstandard

    compressed = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(data)
    magic, payload = (COMPRESSED, compressed) if len(compressed) < len(data) else (RAW, data)
    return HEADER.pack(magic, len(data), hashlib.sha256(data).digest()) + bytes(payload)


def decode_block(data: bytes, expected_size: int) -> bytes:
    import zstandard

    if len(data) < HEADER.size:
        raise SnapshotIntegrityError("truncated snapshot header")
    magic, size, digest = HEADER.unpack(data[: HEADER.size])
    if size != expected_size:
        raise ValueError("snapshot block size differs from the runtime")
    payload = data[HEADER.size :]
    if magic == COMPRESSED:
        # Check the frame size before allocating, even if its header is corrupt.
        try:
            if zstandard.frame_content_size(payload) != expected_size:
                raise SnapshotIntegrityError("snapshot frame size differs from its header")
            payload = zstandard.ZstdDecompressor().decompress(
                payload, max_output_size=expected_size, allow_extra_data=False
            )
        except zstandard.ZstdError as error:
            raise SnapshotIntegrityError("invalid compressed snapshot payload") from error
    elif magic != RAW:
        raise SnapshotIntegrityError("unknown snapshot encoding")
    if len(payload) != expected_size or hashlib.sha256(payload).digest() != digest:
        raise SnapshotIntegrityError("snapshot checksum mismatch")
    return payload


class ChatStore:
    def __init__(self, data_root: Path | str, chat: dict):
        self.chat = identity(chat)
        self.root = Path(data_root)
        real_directory(self.root)
        self.managed = self.root / FORMAT
        self.managed.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.managed)
        self.directory = self.managed / self.chat["id"]
        self.directory.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.directory)
        self.generations = self.directory / "generations"
        self.generations.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.generations)
        self.generation = self.generations / self.chat["generation"]
        self._verified = {}

    @contextlib.contextmanager
    def lock(self, *, exclusive: bool = False):
        fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            os.close(fd)

    def metadata(self) -> dict:
        path = self.directory / "chat.json"
        if path.is_symlink():
            raise ValueError("chat metadata is a symlink")
        return json.loads(path.read_text()) if path.exists() else {}

    def save_metadata(self, value: dict) -> None:
        value = {**value, "updated_at": datetime.now(UTC).isoformat()}
        atomic_write(self.directory / "chat.json", json.dumps(value, sort_keys=True).encode())

    def current(self) -> bool:
        return self.metadata().get("generation") == self.chat["generation"]

    def activate(self) -> dict:
        """Retire old writers, retaining one complete fallback until publication."""
        with self.lock(exclusive=True):
            prior = self.metadata()
            retired = prior.get("retired_generations", [])
            if self.chat["generation"] in retired:
                raise RetiredGenerationError(
                    "stale request targets a retired compaction generation"
                )
            recreated = not self.generation.exists()
            self.generation.mkdir(exist_ok=True, mode=0o700)
            real_directory(self.generation)
            # Persist the new directory before the tombstone can retire its
            # predecessor. Retrying an interrupted activation also repairs it.
            sync_directory(self.generations)
            if prior.get("generation") != self.chat["generation"]:
                if prior.get("generation"):
                    retired = [*retired, prior["generation"]]
                fallback = prior.get("fallback")
                if prior.get("head"):
                    fallback = {k: prior[k] for k in ("generation", "head", "tokens")}
                # The durable tombstone precedes deletion. Retrying after a crash
                # finishes collection; old queued writers see current() == False.
                self.save_metadata(
                    {
                        **self.chat,
                        "format": FORMAT,
                        "retired_generations": retired,
                        "status": "empty",
                        "tokens": 0,
                        "head": [],
                        "fallback": fallback,
                    }
                )
            else:
                if recreated:
                    prior = {**prior, "status": "empty", "tokens": 0, "head": []}
                self.save_metadata({**prior, **self.chat})
            # An exclusive lock proves no live writer owns these temporary files.
            for temporary in self.generation.glob(".pending-*"):
                temporary.unlink()
            info = self.metadata()
            fallback = info.get("fallback") or {}
            removed = retained = 0
            for path in self.generations.iterdir():
                if path.name == self.chat["generation"]:
                    continue
                if not ID.fullmatch(path.name):
                    raise ValueError(f"unexpected generation path: {path}")
                real_directory(path)
                if path.name == fallback.get("generation"):
                    keep = set(fallback["head"])
                    for block in path.iterdir():
                        if block.name in keep:
                            retained += block.lstat().st_size
                        elif KEY.fullmatch(block.name) or block.name.startswith(".pending-"):
                            removed += block.lstat().st_size
                            block.unlink()
                    sync_directory(path)
                    continue
                removed += sum(p.stat().st_size for p in path.iterdir() if p.is_file())
                shutil.rmtree(path)
            sync_directory(self.generations)
            return {
                "chat_id": self.chat["id"],
                "removed_file_bytes": removed,
                "retained_file_bytes": retained,
                "generation": self.chat["generation"],
            }

    def path(self, key: str) -> Path:
        if not KEY.fullmatch(key):
            raise ValueError("invalid snapshot object key")
        real_directory(self.generation)
        path = self.generation / key
        if path.is_symlink():
            raise ValueError("snapshot object is a symlink")
        return path

    def exists(self, key: str) -> bool:
        return self.exists_many([key])[0]

    def exists_many(self, keys: list[str]) -> list[bool]:
        # A hybrid prefix lookup can inspect hundreds of keys. Read the chat
        # generation once for the batch, rather than reopening its manifest for
        # each key on the scheduler thread.
        with self.lock():
            if not self.current():
                return [False] * len(keys)
            return [self.path(key).is_file() for key in keys]

    def io_totals(self) -> dict:
        path = self.directory / "io.json"
        if path.is_symlink():
            raise ValueError("snapshot I/O counters are a symlink")
        return json.loads(path.read_text()) if path.exists() else {"available": False}

    def _record_io(self, changes: dict) -> None:
        """Persist one counter update per transfer batch, outside generations.

        These are completed snapshot payload writes, including objects later
        collected. They exclude metadata and failed partial writes; device
        counters account for those too. A crash can lose the unfinished batch's
        counters, so this is a lower bound rather than a NAND wear estimate.
        """
        if not any(changes.values()):
            return
        fd = os.open(self.directory / ".io.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            prior = self.io_totals()
            now = datetime.now(UTC).isoformat()
            info = {
                **prior,
                "available": True,
                "since": prior.get("since", now),
                "updated_at": now,
                "scope": "completed_snapshot_payload_io_lower_bound",
            }
            for name, amount in changes.items():
                info[name] = prior.get(name, 0) + amount
            atomic_write(self.directory / "io.json", json.dumps(info, sort_keys=True).encode())
        finally:
            os.close(fd)

    def write(self, key: str, data: memoryview) -> bool:
        return self.write_many([(key, data)])

    def write_many(self, blocks) -> bool:
        # Compression holds the read lock too: compaction cannot return before
        # all old writes have stopped, and a late writer cannot recreate a folder.
        with self.lock():
            if not self.current():
                return False
            counts = {
                "written_file_bytes": 0,
                "written_raw_bytes": 0,
                "written_blocks": 0,
                "reused_blocks": 0,
                "compression_seconds": 0.0,
                "write_failures": 0,
            }
            try:
                for key, data in blocks:
                    path = self.path(key)
                    # The engine has one process but several filesystem workers.
                    # Coalesce simultaneous requests for the same immutable key.
                    with _WRITE_LOCKS[hash(str(path)) % len(_WRITE_LOCKS)]:
                        if path.exists():
                            counts["reused_blocks"] += 1
                            continue
                        started = time.monotonic()
                        encoded = encode_block(data)
                        counts["compression_seconds"] += time.monotonic() - started
                        atomic_write(path, encoded)
                        counts["written_file_bytes"] += len(encoded)
                        counts["written_raw_bytes"] += len(data)
                        counts["written_blocks"] += 1
            except Exception:
                counts["write_failures"] += 1
                raise
            finally:
                self._record_io(counts)
            return True

    def read(self, key: str, expected_size: int) -> bytes:
        with contextlib.closing(self.read_many([key], expected_size)) as blocks:
            return next(blocks)

    def read_many(self, keys: list[str], expected_size: int):
        with self.lock():
            if not self.current():
                raise ValueError("snapshot generation was retired")
            counts = {
                "read_file_bytes": 0,
                "read_raw_bytes": 0,
                "read_blocks": 0,
                "read_failures": 0,
                "invalidated_blocks": 0,
                "invalidated_file_bytes": 0,
            }
            try:
                for key in keys:
                    path = self.path(key)
                    # Exclude a simultaneous repair of this immutable key. A
                    # reader must not remove a newer valid replacement after
                    # detecting damage in the old file. Release before yield.
                    with _WRITE_LOCKS[hash(str(path)) % len(_WRITE_LOCKS)]:
                        encoded = path.read_bytes()
                        try:
                            data = decode_block(encoded, expected_size)
                        except SnapshotIntegrityError:
                            # A confirmed corrupt object is not a usable disk
                            # head. Make it a miss so recomputed data can replace
                            # it; preserve valid files on ABI or I/O errors.
                            path.unlink()
                            sync_directory(path.parent)
                            counts["invalidated_blocks"] += 1
                            counts["invalidated_file_bytes"] += len(encoded)
                            raise
                    counts["read_file_bytes"] += len(encoded)
                    counts["read_raw_bytes"] += len(data)
                    counts["read_blocks"] += 1
                    yield data
            except Exception:
                counts["read_failures"] += 1
                raise
            finally:
                self._record_io(counts)

    @staticmethod
    def _fingerprint(path: Path) -> list[int]:
        value = path.stat()
        return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    def prepare_publication(self, keys: list[str], block_size: int) -> dict:
        """Verify new/changed payloads away from the scheduler's critical path.

        Published immutable files retain their verified fingerprints. Normal
        continuations verify only newly written blocks, not the whole prefix.
        """
        import zstandard

        with self.lock():
            prior = self.metadata()
            known = (
                prior.get("verified_head", {})
                if prior.get("verified_block_size") == block_size
                else {}
            )
            result = {"keys": sorted(set(keys)), "missing": [], "invalid": [], "verified": {}}
            if prior.get("generation") != self.chat["generation"]:
                result["missing"] = result["keys"]
                return result
            verified_bytes = 0
            for key in result["keys"]:
                path = self.path(key)
                try:
                    before = self._fingerprint(path)
                    if known.get(key) != before and self._verified.get((key, block_size)) != before:
                        encoded = path.read_bytes()
                        decode_block(encoded, block_size)
                        verified_bytes += len(encoded)
                        if before != self._fingerprint(path):
                            raise ValueError("snapshot changed during verification")
                    result["verified"][key] = before
                    self._verified[key, block_size] = before
                except FileNotFoundError:
                    result["missing"].append(key)
                except (OSError, ValueError, zstandard.ZstdError):
                    result["invalid"].append(key)
            self._record_io({"verification_file_bytes": verified_bytes})
            return result

    def publish(self, keys: list[str], tokens: int, block_size: int, *, prepared=None) -> bool:
        """Commit a complete head or roll back its abandoned writes, then collect.

        The caller must first drain every request and disk job for this chat.
        A rejected successor cannot be repaired after those jobs have finished;
        retain the previous head and discard the failed candidate instead.
        """
        if prepared is None:
            with self.lock():
                if not self.current():
                    return False
            prepared = self.prepare_publication(keys, block_size)
        with self.lock(exclusive=True):
            if not self.current():
                return False
            prior = self.metadata()
            keys = sorted(set(keys))
            if keys != prepared["keys"]:
                raise ValueError("verified snapshot differs from publication candidate")
            missing, invalid = list(prepared["missing"]), list(prepared["invalid"])
            for key, expected in prepared["verified"].items():
                try:
                    if self._fingerprint(self.path(key)) != expected:
                        invalid.append(key)
                except FileNotFoundError:
                    missing.append(key)
                except OSError:
                    invalid.append(key)
            complete = not missing and not invalid
            info = {
                **prior,
                "status": "ready" if complete else "incomplete",
                "publication": {
                    "tokens": tokens,
                    "expected_blocks": len(keys),
                    "missing_keys": missing,
                    "invalid_keys": invalid,
                    "result": "committed" if complete else "rolled_back",
                },
            }
            if complete:
                info.update(
                    head=keys,
                    tokens=tokens,
                    verified_head=prepared["verified"],
                    verified_block_size=block_size,
                    fallback=None,
                )
                self._verified = {
                    (key, block_size): value for key, value in prepared["verified"].items()
                }
            self._collect(info)
            return complete

    def _collect(self, info: dict) -> dict:
        """Under the exclusive chat lock, persist intent before deleting anything."""
        keep = set(info["head"])
        if any(not KEY.fullmatch(key) for key in keep):
            raise ValueError("invalid published head manifest")
        info = {**info, "gc": {"status": "pending"}}
        self.save_metadata(info)
        removed_files = removed_bytes = 0
        try:
            for path in self.generation.iterdir():
                if path.name not in keep and (
                    KEY.fullmatch(path.name) or path.name.startswith(".pending-")
                ):
                    size = path.lstat().st_size
                    path.unlink()
                    removed_files += 1
                    removed_bytes += size
            fallback_generation = (info.get("fallback") or {}).get("generation")
            for directory in self.generations.iterdir():
                if directory.name in {self.chat["generation"], fallback_generation}:
                    continue
                if not ID.fullmatch(directory.name):
                    raise ValueError("invalid snapshot generation directory")
                real_directory(directory)
                for path in directory.iterdir():
                    removed_files += 1
                    removed_bytes += path.lstat().st_size
                shutil.rmtree(directory)
            sync_directory(self.generation)
            sync_directory(self.generations)
        except OSError as error:
            # If even this write fails, the durable pending record still causes
            # startup recovery. Never report a completed GC before directory fsync.
            self.save_metadata({**info, "gc": {"status": "failed", "errno": error.errno}})
            raise
        result = {
            "status": "complete",
            "removed_files": removed_files,
            "removed_file_bytes": removed_bytes,
        }
        self.save_metadata({**info, "gc": result})
        return result

    def collect(self, *, failure: str | None = None) -> dict:
        """Recover abandoned writes with no active requests, preserving the head."""
        with self.lock(exclusive=True):
            if not self.current():
                return {"status": "retired"}
            info = self.metadata()
            if failure is not None:
                info = {**info, "status": "incomplete", "publication": {"result": failure}}
            return self._collect(info)


def report(data_root: Path) -> dict:
    chats = []
    managed = data_root / FORMAT
    if managed.exists():
        real_directory(managed)
        for directory in sorted(managed.iterdir()):
            if not ID.fullmatch(directory.name):
                continue
            real_directory(directory)
            metadata_path = directory / "chat.json"
            if not metadata_path.exists():
                continue
            info = json.loads(metadata_path.read_text())
            if info.get("id") != directory.name:
                raise ValueError("chat metadata identity differs from its directory")
            store = ChatStore(data_root, info)
            with store.lock():
                info = store.metadata()
                sizes = {"files": 0, "file_bytes": 0, "allocated_bytes": 0, "raw_bytes": 0}
                for path in store.generations.rglob("*.qkv"):
                    if path.is_symlink():
                        raise ValueError(f"snapshot object is a symlink: {path}")
                    stat = path.stat()
                    with path.open("rb") as stream:
                        header = stream.read(HEADER.size)
                    sizes["files"] += 1
                    sizes["file_bytes"] += stat.st_size
                    sizes["allocated_bytes"] += stat.st_blocks * 512
                    if len(header) == HEADER.size:
                        sizes["raw_bytes"] += HEADER.unpack(header)[1]
                chats.append(
                    {
                        **{
                            k: v
                            for k, v in info.items()
                            if k not in ("head", "retired_generations")
                        },
                        **sizes,
                    }
                )
    legacy_files = legacy_bytes = 0
    for path in data_root.rglob("*.bin"):
        if path.is_symlink():
            continue
        legacy_files += 1
        legacy_bytes += path.stat().st_size
    return {
        "data_root": str(data_root),
        "chats": chats,
        "legacy_unassigned_files": legacy_files,
        "legacy_unassigned_bytes": legacy_bytes,
    }


def human(size: int) -> str:
    return f"{size / 1024**3:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="ai")
    parser.add_argument("--cache-root", default=DEFAULT_ROOT)
    parser.add_argument("--abi", help="snapshot ABI directory; required for mutation commands")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser(
        "list", help="list chats, compressed sizes, and unassigned legacy cache"
    )
    listing.add_argument("--json", action="store_true")
    compact = sub.add_parser("compact", help="retire a chat generation after Pi commits compaction")
    compact.add_argument("--identity-json", required=True)
    flush = sub.add_parser("flush", help="force the live backend to publish a chat's buffered tail")
    flush.add_argument("--identity-json", required=True)
    flush.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.host not in ("local", "localhost", "127.0.0.1"):
        remote_args = ["--host", "local", "--cache-root", args.cache_root]
        if args.abi:
            remote_args += ["--abi", args.abi]
        remote_args += [args.command]
        remote_args += ["--json"] if args.command == "list" and args.json else []
        if args.command in ("compact", "flush"):
            remote_args += ["--identity-json", args.identity_json]
        if args.command == "flush":
            remote_args += ["--timeout", str(args.timeout)]
        return subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                args.host,
                shlex.join(["python3", "-", *remote_args]),
            ],
            input=Path(__file__).read_text(),
            text=True,
            check=False,
        ).returncode
    snapshots = Path(args.cache_root).expanduser() / "snapshots"
    if args.abi and not ID.fullmatch(args.abi):
        parser.error("--abi must be a SHA256 identifier")
    if args.command in ("compact", "flush") and not args.abi:
        parser.error(f"{args.command} requires --abi")
    if args.command == "flush":
        print(json.dumps(request_tail_flush(json.loads(args.identity_json), timeout=args.timeout)))
        return 0
    if args.command == "compact":
        result = ChatStore(snapshots / args.abi / "data", json.loads(args.identity_json)).activate()
        print(json.dumps(result))
        return 0
    roots = [snapshots / args.abi] if args.abi else sorted(snapshots.iterdir())
    reports = [report(root / "data") for root in roots if (root / "data").is_dir()]
    if args.json:
        print(json.dumps(reports, indent=2))
        return 0
    for value in reports:
        print(value["data_root"])
        print(
            f"{'CHAT':12}  {'DISK FILES':>12}  {'RAW CACHE':>12}  "
            f"{'TOKENS':>8}  {'STATE':10}  TITLE / DIRECTORY"
        )
        for chat in value["chats"]:
            print(
                f"{chat['id'][:12]}  {human(chat['file_bytes']):>12}  "
                f"{human(chat['raw_bytes']):>12}  "
                f"{chat.get('tokens', 0):>8}  {chat.get('status', 'unknown'):10}  "
                f"{chat.get('title') or chat.get('cwd')}"
            )
        print(
            f"Unassigned legacy cache: {human(value['legacy_unassigned_bytes'])} "
            f"({value['legacy_unassigned_files']} files; preserved)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
