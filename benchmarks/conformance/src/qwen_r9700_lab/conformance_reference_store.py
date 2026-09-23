"""One-slot store for independently retained serial qualification baselines.

This owns disposable copies only. Native execution/configuration admission is
the caller's responsibility and must be included in the sealed identity. Every
hit is projected and byte-authenticated again. An interrupted publication or
retirement is an explicit error requiring review, never an implicit cache hit.
"""

import fcntl
import hashlib
import os
import re
import shutil
import stat
import uuid
from pathlib import Path

from qwen_r9700_lab.conformance_native_reference import (
    compatible_serial_plans,
    project_serial_reference,
)
from qwen_r9700_lab.conformance_state import compare_frames, private_directory
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    private_json,
    require_sha,
    seal,
    write_private,
)

SCHEMA = "urn:qwen:serial-reference-store:v1"


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sealed_file(path):
    value = private_json(path)
    authenticate(value)
    return value


def copied_tree_identity(root):
    """Bind a case-local source/JIT tree by all pre-execution bytes and modes.

    Absolute relocation and copy timestamps are not semantic inputs. Symlinks,
    special files, empty trees and concurrent changes are refused, not omitted.
    This is an artifact identity, not a proof about future compiler output.
    """
    from qwen_r9700_lab.conformance_artifacts import file_identity

    root = Path(root)

    def inventory():
        entries = {}
        for path in [root, *sorted(root.rglob("*"))]:
            info = path.lstat()
            require(
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                "copied runtime tree contains a symlink or special file",
            )
            require(info.st_uid == os.getuid(), "copied runtime tree must be owned")
            entries[str(path.relative_to(root))] = tuple(
                getattr(info, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
        require(stat.S_ISDIR(entries["."][2]), "copied runtime tree is not a directory")
        return entries

    before = inventory()
    records = {}
    files = 0
    for relative, metadata in before.items():
        mode = metadata[2]
        records[relative] = {"mode": mode}
        if stat.S_ISREG(mode):
            records[relative].update(file_identity(root / relative))
            files += 1
    require(files > 0, "copied runtime tree is empty")
    require(inventory() == before, "copied runtime tree changed during hashing")
    return digest({"schema": SCHEMA + "/copied-tree", "entries": records})


class SerialReferenceStore:
    """Use under one explicit lock; retire before populating another baseline.

    Keeping replacement population outside this store avoids an unaccounted
    second retained baseline. The producer's complete original capture remains
    in its case for normal independent archival, even when that D7 case fails.
    """

    def __init__(self, root, *, reflink=True):
        self.root = Path(root).absolute()
        self.reflink = reflink
        self.fd = None

    def __enter__(self):
        require(self.fd is None, "reference store is already locked")
        try:
            self.root.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            created = False
        private_directory(self.root)
        self.fd = os.open(self.root / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if created:
                write_private(self.root / "store.json", seal({"schema": SCHEMA}))
                (self.root / "retired").mkdir(mode=0o700)
                sync(self.root)
                sync(self.root.parent)
            marker = sealed_file(self.root / "store.json")
            require(marker == seal({"schema": SCHEMA}), "unrecognized reference store")
            private_directory(self.root / "retired")
            require(
                {p.name for p in self.root.iterdir()}
                <= {"lock", "store.json", "retired", "current"},
                "unfinished or unrecognized reference-store contents; evidence retained",
            )
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _locked(self):
        require(self.fd is not None, "reference store operation requires its lock")
        require(not (self.root / "pending").exists(), "unfinished reference publication retained")

    def _key(self, identity, plan):
        authenticate(identity)
        require(
            set(identity) == {"schema", "campaign", "configuration", "environment", "sha256"}
            and identity["schema"] == SCHEMA + "/identity"
            and isinstance(identity["configuration"], dict)
            and bool(identity["configuration"])
            and isinstance(identity["environment"], dict)
            and bool(identity["environment"]),
            "reference identity requires campaign, effective configuration and environment",
        )
        require_sha(identity["campaign"])
        compatible_serial_plans(plan, plan)
        return digest({"identity": identity["sha256"], "plan": plan["sha256"]})

    def _entry(self):
        self._locked()
        current = self.root / "current"
        if not current.exists() and not current.is_symlink():
            return None
        private_directory(current)
        entry = sealed_file(current / "entry.json")
        require(entry.get("schema") == SCHEMA + "/entry", "invalid reference entry")
        require(
            re.fullmatch(r"[0-9a-f]{32}", entry.get("publication_id", "")) is not None
            and entry["identity"]["campaign"] == entry["campaign"],
            "reference publication identity changed",
        )
        require(
            entry["key"] == self._key(entry["identity"], entry["plan"]),
            "reference entry binding changed",
        )
        projection = sealed_file(current / "capture/reference-projection.json")
        schedule = sealed_file(current / "capture/schedule.json")
        boundaries = sealed_file(current / "capture/boundaries/boundaries.json")
        require(
            projection["sha256"] == entry["projection"]
            and projection["source_plan"] == projection["requested_plan"] == entry["plan"]["sha256"]
            and projection["schedule"] == schedule["sha256"]
            and projection["boundaries"] == boundaries["sha256"]
            and bool(schedule["frames"])
            and bool(boundaries["frames"]),
            "reference entry completion changed",
        )
        return entry

    def project(self, identity, plan, requested, output):
        """Return None for a different identity; corrupt matching entries raise."""
        key = self._key(identity, plan)
        entry = self._entry()
        if entry is None or entry["key"] != key:
            return None
        output = Path(output).absolute()
        require(not output.resolve().is_relative_to(self.root.resolve()), "output is inside store")
        result = project_serial_reference(
            plan, requested, self.root / "current/capture", output, reflink=self.reflink
        )
        write_private(output / "store-origin.json", entry)
        return result

    def publish(self, identity, plan, source, origin):
        """Clone a complete capture from a case, preserving its original there."""
        key = self._key(identity, plan)
        require(self._entry() is None, "retire the previous reference before replacement")
        source, origin = Path(source).resolve(strict=True), Path(origin).resolve(strict=True)
        require(
            re.fullmatch(r"case-\d{5}(?:-attempt-\d{3,})?", origin.name) is not None
            and source.is_relative_to(origin)
            and not source.is_relative_to(self.root.resolve())
            and not self.root.resolve().is_relative_to(origin),
            "reference origin must be a separate qualification case",
        )
        private_directory(origin)
        payload = private_json(origin / "input.json")
        authenticate(payload["campaign"])
        require(payload["case"] in payload["campaign"]["cases"], "origin case is not declared")
        require(
            identity["campaign"] == payload["campaign"]["sha256"],
            "reference producer belongs to another campaign",
        )
        origin_input = hashlib.sha256((origin / "input.json").read_bytes()).hexdigest()
        pending = self.root / "pending"
        pending.mkdir(mode=0o700)
        # A complete full-domain projection authenticates every stored tensor.
        projection = project_serial_reference(
            plan, plan, source, pending / "capture", reflink=self.reflink
        )
        entry = seal(
            {
                "schema": SCHEMA + "/entry",
                "publication_id": uuid.uuid4().hex,
                "key": key,
                "identity": identity,
                "plan": plan,
                "origin": str(origin),
                "capture_relative": str(source.relative_to(origin)),
                "origin_input_sha256": origin_input,
                "campaign": payload["campaign"]["sha256"],
                "case": payload["case"],
                "projection": projection["sha256"],
            }
        )
        write_private(pending / "entry.json", entry)
        sync(pending)
        pending.rename(self.root / "current")
        sync(self.root)
        return entry

    def _retained(self, entry):
        origin = Path(entry["origin"])
        private_directory(origin)
        private_json(origin / "input.json")
        require(
            hashlib.sha256((origin / "input.json").read_bytes()).hexdigest()
            == entry["origin_input_sha256"],
            "reference origin input changed",
        )
        locator = origin / "archive-locator.json"
        if locator.exists():
            archived = private_json(locator)
            retired = private_json(origin / "archive-retirement.json")
            result = sealed_file(origin / "result.json")
            identity = archived.get("identity", {})
            require(
                archived.get("schema") == "urn:qwen:conformance-archive-locator:v1"
                and archived.get("archive_path") == f"/{origin.name}.tar"
                and identity.get("scope") == "finished-case-v1"
                and identity.get("input_sha256") == entry["origin_input_sha256"]
                and identity.get("campaign") == result.get("campaign") == entry["campaign"]
                and identity.get("result") == result["sha256"]
                and result.get("case") == entry["case"]
                and all(retired.get(k) == v for k, v in archived.items())
                and retired.get("metadata_retained") is True,
                "reference archive retention is not established",
            )
            for field in ("snapshot_id", "tar_sha256", "manifest_sha256"):
                require_sha(archived[field])
            # The separate archive writer verifies every tar member before this
            # receipt. This store does not claim a second remote archive readback.
            return {"kind": "verified-case-archive", "locator": archived}
        source = origin / entry["capture_relative"]
        cached = self.root / "current/capture"
        for relative in ("schedule.json", "boundaries/boundaries.json"):
            schedule = sealed_file(cached / relative)
            prefix = Path() if relative == "schedule.json" else Path("boundaries")
            for frame in schedule["frames"]:
                report = compare_frames(
                    source / prefix / frame["name"], cached / prefix / frame["name"]
                )
                require(report["equal"], "retained original no longer matches cached reference")
        return {"kind": "verified-original-capture", "capture": str(source)}

    def retire(self):
        """Remove only this store's copy after establishing retained evidence."""
        entry = self._entry()
        if entry is None:
            return None
        retained = self._retained(entry)
        receipt = seal({"schema": SCHEMA + "/retirement", "entry": entry, "retained": retained})
        write_private(self.root / "retired" / (entry["sha256"] + ".json"), receipt)
        sync(self.root / "retired")
        shutil.rmtree(self.root / "current")
        sync(self.root)
        return receipt
