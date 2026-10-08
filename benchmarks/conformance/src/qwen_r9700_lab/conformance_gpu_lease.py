"""Cooperative GPU ownership for concurrent qualification controllers.

The optional lock serializes GPU phases, allowing independent CPU reference
preparation to overlap. It does not claim to exclude unrelated GPU programs.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
import time
from pathlib import Path

from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)

_FAILED_CLEANUP = {}


def cleanup_block():
    """A released flock is insufficient when an owned worker could not exit."""
    name = os.environ.get("QWEN_CONFORMANCE_GPU_LOCK")
    if name is None:
        return None
    path = Path(name)
    if not name or not path.is_absolute():
        raise DiagnosticError("qualification GPU lock must be an explicit absolute path")
    if name in _FAILED_CLEANUP:
        return _FAILED_CLEANUP[name]
    marker = path.with_name(path.name + ".blocked.json")
    if not marker.exists():
        return None
    receipt = private_json(marker)
    authenticate(receipt)
    return receipt


def block_cleanup(evidence, *, process_group, reason, members):
    receipt = seal(
        {
            "status": "cleanup_incomplete",
            "evidence": str(evidence),
            "process_group": process_group,
            "reason": reason,
            "members": members,
            "observed_ns": time.time_ns(),
        }
    )
    name = os.environ.get("QWEN_CONFORMANCE_GPU_LOCK")
    if name is not None:
        # Do not report a clean release if writing the diagnostic itself fails.
        _FAILED_CLEANUP[name] = receipt
        path = Path(name)
        if not path.is_absolute():
            raise DiagnosticError("qualification GPU lock must be absolute")
        replace_private(path.parent, path.name + ".blocked.json", receipt)
    write_private(Path(evidence) / "cleanup-incomplete.json", receipt)


@contextlib.contextmanager
def gpu_lease(evidence: Path):
    name = os.environ.get("QWEN_CONFORMANCE_GPU_LOCK")
    if name is None:
        yield
        return
    path = Path(name)
    if not name or not path.is_absolute():
        raise DiagnosticError("qualification GPU lock must be an explicit absolute path")
    evidence.mkdir(mode=0o700)
    requested = time.monotonic()
    write_private(
        evidence / "requested.json",
        seal({"pid": os.getpid(), "lock": str(path), "requested_ns": time.time_ns()}),
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DiagnosticError("qualification GPU lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if cleanup_block() is not None:
            raise DiagnosticError(
                "qualification worker cleanup is incomplete; GPU admission blocked"
            )
        owner_name = path.name + ".owner.json"
        owner_path = path.with_name(owner_name)
        if owner_path.exists():
            previous = private_json(owner_path)
            authenticate(previous)
            if previous.get("status") != "released":
                raise DiagnosticError(
                    "previous qualification GPU owner did not release cleanly; "
                    "verify its process group has exited before clearing the lease"
                )
        owner = {"pid": os.getpid(), "evidence": str(evidence), "status": "active"}
        replace_private(path.parent, owner_name, seal(owner))
        write_private(
            evidence / "acquired.json",
            seal({"pid": os.getpid(), "wait_seconds": time.monotonic() - requested}),
        )
        try:
            yield
        finally:
            blocked = cleanup_block() is not None
            write_private(
                evidence / ("scope-cleanup-incomplete.json" if blocked else "released.json"),
                seal(
                    {
                        "pid": os.getpid(),
                        "blocked_ns" if blocked else "released_ns": time.time_ns(),
                    }
                ),
            )
            replace_private(
                path.parent,
                owner_name,
                seal({**owner, "status": "cleanup_incomplete" if blocked else "released"}),
            )
    finally:
        # A killed worker leaves an active owner receipt. The next controller
        # fails closed even if the kernel lock has gone away, since GPU child
        # processes may still require cleanup.
        os.close(descriptor)
