"""Own one test engine's named offload mapping across process teardown.

The runtime can leave its mmap name behind after an abort/crash. Unlink only
the fresh, explicitly claimed conformance engine name, after its owned process
has been stopped. Unlinking does not change any still-open mapping: the OS
reclaims those pages when the last reference closes. Never scan or clear other
engines' shared memory, and never apply this to production engine identities.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class OwnedOffloadRegion:
    def __init__(self, engine_id: str, *, directory=Path("/dev/shm")):
        if not isinstance(engine_id, str) or not re.fullmatch(
            r"conformance-[0-9a-f]{64}", engine_id
        ):
            raise DiagnosticError(
                "offload cleanup requires an isolated conformance engine identity"
            )
        self.engine_id = engine_id
        self.name = f"vllm_offload_{engine_id}.mmap"
        self.path = Path(directory) / self.name
        self.fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            try:
                os.stat(self.name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise DiagnosticError("qualification offload name already exists; not owned")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def release(self):
        """Release the name after process close; report unlink, not proven page reclamation."""
        if self.fd is None:
            raise DiagnosticError("offload ownership was already released")
        try:
            try:
                info = os.stat(self.name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                return {"status": "absent", "engine_id": self.engine_id, "path": str(self.path)}
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise DiagnosticError("qualification offload file type or ownership changed")
            os.unlink(self.name, dir_fd=self.fd)
            return {
                "status": "unlinked",
                "engine_id": self.engine_id,
                "path": str(self.path),
                "device": info.st_dev,
                "inode": info.st_ino,
                "bytes": info.st_size,
                "allocated_bytes_before_unlink": info.st_blocks * 512,
                "page_reclamation": "when the last mapping/file reference closes",
            }
        finally:
            os.close(self.fd)
            self.fd = None
