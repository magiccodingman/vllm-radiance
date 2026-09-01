#!/usr/bin/env python3
"""Backport upstream vLLM #52596's leak-proof CPU offload region lifecycle.

Upstream commit 4c58a0c398b056b135b98bd93c644945be7e3109 adds a
worker rendezvous after every rank maps the shared CPU KV region, then unlinks
the pathname while keeping each mapping alive.  This is the standard POSIX
shared-memory lifetime: the kernel reclaims the pages when the last mapping
closes, including after SIGKILL, so restarts cannot accumulate 24+ GiB orphan
files in /dev/shm.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])


def patch_shared_region() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/shared_offload_region.py"
    apply(
        path,
        """    File path: /dev/shm/vllm_offload_{engine_id}.mmap
""",
        """    File path: /dev/shm/vllm_offload_{engine_id}.mmap.  When a barrier is
    given, the path is unlinked once every worker has mapped the file, so
    the kernel reclaims the memory when the last worker exits, no matter
    how it exits; mappings taken before the unlink stay valid.
""",
        "the path is unlinked once every worker has mapped the file",
        "kv-offload lifecycle: document unlink-after-rendezvous",
    )
    apply(
        path,
        """        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
""",
        """        kv_bytes_per_block: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
    ) -> None:
""",
        "barrier: Callable[[], None] | None = None",
        "kv-offload lifecycle: accept worker barrier",
    )

    old = '''        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
        except FileExistsError:
            # Joiner path — another worker won O_EXCL. Reopen and wait
            # for the file to reach expected size.
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            try:
                _wait_for_file_size(self.fd, self.total_size_bytes)
            except (TimeoutError, OSError):
                os.close(self.fd)
                raise
            logger.info("Opened existing mmap file %s", self.mmap_path)
        else:
            # Creator path. We won O_EXCL, so we own the file: any
            # failure here must clean up so concurrent joiners don't
            # land on a 0-byte stub and spin in _wait_for_file_size
            # for the full 30 s timeout.
            try:
                check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)
            except (RuntimeError, OSError):
                os.unlink(self.mmap_path)
                os.close(self.fd)
                raise
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
'''
    new = '''        try:
            try:
                self.fd: int | None = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                try:
                    _wait_for_file_size(self.fd, self.total_size_bytes)
                except (TimeoutError, OSError):
                    os.close(self.fd)
                    raise
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                try:
                    check_shm_free_space(self.total_size_bytes)
                    os.ftruncate(self.fd, self.total_size_bytes)
                except (RuntimeError, OSError):
                    os.unlink(self.mmap_path)
                    os.close(self.fd)
                    raise
                self._creator = True
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )

            self.mmap_obj: mmap.mmap | None = mmap.mmap(
                self.fd,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except Exception:
            if self._creator:
                os.unlink(self.mmap_path)
                self._creator = False
            # Peers block inside the barrier until the collective times out if
            # we die before reaching it. Arrive anyway so every worker calls
            # barrier() exactly once and preserves its own setup error.
            if barrier is not None:
                try:
                    barrier()
                except Exception:
                    logger.warning(
                        "Failed to release peers waiting at the mmap barrier",
                        exc_info=True,
                    )
            raise

        if barrier is not None:
            # Every worker has mapped the file once the barrier releases, so
            # dropping its name makes every exit path, including SIGKILL,
            # leak-proof while the existing mappings remain valid.
            try:
                barrier()
            except Exception:
                if self._creator:
                    os.unlink(self.mmap_path)
                    self._creator = False
                self.mmap_obj.close()
                os.close(self.fd)
                raise
            if self._creator:
                os.unlink(self.mmap_path)
                self._creator = False
                logger.info("Unlinked mmap file %s", self.mmap_path)
'''
    apply(
        path,
        old,
        new,
        "Failed to release peers waiting at the mmap barrier",
        "kv-offload lifecycle: unlink after all workers map",
    )


def patch_cpu_spec() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/spec.py"
    apply(
        path,
        """from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


class CPUOffloadingSpec(OffloadingSpec):
""",
        """from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _all_workers_barrier() -> None:
    \"\"\"Block until every worker rank reaches this point (gloo CPU group).

    A superset of node-local mmap openers suffices: once the barrier releases,
    every worker sharing the region file has mapped it.
    \"\"\"
    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
    )

    try:
        group = get_inner_dp_world_group()
    except AssertionError:
        group = get_world_group()
    group.barrier()


class CPUOffloadingSpec(OffloadingSpec):
""",
        "def _all_workers_barrier() -> None:",
        "kv-offload lifecycle: define worker rendezvous",
    )
    apply(
        path,
        """                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
""",
        """                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                barrier=_all_workers_barrier,
            )
""",
        "barrier=_all_workers_barrier",
        "kv-offload lifecycle: rendezvous every worker",
    )


patch_shared_region()
patch_cpu_spec()
