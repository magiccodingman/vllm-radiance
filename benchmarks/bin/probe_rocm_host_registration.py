#!/usr/bin/env python3
"""Destructive-to-capacity (not data) ROCm host-registration probe.

This allocates and pre-faults a large shared-memory mmap, maps it in one process
per GPU, and measures whole/chunked ``hipHostRegister`` behavior under serial
and simultaneous TP-like registration.  It must only run in a declared GPU
maintenance window.  The production API guard is intentionally fail-closed.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import multiprocessing as mp
import os
import queue
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from radiance_kv_offload import register_host_chunks, rollback_host_chunks


@dataclass
class WorkerResult:
    rank: int
    gpu: int
    register_ok: bool
    register_error_code: int | None
    drained_error_code: int | None
    chunks: list[tuple[int, int]]
    register_seconds: float
    post_register_runtime_ok: bool
    post_register_runtime_code: int
    cleanup_ok: bool
    exception: str | None = None


class HipRuntime:
    def __init__(self) -> None:
        path = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        self.lib = ctypes.CDLL(path)
        self.lib.hipSetDevice.argtypes = [ctypes.c_int]
        self.lib.hipSetDevice.restype = ctypes.c_int
        self.lib.hipGetLastError.argtypes = []
        self.lib.hipGetLastError.restype = ctypes.c_int
        self.lib.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.lib.hipHostRegister.restype = ctypes.c_int
        self.lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
        self.lib.hipHostUnregister.restype = ctypes.c_int
        self.lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.hipMalloc.restype = ctypes.c_int
        self.lib.hipFree.argtypes = [ctypes.c_void_p]
        self.lib.hipFree.restype = ctypes.c_int

    def set_device(self, device: int) -> int:
        return int(self.lib.hipSetDevice(device))

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int:
        return int(self.lib.hipHostRegister(ctypes.c_void_p(ptr), size, flags))

    def cudaHostUnregister(self, ptr: int) -> int:
        return int(self.lib.hipHostUnregister(ctypes.c_void_p(ptr)))

    def drain_pending_error(self) -> int:
        return int(self.lib.hipGetLastError())

    def allocation_smoke(self) -> int:
        ptr = ctypes.c_void_p()
        result = int(self.lib.hipMalloc(ctypes.byref(ptr), 4096))
        if result == 0:
            free_result = int(self.lib.hipFree(ptr))
            return free_result
        return result


def api_port_is_open(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def worker(
    rank: int,
    gpu: int,
    path: str,
    total_size: int,
    row_stride: int,
    chunk_bytes: int,
    mode: str,
    barrier: Any,
    result_queue: Any,
) -> None:
    mapped: mmap.mmap | None = None
    try:
        runtime = HipRuntime()
        set_device_result = runtime.set_device(gpu)
        if set_device_result != 0:
            raise RuntimeError(f"hipSetDevice({gpu}) failed: {set_device_result}")

        fd = os.open(path, os.O_RDWR)
        try:
            mapped = mmap.mmap(
                fd,
                total_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)

        # Match the server's resident shared mmap. One worker pre-faulting the
        # MAP_SHARED backing is sufficient; all workers then wait for it.
        if rank == 0:
            populate = getattr(mmap, "MADV_POPULATE_WRITE", 23)
            mapped.madvise(populate)
        barrier.wait()

        base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
        start = time.perf_counter()
        registration = None
        if mode == "simultaneous":
            barrier.wait()
            registration = register_host_chunks(
                runtime, base_ptr, total_size, row_stride, chunk_bytes
            )
        else:
            for turn in range(2):
                if rank == turn:
                    registration = register_host_chunks(
                        runtime, base_ptr, total_size, row_stride, chunk_bytes
                    )
                barrier.wait()
        elapsed = time.perf_counter() - start
        assert registration is not None

        # All successful registrations remain live until both ranks have
        # completed, reproducing the shared-backing overlap at server startup.
        barrier.wait()
        smoke_code = runtime.allocation_smoke()
        cleanup = rollback_host_chunks(runtime, base_ptr, registration.chunks)
        result_queue.put(
            asdict(
                WorkerResult(
                    rank=rank,
                    gpu=gpu,
                    register_ok=registration.ok,
                    register_error_code=registration.error_code,
                    drained_error_code=registration.drained_error_code,
                    chunks=list(registration.chunks),
                    register_seconds=elapsed,
                    post_register_runtime_ok=smoke_code == 0,
                    post_register_runtime_code=smoke_code,
                    cleanup_ok=cleanup.ok,
                )
            )
        )
    except Exception as error:
        result_queue.put(
            asdict(
                WorkerResult(
                    rank=rank,
                    gpu=gpu,
                    register_ok=False,
                    register_error_code=None,
                    drained_error_code=None,
                    chunks=[],
                    register_seconds=0.0,
                    post_register_runtime_ok=False,
                    post_register_runtime_code=-1,
                    cleanup_ok=False,
                    exception=repr(error),
                )
            )
        )
    finally:
        if mapped is not None:
            mapped.close()


def run_case(
    shm_dir: Path,
    size_gib: float,
    chunk_gib: float,
    mode: str,
    gpus: list[int],
    timeout: float,
) -> dict[str, Any]:
    total_size = int(size_gib * 1024**3)
    chunk_bytes = int(chunk_gib * 1024**3)
    row_stride = mmap.PAGESIZE
    path = shm_dir / f"radiance_host_register_probe_{os.getpid()}.mmap"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, total_size)
    finally:
        os.close(fd)

    context = mp.get_context("spawn")
    barrier = context.Barrier(len(gpus))
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(
                rank,
                gpu,
                str(path),
                total_size,
                row_stride,
                chunk_bytes,
                mode,
                barrier,
                result_queue,
            ),
        )
        for rank, gpu in enumerate(gpus)
    ]
    started = time.perf_counter()
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout
        for process in processes:
            process.join(max(deadline - time.monotonic(), 0))
        timed_out = [process for process in processes if process.is_alive()]
        for process in timed_out:
            process.terminate()
        for process in timed_out:
            process.join(10)

        results: list[dict[str, Any]] = []
        while len(results) < len(gpus):
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break
        return {
            "size_gib": size_gib,
            "chunk_gib": chunk_gib,
            "mode": mode,
            "elapsed_seconds": time.perf_counter() - started,
            "timed_out_ranks": [process.name for process in timed_out],
            "exit_codes": [process.exitcode for process in processes],
            "workers": sorted(results, key=lambda item: item["rank"]),
        }
    finally:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-maintenance", action="store_true")
    parser.add_argument(
        "--sizes-gib", nargs="+", type=float, default=[24, 28, 30, 32, 36]
    )
    parser.add_argument("--chunk-gib", nargs="+", type=float, default=[0, 8])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("sequential", "simultaneous"),
        default=["sequential", "simultaneous"],
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--shm-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_maintenance:
        raise SystemExit("Refusing GPU probe without --confirm-maintenance")
    if len(args.gpus) != 2:
        raise SystemExit("This qualification probe currently requires exactly two GPUs")
    if api_port_is_open():
        raise SystemExit("Refusing probe: local API port 8000 is accepting connections")
    if not args.shm_dir.is_dir():
        raise SystemExit(f"Shared-memory directory does not exist: {args.shm_dir}")
    required = int(max(args.sizes_gib) * 1024**3)
    free = shutil.disk_usage(args.shm_dir).free
    if free < required + 1024**3:
        raise SystemExit(
            f"Insufficient {args.shm_dir} capacity: "
            f"need >{required + 1024**3}, have {free}"
        )

    document: dict[str, Any] = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpus": args.gpus,
        "shm_dir": str(args.shm_dir),
        "free_bytes_before": free,
        "cases": [],
    }
    for size_gib in args.sizes_gib:
        for chunk_gib in args.chunk_gib:
            for mode in args.modes:
                case = run_case(
                    args.shm_dir,
                    size_gib,
                    chunk_gib,
                    mode,
                    args.gpus,
                    args.timeout,
                )
                document["cases"].append(case)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(document, indent=2) + "\n")
                print(
                    f"size={size_gib:g} GiB chunk={chunk_gib:g} GiB "
                    f"mode={mode}: {case['workers']}"
                )


if __name__ == "__main__":
    main()
