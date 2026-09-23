#!/usr/bin/env python3
"""Destructive-to-capacity (not data) ROCm host-registration probe.

This allocates and pre-faults a large shared-memory mmap, maps it in one process
per GPU, and measures whole/chunked ``hipHostRegister`` behavior under serial
and simultaneous TP-like registration. The legacy ``shared`` layout registers
the complete mmap in every worker. The experimental ``rank-sharded`` layout
registers one disjoint contiguous rank span per worker while keeping the same
combined physical allocation.

It must only run in a declared GPU maintenance window. The production API guard
is intentionally fail-closed.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
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

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from radiance_kv_offload import register_host_chunks, rollback_host_chunks


@dataclass
class WorkerResult:
    rank: int
    gpu: int
    layout: str
    registration_offset: int
    registration_size: int
    register_ok: bool
    register_error_code: int | None
    drained_error_code: int | None
    chunks: list[tuple[int, int]]
    registered_bytes: int
    register_seconds: float
    post_register_runtime_ok: bool
    post_register_runtime_code: int
    cleanup_ok: bool
    stage: str
    exception: str | None = None


class HipRuntime:
    def __init__(self) -> None:
        candidates = [ctypes.util.find_library("amdhip64")]
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidates.extend(
            [
                f"{rocm}/lib/libamdhip64.so",
                f"{rocm}/lib64/libamdhip64.so",
                *glob.glob(f"{rocm}/core-*/lib/libamdhip64.so"),
            ]
        )
        path = next(
            (
                candidate
                for candidate in candidates
                if candidate and Path(candidate).exists()
            ),
            None,
        )
        path = path or "libamdhip64.so"
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


def registration_range(
    layout: str,
    rank: int,
    world_size: int,
    total_size: int,
) -> tuple[int, int]:
    if layout == "shared":
        return 0, total_size
    if layout != "rank-sharded":
        raise ValueError(f"unknown layout: {layout}")
    if total_size % world_size:
        raise ValueError("rank-sharded probe requires total_size divisible by world_size")
    size = total_size // world_size
    offset = rank * size
    if offset % mmap.PAGESIZE or size % mmap.PAGESIZE:
        raise ValueError("rank-sharded probe ranges must be page aligned")
    return offset, size


def worker(
    rank: int,
    gpu: int,
    path: str,
    total_size: int,
    row_stride: int,
    chunk_bytes: int,
    mode: str,
    prefault: str,
    layout: str,
    world_size: int,
    barrier: Any,
    barrier_timeout: float,
    result_queue: Any,
) -> None:
    mapped: mmap.mmap | None = None
    stage = "runtime-init"
    registration_offset = 0
    registration_size = 0
    try:
        runtime = HipRuntime()
        set_device_result = runtime.set_device(gpu)
        if set_device_result != 0:
            raise RuntimeError(f"hipSetDevice({gpu}) failed: {set_device_result}")

        stage = "mmap"
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

        # Match vLLM's distributed residency: each worker makes a disjoint part
        # of the MAP_SHARED backing resident before registration.
        stage = "prefault"
        if prefault == "distributed":
            start = (total_size * rank // world_size // mmap.PAGESIZE) * mmap.PAGESIZE
            end = (
                total_size
                if rank == world_size - 1
                else (total_size * (rank + 1) // world_size // mmap.PAGESIZE)
                * mmap.PAGESIZE
            )
            resident = np.frombuffer(mapped, dtype=np.uint8)
            resident[start:end:mmap.PAGESIZE] |= 0
            del resident
        stage = "prefault-barrier"
        barrier.wait(timeout=barrier_timeout)

        stage = "register"
        base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
        registration_offset, registration_size = registration_range(
            layout, rank, world_size, total_size
        )
        registration_ptr = base_ptr + registration_offset
        start = time.perf_counter()
        registration = None
        if mode == "simultaneous":
            barrier.wait(timeout=barrier_timeout)
            registration = register_host_chunks(
                runtime,
                registration_ptr,
                registration_size,
                row_stride,
                chunk_bytes,
            )
        else:
            for turn in range(world_size):
                if rank == turn:
                    registration = register_host_chunks(
                        runtime,
                        registration_ptr,
                        registration_size,
                        row_stride,
                        chunk_bytes,
                    )
                barrier.wait(timeout=barrier_timeout)
        elapsed = time.perf_counter() - start
        assert registration is not None

        # Convert local registration offsets to full-mmap offsets, matching the
        # production overlay's ownership convention.
        owned_chunks = [
            (registration_offset + offset, size) for offset, size in registration.chunks
        ]

        # Successful registrations remain live until every rank has completed,
        # reproducing the startup overlap/accounting behavior.
        stage = "post-register-barrier"
        barrier.wait(timeout=barrier_timeout)
        stage = "runtime-smoke"
        smoke_code = runtime.allocation_smoke()
        stage = "cleanup"
        cleanup = rollback_host_chunks(runtime, base_ptr, owned_chunks)
        result_queue.put(
            asdict(
                WorkerResult(
                    rank=rank,
                    gpu=gpu,
                    layout=layout,
                    registration_offset=registration_offset,
                    registration_size=registration_size,
                    register_ok=registration.ok,
                    register_error_code=registration.error_code,
                    drained_error_code=registration.drained_error_code,
                    chunks=owned_chunks,
                    registered_bytes=sum(size for _, size in owned_chunks),
                    register_seconds=elapsed,
                    post_register_runtime_ok=smoke_code == 0,
                    post_register_runtime_code=smoke_code,
                    cleanup_ok=cleanup.ok,
                    stage="complete",
                )
            )
        )
    except Exception as error:
        result_queue.put(
            asdict(
                WorkerResult(
                    rank=rank,
                    gpu=gpu,
                    layout=layout,
                    registration_offset=registration_offset,
                    registration_size=registration_size,
                    register_ok=False,
                    register_error_code=None,
                    drained_error_code=None,
                    chunks=[],
                    registered_bytes=0,
                    register_seconds=0.0,
                    post_register_runtime_ok=False,
                    post_register_runtime_code=-1,
                    cleanup_ok=False,
                    stage=stage,
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
    prefault: str,
    layout: str,
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
                prefault,
                layout,
                len(gpus),
                barrier,
                min(timeout, 120),
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
        while time.monotonic() < deadline and any(
            process.is_alive() for process in processes
        ):
            for process in processes:
                process.join(0.1)
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
            "prefault": prefault,
            "layout": layout,
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
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("shared", "rank-sharded"),
        default=["shared"],
        help="shared preserves the historical full-mmap registration; rank-sharded registers one contiguous span per rank",
    )
    parser.add_argument(
        "--prefault",
        choices=("none", "distributed"),
        default="none",
        help="Populate pages before registration; distributed matches vLLM residency",
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
        "schema": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpus": args.gpus,
        "shm_dir": str(args.shm_dir),
        "free_bytes_before": free,
        "cases": [],
    }
    for size_gib in args.sizes_gib:
        for chunk_gib in args.chunk_gib:
            for layout in args.layouts:
                for mode in args.modes:
                    case = run_case(
                        args.shm_dir,
                        size_gib,
                        chunk_gib,
                        mode,
                        args.prefault,
                        layout,
                        args.gpus,
                        args.timeout,
                    )
                    document["cases"].append(case)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(document, indent=2) + "\n")
                    print(
                        f"size={size_gib:g} GiB chunk={chunk_gib:g} GiB "
                        f"layout={layout} mode={mode} prefault={args.prefault}: "
                        f"{case['workers']}"
                    )


if __name__ == "__main__":
    main()
