#!/usr/bin/env python3
"""Retain coordinated ROCm host registration on v0.33's native CPU KV worker.

Upstream owns pageable-safe DMA selection and mapped-file lifetime. Radiance
adds its explicit pin policy, shared runtime-handle ownership, bounded chunks,
and coordinated failure/rollback across model-parallel ranks.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply
LIB = Path(sysconfig.get_paths()["purelib"])

def patch_cuda_wrapper() -> None:
    path = LIB / "vllm/distributed/device_communicators/cuda_wrapper.py"
    apply(
        path,
        '''        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
    ]
''',
        '''        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        Function("cudaGetLastError", cudaError_t, []),
        Function(
            "cudaHostRegister",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint],
        ),
        Function("cudaHostUnregister", cudaError_t, [ctypes.c_void_p]),
    ]
''',
        '"cudaHostRegister",\n            cudaError_t,',
        "kv-offload: runtime host APIs",
    )
    apply(
        path,
        '''        "cudaIpcGetMemHandle": "hipIpcGetMemHandle",
        "cudaIpcOpenMemHandle": "hipIpcOpenMemHandle",
    }
''',
        '''        "cudaIpcGetMemHandle": "hipIpcGetMemHandle",
        "cudaIpcOpenMemHandle": "hipIpcOpenMemHandle",
        "cudaGetLastError": "hipGetLastError",
        "cudaHostRegister": "hipHostRegister",
        "cudaHostUnregister": "hipHostUnregister",
    }
''',
        '"cudaHostRegister": "hipHostRegister"',
        "kv-offload: HIP host API mappings",
    )
    apply(
        path,
        '''    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")
''',
        '''    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            self.drain_pending_error()
            raise RuntimeError(f"CUDART error: {error_str}")
''',
        "self.drain_pending_error()\n            raise RuntimeError",
        "kv-offload: drain checked runtime failures",
    )
    apply(
        path,
        '''        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr
''',
        '''        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr

    def cudaGetLastError(self) -> int:
        """Return and clear the thread-local pending runtime error."""
        return int(self.funcs["cudaGetLastError"]())

    def drain_pending_error(self) -> int:
        """Consume a pending CUDA/HIP error without raising."""
        return self.cudaGetLastError()

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int:
        """Best-effort host registration returning the raw runtime status."""
        return int(self.funcs["cudaHostRegister"](ctypes.c_void_p(ptr), size, flags))

    def cudaHostUnregister(self, ptr: int) -> int:
        """Best-effort host unregister returning the raw runtime status."""
        return int(self.funcs["cudaHostUnregister"](ctypes.c_void_p(ptr)))
''',
        "def drain_pending_error(self)",
        "kv-offload: same-handle host helpers",
    )

def patch_shared_region():
    path = LIB / "vllm/v1/kv_offload/cpu/shared_offload_region.py"
    apply(path, "from vllm.v1.kv_offload.cpu.host_register import host_unregister\n",
          "from radiance_kv_offload import rollback_host_chunks\n",
          "rollback_host_chunks", "KV rollback handle import")
    apply(path, '        self.is_pinned: bool = False\n',
          '        self.is_pinned: bool = False\n        self.pinned_chunks: list[tuple[int, int]] = []\n        self._cudart_lib = None\n',
          'self.pinned_chunks: list[tuple[int, int]]', "KV registered chunk ownership")
    apply(path, '''    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            addresses = self.pinned_addresses or [base_ptr]
            for address in reversed(addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
''',
          '''    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if self._cudart_lib is None:
                raise RuntimeError("Pinned KV mmap lost its registration runtime handle")
            result = rollback_host_chunks(
                self._cudart_lib, self._base.data_ptr(), self.pinned_chunks
            )
            self.pinned_chunks = list(result.remaining_chunks)
            if not result.ok:
                raise RuntimeError("KV mmap unregister failed; refusing to unmap pinned storage")
            self.pinned_addresses.clear()
            self.is_pinned = False
            self._cudart_lib = None
''',
          "refusing to unmap pinned storage", "KV exact-handle cleanup")


def patch_gpu_worker():
    path = LIB / "vllm/v1/kv_offload/cpu/gpu_worker.py"
    apply(path, "import functools\n", "import functools\nimport math\n", "import math\n", "KV coordination math")
    apply(path, 'from vllm import _custom_ops as ops\n',
          '''from radiance_kv_offload import (
    get_pin_policy, get_register_chunk_bytes, register_host_chunks, rollback_host_chunks,
)
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.parallel_state import (
    GroupCoordinator, get_pcp_group, get_pp_group, get_tp_group, model_parallel_is_initialized,
)
from vllm import _custom_ops as ops
''', "from radiance_kv_offload import (", "KV registration imports")
    old_pin = '''def pin_mmap_region(
    region: SharedOffloadRegion, worker_slot_only: bool = False
) -> None:
    """Register row-aligned chunks or this worker's slots, rolling back on failure."""
    rank = region.rank
    base_ptr = region._base.data_ptr()
    total_size = region.total_size_bytes
    # Chunks end on block-row boundaries, which are page aligned, so neither the
    # driver's page rounding nor any single block transfer straddles two
    # registrations.
    rows_per_chunk = max(MAX_HOST_REGISTER_CHUNK_BYTES // region._row_stride, 1)
    chunk_size = rows_per_chunk * region._row_stride
    ranges: Iterator[tuple[int, int]] = (
        (offset, min(chunk_size, total_size - offset))
        for offset in range(0, total_size, chunk_size)
    )
    stride, slot, page = region._row_stride, region.cpu_page_size, region.page_size
    min_savings = max(page, MIN_HOST_REGISTER_SAVED_BYTES)
    if worker_slot_only and region.rank is not None and stride - slot >= min_savings:
        # Round each slot out to pages; the worker's next slot is a page away.
        ranges = region.iter_worker_slot_ranges()
    else:
        reason = (
            "whole-region layout requested"
            if not worker_slot_only
            else "no worker rank"
            if region.rank is None
            else f"less than {min_savings} bytes saved per row"
        )
        logger.info(
            "Registering full CPU offload region rank=%s (row=%d, slot=%d bytes): %s",
            rank,
            stride,
            slot,
            reason,
        )

    # Register, drain and roll back through the same helper, so a failed
    # chunk leaves neither a pending error nor a partly pinned region.
    addresses: list[int] = []
    registered_bytes = 0
    for offset, size in ranges:
        address = base_ptr + offset
        if host_register(address, size):
            addresses.append(address)
            registered_bytes += size
            continue
        logger.warning(
            "host_register failed for rank=%d at %.2f of %.2f GB; "
            "the offload region stays pageable",
            rank,
            offset / 1e9,
            total_size / 1e9,
        )
        for registered in reversed(addresses):
            host_unregister(registered)
        return

    region.pinned_addresses.extend(addresses)
    region.is_pinned = True
    logger.debug(
        "Host-registered mmap region rank=%d %.2f GB in %d chunk(s)",
        rank,
        registered_bytes / 1e9,
        len(addresses),
    )
'''
    new_pin = '''@dataclass(frozen=True)
class _ModelParallelCoordination:
    groups: tuple[GroupCoordinator, GroupCoordinator, GroupCoordinator]

    @property
    def rank(self) -> int:
        tp, pcp, pp = self.groups
        return (
            (pp.rank_in_group * pcp.world_size + pcp.rank_in_group) * tp.world_size
            + tp.rank_in_group
        )

    @property
    def world_size(self) -> int:
        return math.prod(group.world_size for group in self.groups)

    def barrier(self) -> None:
        for group in self.groups:
            if group.world_size > 1:
                group.barrier()


def _model_parallel_coordination() -> _ModelParallelCoordination:
    return _ModelParallelCoordination((get_tp_group(), get_pcp_group(), get_pp_group()))


def _group_max(status: int, groups: Sequence[GroupCoordinator]) -> int:
    status_tensor = torch.tensor([status], dtype=torch.int32, device="cpu")
    for group in groups:
        if group.world_size > 1:
            torch.distributed.all_reduce(
                status_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group.cpu_group,
            )
    return int(status_tensor.item())


def pin_mmap_region(
    region: SharedOffloadRegion,
    worker_slot_only: bool = False,
    coordination: _ModelParallelCoordination | None = None,
) -> None:
    """Register mmap consistently or coherently fall back to pageable DMA."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    # Preserve the qualified whole-region policy on block-major mappings.
    # The v0.33 worker-slot hint is not a rank-major layout declaration.
    policy = get_pin_policy()
    if policy == "disabled":
        logger.info("KV mmap host registration disabled by Radiance policy")
        return

    # Without distributed coordination this process is the only registration
    # participant, regardless of the mmap layout's diagnostic rank field.
    rank = coordination.rank if coordination is not None else 0
    local_status = 0
    local_error: Exception | None = None
    base_ptr: int | None = None
    runtime: CudaRTLibrary | None = None
    chunk_bytes = get_register_chunk_bytes()

    try:
        base_ptr = region._base.data_ptr()
        runtime = CudaRTLibrary()
    except Exception as error:
        local_status = 2
        local_error = error
        logger.exception("Failed to prepare KV mmap registration for rank=%d", rank)

    if coordination is not None:
        coordination.barrier()
        turns = range(coordination.world_size)
    else:
        turns = range(1)

    for turn in turns:
        if rank == turn and local_status == 0:
            assert base_ptr is not None and runtime is not None
            result = register_host_chunks(
                runtime,
                base_ptr,
                region.total_size_bytes,
                region._row_stride,
                chunk_bytes,
            )
            region.pinned_chunks = list(result.chunks)
            region.is_pinned = bool(result.chunks)
            region._cudart_lib = runtime if result.chunks else None
            if not result.ok:
                local_status = 2 if result.exception or not result.rollback_ok else 1
                local_error = result.exception
                logger.warning(
                    "KV mmap host registration failed for group rank=%d, mmap "
                    "rank=%s (code=%s, drained=%s); policy=%s",
                    rank,
                    region.rank,
                    result.error_code,
                    result.drained_error_code,
                    policy,
                )
        if coordination is not None:
            coordination.barrier()

    group_status = (
        _group_max(local_status, coordination.groups)
        if coordination is not None
        else local_status
    )
    if group_status == 0:
        logger.info(
            "KV mmap host registration active for rank=%d: %.2f GB in %d chunk(s)",
            rank,
            region.total_size_bytes / 1e9,
            len(region.pinned_chunks),
        )
        return

    rollback_status = 0
    if coordination is not None:
        rollback_turns = range(coordination.world_size)
    else:
        rollback_turns = range(1)
    for turn in rollback_turns:
        if rank == turn and region.is_pinned:
            assert base_ptr is not None and runtime is not None
            rollback = rollback_host_chunks(
                runtime, base_ptr, region.pinned_chunks
            )
            region.pinned_chunks = list(rollback.remaining_chunks)
            region.is_pinned = bool(region.pinned_chunks)
            if not region.is_pinned:
                region._cudart_lib = None
            if not rollback.ok:
                rollback_status = 2
                local_error = rollback.exception or local_error
                logger.error(
                    "KV mmap host-registration rollback failed for rank=%d "
                    "(code=%s, remaining_chunks=%d)",
                    rank,
                    rollback.error_code,
                    len(rollback.remaining_chunks),
                )
        if coordination is not None:
            coordination.barrier()

    group_rollback_status = (
        _group_max(rollback_status, coordination.groups)
        if coordination is not None
        else rollback_status
    )
    if group_status == 1 and group_rollback_status == 0 and policy == "auto":
        if rank == 0:
            logger.warning(
                "KV mmap registration failed on at least one worker; all "
                "workers will use pageable DMA"
            )
        return

    if policy == "required" and group_status == 1 and group_rollback_status == 0:
        error = RuntimeError(
            "KV mmap registration is required but at least one worker could not pin"
        )
    else:
        error = RuntimeError(
            "Coordinated KV mmap registration failed or could not roll back safely"
        )
    if local_error is not None:
        raise error from local_error
    raise error
'''
    apply(path, old_pin, new_pin, "KV mmap registration failed on at least one worker", "KV coordinated registration")
    apply(path,
          '        if mmap_region is not None and pin_memory:\n            pin_mmap_region(mmap_region, worker_slot_only=not canonical_layout)\n',
          '''        if mmap_region is not None and pin_memory:
            coordination = _model_parallel_coordination() if model_parallel_is_initialized() else None
            pin_mmap_region(mmap_region, worker_slot_only=not canonical_layout, coordination=coordination)
''', 'coordination=coordination)', "KV worker coordination")
    text = path.read_text()
    for contract in ("gpu_to_cpu or not host_memory_is_pinned", "current_platform.is_rocm()", "host_memory_is_pinned=host_memory_is_pinned"):
        if contract not in text:
            raise SystemExit(f"Upstream pageable/ROCm DMA contract drift: {contract}")


def main():
    patch_cuda_wrapper()
    patch_shared_region()
    patch_gpu_worker()

if __name__ == "__main__":
    main()
