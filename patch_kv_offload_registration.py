#!/usr/bin/env python3
"""Harden mmap KV-offload host registration on CUDA/ROCm.

This v0.28.0 overlay combines the narrowly relevant pieces of upstream vLLM
PRs #50070, #51081, and #52296:

* call host-register/unregister through one runtime-library handle and drain a
  failed registration's pending HIP/CUDA error;
* serialize registration across TP/PCP/PP workers and require an all-pinned or
  coherently all-pageable result;
* retain every successful row-aligned chunk for exact rollback and cleanup.

Chunking is implemented but opt-in pending gfx1201 qualification.  The default
is one whole-region registration, matching vLLM v0.28.0 behavior.  The default
``auto`` policy falls back to pageable DMA if any worker cannot pin; ``required``
fails startup, and ``disabled`` skips registration.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
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


def patch_shared_region() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/shared_offload_region.py"
    apply(
        path,
        "from vllm.distributed.device_communicators.shm_broadcast import (\n",
        "from radiance_kv_offload import rollback_host_chunks\n\n"
        "from vllm.distributed.device_communicators.shm_broadcast import (\n",
        "from radiance_kv_offload import rollback_host_chunks",
        "kv-offload: import registration rollback",
    )
    apply(
        path,
        '''        self._canonical_offset = 0
        self.is_pinned: bool = False
''',
        '''        self._canonical_offset = 0
        self.is_pinned: bool = False
        self.pinned_chunks: list[tuple[int, int]] = []
        self._cudart_lib = None
''',
        "self.pinned_chunks: list[tuple[int, int]]",
        "kv-offload: retain registered chunks",
    )
    apply(
        path,
        '''    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                base_ptr = self._base.data_ptr()
                result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.is_pinned = False
''',
        '''    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike() and self._cudart_lib is not None:
                rollback = rollback_host_chunks(
                    self._cudart_lib,
                    self._base.data_ptr(),
                    self.pinned_chunks,
                )
                if not rollback.ok:
                    logger.warning(
                        "cudaHostUnregister cleanup incomplete for rank=%d "
                        "(code=%s, remaining_chunks=%d)",
                        self.rank,
                        rollback.error_code,
                        len(rollback.remaining_chunks),
                    )
            elif current_platform.is_cuda_alike():
                logger.warning(
                    "Missing runtime handle while unregistering rank=%d", self.rank
                )
            self.pinned_chunks.clear()
            self.is_pinned = False
            self._cudart_lib = None
''',
        "cudaHostUnregister cleanup incomplete",
        "kv-offload: exact same-handle cleanup",
    )


def patch_gpu_worker() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/gpu_worker.py"
    apply(
        path,
        "import functools\nimport time\n",
        "import functools\nimport math\nimport time\n",
        "import math\nimport time",
        "kv-offload: import math",
    )
    apply(
        path,
        "from vllm import _custom_ops as ops\nfrom vllm.logger import init_logger\n",
        '''from radiance_kv_offload import (
    get_pin_policy,
    get_register_chunk_bytes,
    register_host_chunks,
    rollback_host_chunks,
)
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.logger import init_logger
''',
        "from radiance_kv_offload import (",
        "kv-offload: import coordinated registration",
    )
    apply(
        path,
        '''def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
''',
        '''def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
    host_memory_is_pinned: bool = True,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; pageable host memory also requires DMA.
    if gpu_to_cpu or not host_memory_is_pinned:
        return ops.swap_blocks_batch
''',
        "gpu_to_cpu or not host_memory_is_pinned",
        "kv-offload: force DMA for pageable host memory",
    )

    old_pin = '''def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region._base.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True
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
    apply(
        path,
        old_pin,
        new_pin,
        "KV mmap registration failed on at least one worker",
        "kv-offload: coordinated host registration",
    )

    apply(
        path,
        '''        gpu_to_cpu: bool,
        canonical_layout: bool = False,
    ):
''',
        '''        gpu_to_cpu: bool,
        canonical_layout: bool = False,
        host_memory_is_pinned: bool = True,
    ):
''',
        "canonical_layout: bool = False,\n        host_memory_is_pinned: bool = True",
        "kv-offload: handler pin-state argument",
    )
    apply(
        path,
        '''            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
        """
''',
        '''            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            host_memory_is_pinned: whether kernels may dereference host tensors.
        """
''',
        "host_memory_is_pinned: whether kernels",
        "kv-offload: document handler pin state",
    )
    apply(
        path,
        '''        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu
        )
''',
        '''        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu, host_memory_is_pinned
        )
''',
        "layer_refs_per_group, gpu_to_cpu, host_memory_is_pinned",
        "kv-offload: select pageable-safe transfer",
    )
    apply(
        path,
        '''        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        canonical_bytes_per_block = (
''',
        '''        if mmap_region is not None and pin_memory:
            coordination = (
                _model_parallel_coordination()
                if model_parallel_is_initialized()
                else None
            )
            pin_mmap_region(mmap_region, coordination)

        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned
        )

        canonical_bytes_per_block = (
''',
        "pin_mmap_region(mmap_region, coordination)",
        "kv-offload: coordinate worker pinning",
    )
    apply(
        path,
        '''            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )
''',
        '''            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )
''',
        "gpu_to_cpu=True,\n"
        "            canonical_layout=canonical_layout,\n"
        "            host_memory_is_pinned",
        "kv-offload: store handler pin state",
    )
    apply(
        path,
        '''            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
        )
''',
        '''            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )
''',
        "gpu_to_cpu=False,\n"
        "            canonical_layout=canonical_layout,\n"
        "            host_memory_is_pinned",
        "kv-offload: load handler pin state",
    )


def main() -> None:
    patch_cuda_wrapper()
    patch_shared_region()
    patch_gpu_worker()


if __name__ == "__main__":
    main()
