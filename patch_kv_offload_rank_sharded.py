#!/usr/bin/env python3
"""Opt-in rank-major layout for native mmap-backed CPU KV offload.

The pinned vLLM v0.28.0 native CPU tier lays the shared mmap out block-major:
each logical chunk row contains every model-parallel worker's CPU page. That
preserves a convenient whole-buffer secondary-tier memoryview, but it forces
every TP worker to host-register the complete shared mmap even though direct
CPU<->GPU DMA only touches that worker's slot in each row.

When ``RADIANCE_KV_OFFLOAD_RANK_SHARDED=1`` this overlay transposes only the
native direct-worker physical layout to rank-major:

    rank0 block0 | rank0 block1 | ... | rank0 blockN
    rank1 block0 | rank1 block1 | ... | rank1 blockN

The shared mmap remains one allocation of the same total byte size and every
worker keeps the same logical block IDs/count. Each worker tensor instead uses
a rank-local row stride and host registration targets only that rank's one
contiguous span. Registered chunk ownership is converted back to offsets from
the full mmap base so the existing same-runtime cleanup and group-atomic
rollback logic remains unchanged.

Canonical/secondary-tier views deliberately stay on the legacy block-major
layout. The native ``CPUOffloadingSpec`` is the only constructor that opts into
rank-major mode; tiering callers retain the default ``rank_sharded=False``.
Guards reject accidental canonical/memoryview use on a rank-major region rather
than returning a non-contiguous Python memoryview to downstream consumers.

This overlay is applied after ``patch_kv_offload_registration.py`` and
``patch_kv_offload_lifecycle.py``. Idempotent; exact-anchor guarded; ast.parse
checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])


def patch_shared_region() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/shared_offload_region.py"
    apply(
        path,
        "from radiance_kv_offload import rollback_host_chunks\n",
        '''from radiance_kv_offload import (
    plan_rank_major_layout,
    rollback_host_chunks,
)
''',
        "plan_rank_major_layout,",
        "kv-offload rank-sharded: import layout planner",
    )
    apply(
        path,
        '''        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
        populate_only_on_creator: bool = False,
    ) -> None:
        if populate_only_on_creator and barrier is None:
''',
        '''        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
        populate_only_on_creator: bool = False,
        rank_sharded: bool = False,
        world_size: int | None = None,
    ) -> None:
        if rank_sharded and populate_only_on_creator:
            raise ValueError("rank-major layout requires per-worker population")
        if populate_only_on_creator and barrier is None:
''',
        "rank_sharded: bool = False",
        "kv-offload rank-sharded: constructor controls",
    )
    apply(
        path,
        '''        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each chunk row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
''',
        '''        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        self._rank_sharded = rank_sharded
        self._worker_row_stride = self._row_stride
        self._registration_offset = 0
        self._registration_size = self.total_size_bytes
        self._registration_row_stride = self._row_stride
        if rank_sharded:
            if rank is None or world_size is None:
                raise ValueError(
                    "rank-major KV offload requires both rank and world_size"
                )
            layout = plan_rank_major_layout(
                total_size=self.total_size_bytes,
                num_blocks=self.num_chunks,
                world_size=world_size,
                worker_page_size=cpu_page_size,
                page_size=self.page_size,
            )
            shard = layout.range_for_rank(rank)
            self._worker_row_stride = shard.row_stride
            self._worker_offset = shard.offset
            # Tensor payloads occupy worker_page_size bytes; any remaining
            # rank-local row bytes preserve the original alignment padding.
            self._worker_area_end = shard.offset + cpu_page_size
            self._registration_offset = shard.offset
            self._registration_size = shard.size
            self._registration_row_stride = shard.row_stride
        elif rank is not None:
            # Legacy block-major layout: byte offset to this worker's first
            # slot within each chunk row.
            self._worker_offset = rank * cpu_page_size
            self._worker_area_end = (rank + 1) * cpu_page_size
''',
        "self._registration_offset = shard.offset",
        "kv-offload rank-sharded: plan rank-major span",
    )
    apply(
        path,
        '''        if rank is not None:
            # Populate only this worker's pages (one slot per chunk row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for chunk in range(num_chunks):
                raw_offset = chunk * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                populate_write_fn(self.mmap_obj, aligned_offset, aligned_length)
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d chunks in %.3f s",
                num_chunks,
                time.perf_counter() - _t0,
            )
''',
        '''        if rank is not None:
            _t0 = time.perf_counter()
            if self._rank_sharded:
                # The rank-major worker span is one page-aligned contiguous
                # range, so prefault it in one call instead of touching a slot
                # from every global chunk row.
                populate_write_fn(
                    self.mmap_obj,
                    self._registration_offset,
                    self._registration_size,
                )
                logger.debug(
                    "MADV_POPULATE_WRITE rank-major span: %.2f GB in %.3f s",
                    self._registration_size / 1e9,
                    time.perf_counter() - _t0,
                )
            else:
                # Legacy block-major layout: populate only this worker's pages
                # (one slot per chunk row).
                worker_offset = rank * cpu_page_size
                page_size = self.page_size
                for chunk in range(num_chunks):
                    raw_offset = chunk * self._row_stride + worker_offset
                    aligned_offset = (raw_offset // page_size) * page_size
                    end = raw_offset + cpu_page_size
                    aligned_length = end - aligned_offset
                    populate_write_fn(self.mmap_obj, aligned_offset, aligned_length)
                logger.debug(
                    "MADV_POPULATE_WRITE loop: %d chunks in %.3f s",
                    num_chunks,
                    time.perf_counter() - _t0,
                )
''',
        "MADV_POPULATE_WRITE rank-major span",
        "kv-offload rank-sharded: contiguous prefault",
    )
    apply(
        path,
        '''        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
''',
        '''        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._worker_row_stride, 1),
            storage_offset=self._worker_offset,
        )
''',
        "stride=(self._worker_row_stride, 1)",
        "kv-offload rank-sharded: worker tensor stride",
    )
    apply(
        path,
        '''        Args:
            tensor_page_size: Canonical bytes per chunk for this tensor.
        """
        new_offset = self._canonical_offset + tensor_page_size
''',
        '''        Args:
            tensor_page_size: Canonical bytes per chunk for this tensor.
        """
        if self._rank_sharded:
            raise RuntimeError(
                "rank-major layout is native-worker-only; canonical KV mappings "
                "require the legacy block-major shared region"
            )
        new_offset = self._canonical_offset + tensor_page_size
''',
        "rank-major layout is native-worker-only",
        "kv-offload rank-sharded: protect canonical layout",
    )
    apply(
        path,
        '''        Shape: (num_chunks, row_stride_bytes). Secondary tiers address
        chunk *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_chunks, self._row_stride)
''',
        '''        Shape: (num_chunks, row_stride_bytes). Secondary tiers address
        chunk *b* as ``view[b]``.
        """
        if self._rank_sharded:
            raise RuntimeError(
                "rank-major layout cannot expose the legacy contiguous "
                "block-major secondary-tier memoryview"
            )
        kv_tensor = self._base.view(self.num_chunks, self._row_stride)
''',
        "rank-major layout cannot expose the legacy contiguous",
        "kv-offload rank-sharded: protect secondary-tier memoryview",
    )


def patch_cpu_spec() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/spec.py"
    apply(
        path,
        "import torch\nfrom typing_extensions import override\n",
        '''import torch
from typing_extensions import override

from radiance_kv_offload import get_rank_sharded_enabled
''',
        "from radiance_kv_offload import get_rank_sharded_enabled",
        "kv-offload rank-sharded: import opt-in",
    )
    apply(
        path,
        '''        if self._uses_shared_region() and self.num_chunks > 0:
            # Replicated layout puts all ranks on slot 0 (single MLA copy);
            # otherwise each rank takes its own slot by physical device index.
            if self.replicated_layout:
                rank = 0
            else:
                world_size = self.config.parallel.world_size
                rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_chunks=self.num_chunks,
                rank=rank,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                barrier=_all_workers_barrier,
            )
''',
        '''        if self._uses_shared_region() and self.num_chunks > 0:
            world_size = self.config.parallel.world_size
            rank_sharded = get_rank_sharded_enabled() and world_size > 1
            # Replicated layout puts all ranks on slot 0 (single MLA copy). A
            # rank-major private span would change that canonical ownership, so
            # reject the experimental mode instead of silently changing it.
            if self.replicated_layout:
                if rank_sharded:
                    raise RuntimeError(
                        "RADIANCE_KV_OFFLOAD_RANK_SHARDED is incompatible with "
                        "replicated CPU KV layout"
                    )
                rank = 0
            else:
                rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_chunks=self.num_chunks,
                rank=rank,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                barrier=_all_workers_barrier,
                rank_sharded=rank_sharded,
                world_size=world_size if rank_sharded else None,
            )
''',
        "rank_sharded=rank_sharded,",
        "kv-offload rank-sharded: native CPU worker opt-in",
    )


def patch_gpu_worker_registration() -> None:
    path = LIB / "vllm/v1/kv_offload/cpu/gpu_worker.py"
    apply(
        path,
        '''        base_ptr = region._base.data_ptr()
        runtime = CudaRTLibrary()
''',
        '''        base_ptr = region._base.data_ptr()
        registration_offset = region._registration_offset
        registration_ptr = base_ptr + registration_offset
        registration_size = region._registration_size
        registration_row_stride = region._registration_row_stride
        runtime = CudaRTLibrary()
''',
        "registration_ptr = base_ptr + registration_offset",
        "kv-offload rank-sharded: select registration range",
    )
    apply(
        path,
        '''            result = register_host_chunks(
                runtime,
                base_ptr,
                region.total_size_bytes,
                region._row_stride,
                chunk_bytes,
            )
            region.pinned_chunks = list(result.chunks)
            region.is_pinned = bool(result.chunks)
''',
        '''            result = register_host_chunks(
                runtime,
                registration_ptr,
                registration_size,
                registration_row_stride,
                chunk_bytes,
            )
            # register_host_chunks reports offsets relative to the pointer it
            # receives. Store mmap-relative offsets so the existing rollback
            # and cleanup code can continue using the full mmap base pointer.
            region.pinned_chunks = [
                (registration_offset + offset, size)
                for offset, size in result.chunks
            ]
            region.is_pinned = bool(region.pinned_chunks)
''',
        "registration_offset + offset, size",
        "kv-offload rank-sharded: register local span",
    )
    apply(
        path,
        '''            "KV mmap host registration active for rank=%d: %.2f GB in %d chunk(s)",
            rank,
            region.total_size_bytes / 1e9,
            len(region.pinned_chunks),
''',
        '''            "KV mmap host registration active for rank=%d: %.2f GB in %d chunk(s)",
            rank,
            registration_size / 1e9,
            len(region.pinned_chunks),
''',
        "registration_size / 1e9,\n            len(region.pinned_chunks)",
        "kv-offload rank-sharded: log local registration size",
    )


def main() -> None:
    patch_shared_region()
    patch_cpu_spec()
    patch_gpu_worker_registration()


if __name__ == "__main__":
    main()
