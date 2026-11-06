#!/usr/bin/env python3
"""Preserve the explicit rank-major CPU KV layout on v0.33 segmented mappings.

The default remains upstream block-major. Rank-major private spans may not be
used for canonical/replicated secondary tiers. Registration ownership remains
mmap-relative, so coordinated rollback and cleanup cover the exact same bytes.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply
LIB = Path(sysconfig.get_paths()["purelib"])


def patch_shared_region():
    path = LIB / "vllm/v1/kv_offload/cpu/shared_offload_region.py"
    apply(path, 'from radiance_kv_offload import rollback_host_chunks\n',
          'from radiance_kv_offload import plan_rank_major_layout, rollback_host_chunks\n',
          'import plan_rank_major_layout, rollback_host_chunks', 'rank-major planner import')
    apply(path, '        num_segments: int = 1,\n',
          '        num_segments: int = 1,\n        rank_sharded: bool = False,\n        world_size: int | None = None,\n',
          'rank_sharded: bool = False', 'rank-major constructor')
    anchor = '        barriers_remaining = 2 if populate_shard is not None else 1\n'
    new = '''        self._rank_sharded = rank_sharded
        self._worker_row_stride = self._row_stride
        self._registration_offset = 0
        self._registration_size = self.total_size_bytes
        self._registration_row_stride = self._row_stride
        if rank_sharded:
            if rank is None or world_size is None or populate_shard is not None:
                raise ValueError("rank-major KV requires a private worker rank and per-worker population")
            layout = plan_rank_major_layout(
                total_size=self.total_size_bytes, num_blocks=self.num_chunks,
                world_size=world_size, worker_page_size=cpu_page_size, page_size=self.page_size,
            )
            shard = layout.range_for_rank(rank)
            self._worker_row_stride = shard.row_stride
            self._worker_offset = shard.offset
            self._worker_area_end = shard.offset + cpu_page_size
            self._registration_offset = shard.offset
            self._registration_size = shard.size
            self._registration_row_stride = shard.row_stride
''' + anchor
    apply(path, anchor, new, 'self._registration_offset = shard.offset', 'rank-major private span')
    anchor = '        assert self.rank is not None\n        worker_offset = self.rank * self.cpu_page_size\n'
    new = '        if self._rank_sharded:\n            yield self._registration_offset, self._registration_size\n            return\n' + anchor
    apply(path, anchor, new, 'yield self._registration_offset, self._registration_size', 'rank-major contiguous population')
    apply(path, '            stride=(self._row_stride, 1),\n            storage_offset=self._worker_offset,\n',
          '            stride=(self._worker_row_stride, 1),\n            storage_offset=self._worker_offset,\n',
          'stride=(self._worker_row_stride, 1)', 'rank-major worker tensor stride')
    anchor = '        new_offset = self._canonical_offset + tensor_page_size\n'
    apply(path, anchor, '        if self._rank_sharded:\n            raise RuntimeError("rank-major KV layout does not expose canonical views")\n' + anchor,
          'rank-major KV layout does not expose canonical views', 'rank-major canonical refusal')
    anchor = '        kv_tensor = self._base.view(self.num_chunks, self._row_stride)\n'
    apply(path, anchor, '        if self._rank_sharded:\n            raise RuntimeError("rank-major KV layout does not expose a canonical memoryview")\n' + anchor,
          'rank-major KV layout does not expose a canonical memoryview', 'rank-major tier refusal')


def patch_spec():
    path = LIB / "vllm/v1/kv_offload/cpu/spec.py"
    apply(path, 'from typing_extensions import override\n', 'from typing_extensions import override\nfrom radiance_kv_offload import get_rank_sharded_enabled\n',
          'from radiance_kv_offload import get_rank_sharded_enabled', 'rank-major opt-in import')
    anchor = '            worker_index = self.config.parallel.rank\n'
    new = anchor + '''            rank_sharded = get_rank_sharded_enabled() and world_size > 1
            if rank_sharded and self.replicated_layout:
                raise RuntimeError("RADIANCE_KV_OFFLOAD_RANK_SHARDED is incompatible with replicated CPU KV layout")
'''
    apply(path, anchor, new, 'rank_sharded = get_rank_sharded_enabled()', 'rank-major opt-in policy')
    apply(path, '                num_segments=_REGION_SEGMENTS,\n',
          '                num_segments=_REGION_SEGMENTS,\n                rank_sharded=rank_sharded,\n                world_size=world_size if rank_sharded else None,\n',
          'rank_sharded=rank_sharded,', 'rank-major logical worker ownership')


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


def main():
    patch_shared_region()
    patch_spec()
    patch_gpu_worker_registration()

if __name__ == "__main__":
    main()
