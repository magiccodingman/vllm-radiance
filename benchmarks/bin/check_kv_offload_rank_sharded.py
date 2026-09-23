#!/usr/bin/env python3
"""GPU-free checks for the experimental rank-major KV offload layout."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from radiance_kv_offload import (  # noqa: E402
    RANK_SHARDED_ENV,
    get_rank_sharded_enabled,
    plan_rank_major_layout,
    register_host_chunks,
    rollback_host_chunks,
)


class FakeRuntime:
    def __init__(self, register_results: list[int] | None = None) -> None:
        self.register_results = list(register_results or [])
        self.events: list[tuple] = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int = 0) -> int:
        self.events.append(("register", ptr, size))
        return self.register_results.pop(0) if self.register_results else 0

    def cudaHostUnregister(self, ptr: int) -> int:
        self.events.append(("unregister", ptr))
        return 0

    def drain_pending_error(self) -> int:
        self.events.append(("drain",))
        return 1


def layout_for_gib(size_gib: int):
    total = size_gib * 1024**3
    # 8 KiB global rows => 4 KiB rank-local rows for TP2. This keeps every
    # synthetic registration boundary page aligned while covering exact GiB.
    num_blocks = total // 8192
    return plan_rank_major_layout(
        total_size=total,
        num_blocks=num_blocks,
        world_size=2,
        worker_page_size=4096,
        page_size=4096,
    )


def check_controls() -> None:
    assert not get_rank_sharded_enabled({})
    for value in ("1", "true", "YES", "on"):
        assert get_rank_sharded_enabled({RANK_SHARDED_ENV: value})
    for value in ("0", "false", "NO", "off", ""):
        assert not get_rank_sharded_enabled({RANK_SHARDED_ENV: value})
    try:
        get_rank_sharded_enabled({RANK_SHARDED_ENV: "maybe"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid rank-sharded boolean was accepted")


def check_large_tp2_ranges() -> None:
    for size_gib, per_rank_gib in ((40, 20.0), (45, 22.5)):
        layout = layout_for_gib(size_gib)
        rank0, rank1 = layout.ranges
        assert rank0.offset == 0
        assert rank0.end == rank1.offset
        assert rank1.end == layout.total_size
        assert rank0.size == rank1.size == int(per_rank_gib * 1024**3)
        assert sum(item.size for item in layout.ranges) == size_gib * 1024**3
        assert rank0.end <= rank1.offset


def check_block_addressing() -> None:
    layout = plan_rank_major_layout(
        total_size=64,
        num_blocks=4,
        world_size=2,
        worker_page_size=8,
        page_size=8,
    )
    assert layout.global_row_stride == 16
    assert layout.worker_row_stride == 8
    assert layout.ranges[0].offset == 0
    assert layout.ranges[1].offset == 32

    backing = bytearray(64)
    rank_views = [
        memoryview(backing)[rank_range.offset : rank_range.end]
        for rank_range in layout.ranges
    ]
    for rank_range, rank_view in zip(layout.ranges, rank_views):
        for block in range(layout.num_blocks):
            rank_view[block * rank_range.row_stride] = (
                10 * rank_range.rank + block + 1
            )

    assert [rank_views[0][b * layout.worker_row_stride] for b in range(4)] == [1, 2, 3, 4]
    assert [rank_views[1][b * layout.worker_row_stride] for b in range(4)] == [11, 12, 13, 14]
    assert [backing[layout.ranges[0].block_offset(b)] for b in range(4)] == [1, 2, 3, 4]
    assert [backing[layout.ranges[1].block_offset(b)] for b in range(4)] == [11, 12, 13, 14]
    assert [layout.ranges[0].block_offset(b) for b in range(4)] == [0, 8, 16, 24]
    assert [layout.ranges[1].block_offset(b) for b in range(4)] == [32, 40, 48, 56]


def check_group_atomic_rollback() -> None:
    layout = plan_rank_major_layout(64, 4, 2, 8, 8)
    base_ptr = 1000
    rank0, rank1 = layout.ranges

    runtime0 = FakeRuntime()
    result0 = register_host_chunks(
        runtime0,
        base_ptr + rank0.offset,
        rank0.size,
        rank0.row_stride,
    )
    assert result0.ok
    owned0 = tuple((rank0.offset + off, size) for off, size in result0.chunks)

    runtime1 = FakeRuntime(register_results=[1])
    result1 = register_host_chunks(
        runtime1,
        base_ptr + rank1.offset,
        rank1.size,
        rank1.row_stride,
    )
    assert not result1.ok and result1.error_code == 1
    assert result1.drained_error_code == 1
    assert result1.chunks == ()

    # Production coordination observes the rank-1 failure and rolls rank 0
    # back. No rank is left pinned after the group decision.
    rollback0 = rollback_host_chunks(runtime0, base_ptr, owned0)
    assert rollback0.ok and rollback0.remaining_chunks == ()
    assert runtime0.events == [
        ("register", base_ptr + rank0.offset, rank0.size),
        ("unregister", base_ptr + rank0.offset),
    ]
    assert runtime1.events == [
        ("register", base_ptr + rank1.offset, rank1.size),
        ("drain",),
    ]


def check_cleanup_owns_exact_registered_range() -> None:
    layout = plan_rank_major_layout(128, 8, 2, 8, 8)
    base_ptr = 5000
    rank1 = layout.range_for_rank(1)
    runtime = FakeRuntime()
    result = register_host_chunks(
        runtime,
        base_ptr + rank1.offset,
        rank1.size,
        rank1.row_stride,
        requested_chunk_bytes=16,
    )
    assert result.ok
    absolute = tuple((rank1.offset + off, size) for off, size in result.chunks)
    assert all(rank1.offset <= off < rank1.end for off, _ in absolute)
    assert sum(size for _, size in absolute) == rank1.size

    cleanup = rollback_host_chunks(runtime, base_ptr, absolute)
    assert cleanup.ok
    unregister_ptrs = [event[1] for event in runtime.events if event[0] == "unregister"]
    assert unregister_ptrs == [
        base_ptr + rank1.offset + 48,
        base_ptr + rank1.offset + 32,
        base_ptr + rank1.offset + 16,
        base_ptr + rank1.offset,
    ]
    assert all(base_ptr + rank1.offset <= ptr < base_ptr + rank1.end for ptr in unregister_ptrs)


def main() -> None:
    check_controls()
    check_large_tp2_ranges()
    check_block_addressing()
    check_group_atomic_rollback()
    check_cleanup_owns_exact_registered_range()
    print("KV offload rank-sharded CPU layout checks: PASS")


if __name__ == "__main__":
    main()
