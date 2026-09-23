#!/usr/bin/env python3
"""Verify upstream mmap lifetime, then apply Radiance's separate layout extension.

v0.30 owns upstream 4c58a0c398b056b135b98bd93c644945be7e3109:
all workers map, rendezvous, unlink while mappings remain alive. No lifecycle
implementation is replaced here. Historical backport remains in Git history.
"""
import ast
import sysconfig
from pathlib import Path

from patch_kv_offload_rank_sharded import main as patch_rank_sharded_layout

lib = Path(sysconfig.get_paths()["purelib"])
contracts = {
    "vllm/v1/kv_offload/cpu/shared_offload_region.py": (
        "barrier: Callable[[], None] | None = None",
        "Failed to release peers waiting at the mmap barrier",
        "Unlinked mmap file %s", "os.unlink(self.mmap_path)"),
    "vllm/v1/kv_offload/cpu/spec.py": (
        "def _all_workers_barrier() -> None:", "barrier=_all_workers_barrier"),
}
for relative, required in contracts.items():
    source = (lib / relative).read_text()
    ast.parse(source)
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Upstream mmap lifetime contract drift: {relative}: {fragment}")
print("[radiance] upstream mmap rendezvous/unlink contract verified; no lifecycle rewrite")
patch_rank_sharded_layout()
