"""Execute the patched CPU-worker constructor with TP2/TP4 topology doubles."""
import ast
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from radiance_kv_offload import plan_rank_major_layout


SOURCE = Path(os.environ.get("RADIANCE_KV_RANK_PATCH_SOURCE",
    str(Path(__file__).resolve().parents[1] / "patch_kv_offload_rank_sharded.py")))


def constructor(upgrade=False):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_cpu_spec")
    call = next(n for n in ast.walk(function) if isinstance(n, ast.Call)
                and len(n.args) >= 4 and isinstance(n.args[3], ast.Constant)
                and n.args[3].value == "rank_sharded=rank_sharded,")
    # Retain the existing exact upstream anchor as the fixture. The method body
    # that executes below is the output of the real overlay, not a rank formula copy.
    block = ast.literal_eval(call.args[2 if upgrade else 1])
    fixture = "import torch\nfrom typing_extensions import override\n"
    if upgrade:
        fixture += "\nfrom radiance_kv_offload import get_rank_sharded_enabled\n"
    fixture += "\nclass Spec:\n    def create_worker(self):\n        mmap_region = None\n"
    fixture += block + "        return mmap_region\n"
    spec = importlib.util.spec_from_file_location("worker_rank_overlay", SOURCE)
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", io.StringIO()):
        overlay.LIB = Path(folder)
        path = overlay.LIB / "vllm/v1/kv_offload/cpu/spec.py"
        path.parent.mkdir(parents=True)
        path.write_text(fixture, encoding="utf-8")
        overlay.patch_cpu_spec()
        output = path.read_text()
        overlay.patch_cpu_spec()
        assert output == path.read_text(), "overlay must remain idempotent"
    tree = ast.parse(output)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "create_worker")
    return ast.Module(body=[method], type_ignores=[]), output


class KVWorkerRankTests(unittest.TestCase):
    def worker(self, world_size, rank, ordinal, *, sharded=False, replicated=False,
               upgrade=False, chunks=4):
        tree, _ = constructor(upgrade)
        world = Mock(return_value=types.SimpleNamespace(rank_in_group=rank))
        region = Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
        scope = dict(torch=types.SimpleNamespace(accelerator=types.SimpleNamespace(
                         current_device_index=lambda: ordinal)),
                     get_world_group=world, get_rank_sharded_enabled=lambda: sharded,
                     SharedOffloadRegion=region, _all_workers_barrier=lambda: None)
        exec(compile(tree, str(SOURCE), "exec"), scope)
        spec = types.SimpleNamespace(_uses_shared_region=lambda: True, num_chunks=chunks,
            config=types.SimpleNamespace(engine_id="synthetic", parallel=types.SimpleNamespace(world_size=world_size)),
            replicated_layout=replicated, kv_bytes_per_chunk=4096 * world_size,
            cpu_page_size_per_worker=4096)
        return scope["create_worker"](spec), world, region

    def test_tp2_reordered_devices_use_logical_rank(self):
        for sharded in (False, True):
            for rank, ordinal in enumerate((1, 0)):
                with self.subTest(sharded=sharded, rank=rank):
                    region, _, _ = self.worker(2, rank, ordinal, sharded=sharded)
                    self.assertEqual(region.rank, rank)

    def test_tp2_and_tp4_isolated_device_zero_do_not_alias_slots(self):
        for size in (2, 4):
            for sharded in (False, True):
                with self.subTest(size=size, sharded=sharded):
                    ranks = [self.worker(size, rank, 0, sharded=sharded)[0].rank for rank in range(size)]
                    self.assertEqual(ranks, list(range(size)))
                    if sharded:
                        layout = plan_rank_major_layout(size * 4 * 4096, 4, size, 4096, 4096)
                        spans = [layout.range_for_rank(rank) for rank in ranks]
                        self.assertEqual(len({span.offset for span in spans}), size)
                        self.assertEqual(sum(span.size for span in spans), layout.total_size)

    def test_tp4_nonconsecutive_device_mapping(self):
        for rank, ordinal in enumerate((2, 0, 3, 1)):
            with self.subTest(rank=rank, ordinal=ordinal):
                self.assertEqual(self.worker(4, rank, ordinal)[0].rank, rank)

    def test_tp4_multinode_local_ordinals_may_repeat(self):
        ranks = [self.worker(4, rank, ordinal)[0].rank for rank, ordinal in enumerate((0, 1, 0, 1))]
        self.assertEqual(ranks, [0, 1, 2, 3])

    def test_outer_data_parallel_rank_offsets_are_removed(self):
        for size in (2, 4):
            with self.subTest(size=size):
                ranks = [self.worker(size, size + rank, 0)[0].rank for rank in range(size)]
                self.assertEqual(ranks, list(range(size)))

    def test_already_patched_images_receive_logical_rank_repair(self):
        for size in (2, 4):
            with self.subTest(size=size):
                self.assertEqual(self.worker(size, size - 1, 0, upgrade=True)[0].rank, size - 1)

    def test_replicated_layout_keeps_shared_slot_zero(self):
        region, world, _ = self.worker(4, 3, 0, replicated=True)
        self.assertEqual(region.rank, 0)
        world.assert_not_called()

    def test_replicated_rank_sharding_still_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.worker(4, 3, 0, replicated=True, sharded=True)

    def test_empty_region_does_not_resolve_rank(self):
        region, world, factory = self.worker(4, 3, 0, chunks=0)
        self.assertIsNone(region)
        world.assert_not_called()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
