"""Threaded CPU rendezvous tests of real startup logic with synthetic IPC resources.

This checks protocol agreement, not native collectives, Gloo, HIP or GPU memory.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch


SOURCE = Path(os.environ.get("RADIANCE_AR_QUANT_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_allreduce.py")))
LOCAL = threading.local()


class Rendezvous:
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=5)
        self.values = [None, None]
        self.calls = [0, 0]

    def gather(self, output, value, rank):
        self.values[rank] = value
        self.calls[rank] += 1
        self.barrier.wait()
        output[:] = self.values
        self.barrier.wait()


def constructor(rank, group, flag, failure):
    class Buffer:
        def data_ptr(self):
            return 4096

    def empty(*args, **kwargs):
        if failure == "allocation":
            raise RuntimeError("synthetic quant pack allocation failure")
        return Buffer()

    torch = types.SimpleNamespace(Tensor=Buffer, bfloat16=object(), float16=object(), float32=object(),
        uint8=object(), int32=object(), device=lambda name: name,
        cuda=types.SimpleNamespace(set_device=lambda device: None),
        zeros=lambda *args, **kwargs: Buffer(), empty=empty)
    dist = types.SimpleNamespace(get_world_size=lambda group: 2, get_rank=lambda group: rank,
        all_gather_object=lambda output, value, group: group.gather(output, value, rank))
    scope = dict(torch=torch, dist=dist, contextmanager=contextmanager,
        os=types.SimpleNamespace(environ={"RADIANCE_USE_R4D_AR_QUANT": flag}),
        _MAX_BYTES=1024 * 1024, _QUANT_MIN_BYTES=128 * 1024, _log=lambda message: None)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RadianceAllreduce"]
    exec(compile(tree, str(SOURCE), "exec"), scope)
    LOCAL.failure = failure
    return scope["RadianceAllreduce"](group, rank)


class AllreduceQuantConsensusTests(unittest.TestCase):
    def run_groups(self, settings):
        ext = types.ModuleType("r4d")
        ext.AR_MAX_BLOCKS, ext.AR_WHT6_GROUP, ext.AR_WHT6_BITS = 24, 256, 6
        ext.ar_oneshot_2rank_exact, ext.ar_oneshot_2rank_wht6 = object(), object()
        ext.ar_ipc_alloc = lambda size, fine: (4096, b"synthetic IPC handle")
        ext.ar_ipc_open = lambda handle: 8192

        def select(operation, *, exact, **geometry):
            if not exact and LOCAL.failure == "selection":
                return None
            return "ar_oneshot_2rank_exact" if exact else "ar_oneshot_2rank_wht6"

        ext.select = select
        groups = [Rendezvous() for _ in settings]
        with patch.dict(sys.modules, {"r4d": ext}), ThreadPoolExecutor(max_workers=2 * len(groups)) as pool:
            futures = [pool.submit(constructor, rank, group, flag, failure)
                       for group, values in zip(groups, settings)
                       for rank, (flag, failure) in enumerate(values)]
            result = [future.result(timeout=10) for future in futures]
        self.assertTrue(all(not comm.disabled for comm in result), "exact IPC setup must remain enabled")
        self.assertTrue(all(comm._ar_exact is ext.ar_oneshot_2rank_exact for comm in result))
        for group in groups:
            self.assertEqual(group.calls[0], group.calls[1], "every rank must rendezvous equally")
        return result

    def assert_exact_fallback(self, values):
        comms = self.run_groups([values])
        self.assertEqual([comm.ar_quant for comm in comms], [False, False])
        self.assertTrue(all(comm._locpk is None for comm in comms))

    def test_two_ready_ranks_keep_compressed_protocol(self):
        comms = self.run_groups([[('1', None), ('1', None)]])
        self.assertEqual([comm.ar_quant for comm in comms], [True, True])
        self.assertTrue(all(comm._locpk is not None for comm in comms))

    def test_two_disabled_ranks_keep_exact_protocol(self):
        self.assert_exact_fallback([('0', None), ('0', None)])

    def test_disagreeing_rank_controls_fall_back_together(self):
        self.assert_exact_fallback([('1', None), ('0', None)])

    def test_one_rank_quant_allocation_failure_falls_back_together(self):
        self.assert_exact_fallback([('1', None), ('1', 'allocation')])

    def test_one_rank_missing_quant_kernel_falls_back_together(self):
        self.assert_exact_fallback([('1', 'selection'), ('1', None)])

    def test_four_workers_in_two_tp2_groups_negotiate_independently(self):
        comms = self.run_groups([[('1', None), ('1', None)], [('1', None), ('1', 'allocation')]])
        self.assertEqual([comm.ar_quant for comm in comms], [True, True, False, False])
        self.assertTrue(all(comm._locpk is None for comm in comms[2:]))


if __name__ == "__main__":
    unittest.main()
