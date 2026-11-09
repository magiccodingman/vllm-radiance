"""Real two/four-process CPU group checks; native allocation/IPC remain doubles."""
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SOURCE = Path(os.environ.get("RADIANCE_AR_QUANT_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_allreduce.py")))


def worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=30))
    try:
        spec = importlib.util.spec_from_file_location("tested_allreduce_gloo", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.environ["RADIANCE_USE_R4D"] = "1"
        os.environ["RADIANCE_USE_R4D_AR"] = "1"
        os.environ["RADIANCE_USE_R4D_AR_QUANT"] = "1"
        if size == 4:
            # Exercise the real Radiance hook while the stock communicator uses
            # real Gloo. TP4 must not initialize a two-peer native communicator.
            class Communicator:
                def __init__(self):
                    self.unique_name, self.world_size = "tp:0", size
                    self.cpu_group, self.device = dist.group.WORLD, torch.device(f"cuda:{rank}")

                def all_reduce(self, value):
                    result = value.clone()
                    dist.all_reduce(result)
                    return result

            calls = []
            native_class = module.RadianceAllreduce
            def forbidden_native(*args):
                calls.append(args)
                raise AssertionError("TP4 entered the two-peer native constructor")

            stub = types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
            stub.CudaCommunicator = Communicator
            with patch.dict(sys.modules, {stub.__name__: stub}):
                module.RadianceAllreduce = forbidden_native
                module.install_custom_ar()
                communicator = Communicator()
                assert not calls and communicator.radiance_comm is None
                actual = communicator.all_reduce(torch.tensor([float(rank + 1)]))
                assert actual.item() == 10.0
                module.RadianceAllreduce = native_class

        # Every process creates groups in the same order. Four workers form two
        # independent TP2 groups; only the last pair has an injected pack failure.
        groups = [dist.new_group([first, first + 1], backend="gloo") for first in range(0, size, 2)]
        group = groups[rank // 2]
        ext = types.ModuleType("r4d")
        ext.AR_MAX_BLOCKS, ext.AR_WHT6_GROUP, ext.AR_WHT6_BITS = 24, 256, 6
        ext.ar_oneshot_2rank_exact, ext.ar_oneshot_2rank_wht6 = object(), object()
        ext.select = lambda operation, exact, **kwargs: "ar_oneshot_2rank_exact" if exact else "ar_oneshot_2rank_wht6"
        ext.ar_ipc_alloc = lambda size, fine: (4096, f"rank-{rank}".encode())
        ext.ar_ipc_open = lambda handle: 8192

        def pack(*args, **kwargs):
            if rank == size - 1:
                raise RuntimeError("synthetic one-rank pack allocation failure")
            return torch.empty(1, dtype=torch.uint8)

        # Replace only the module's allocator facade. torch.distributed itself
        # retains real Torch tensors and real Gloo all_gather_object collectives.
        module.torch = types.SimpleNamespace(device=torch.device, uint8=torch.uint8, int32=torch.int32,
            cuda=types.SimpleNamespace(set_device=lambda device: None), empty=pack,
            zeros=lambda count, **kwargs: torch.zeros(count, dtype=torch.int32))
        with patch.dict(sys.modules, {"r4d": ext}):
            communicator = module.RadianceAllreduce(group, rank)
        expected_quant = size == 4 and rank < 2
        assert not communicator.disabled
        assert communicator.rank == rank % 2
        assert communicator.ar_quant is expected_quant
        assert (communicator._locpk is not None) is expected_quant
        result = torch.tensor([float(rank + 1)])
        dist.all_reduce(result, group=group)
        assert result.item() == (3.0 if rank < 2 else 7.0)
    finally:
        dist.destroy_process_group()


class AllreduceGlooTests(unittest.TestCase):
    def run_world(self, size):
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = (Path(folder) / "cpu-rendezvous").as_uri()
            mp.spawn(worker, args=(size, rendezvous), nprocs=size, join=True)

    def test_two_process_group_agrees_after_one_rank_pack_failure(self):
        self.run_world(2)

    def test_four_process_fallback_and_independent_tp2_protocols(self):
        self.run_world(4)


if __name__ == "__main__":
    unittest.main()
