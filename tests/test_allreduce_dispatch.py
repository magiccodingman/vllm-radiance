"""CPU metadata tests for the real all-reduce dispatch, without native imports."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


BF16, FP16, FP32 = object(), object(), object()
SOURCE = Path(os.environ.get("RADIANCE_AR_DISPATCH_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_allreduce.py")))


class Tensor:
    def __init__(self, *, device="cuda:1", dtype=BF16, count=16, pointer=4096, contiguous=True, width=2):
        self.device, self.dtype, self.count = device, dtype, count
        self.pointer, self.contiguous, self.width = pointer, contiguous, width
        self.is_cuda = device.startswith("cuda:")

    def numel(self):
        return self.count

    def element_size(self):
        return self.width

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous


class Cuda:
    def __init__(self):
        self.current = "cuda:0"
        self.streams = {"cuda:0": 100, "cuda:1": 211}

    @contextmanager
    def device(self, device):
        previous = self.current
        self.current = device
        try:
            yield
        finally:
            self.current = previous

    def current_stream(self, device=None):
        return types.SimpleNamespace(cuda_stream=self.streams[device or self.current])


def load():
    cuda = Cuda()
    torch = types.SimpleNamespace(Tensor=Tensor, bfloat16=BF16, float16=FP16, float32=FP32,
                                  cuda=cuda, empty_like=Mock(return_value=Tensor(pointer=8192)))
    scope = dict(torch=torch, os=types.SimpleNamespace(environ={}), contextmanager=contextmanager,
                 _log=lambda message: None, _DTYPE_CODE={BF16: 0, FP16: 1, FP32: 2})
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in {"RadianceAllreduce", "install_custom_ar"}]
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope, cuda


class AllreduceOperandTests(unittest.TestCase):
    def setUp(self):
        self.scope, self.cuda = load()
        cls = self.scope["RadianceAllreduce"]
        self.comm = cls.__new__(cls)
        self.comm.disabled, self.comm.device, self.comm.max_bytes = False, "cuda:1", 1024
        self.comm._quant_ok = Mock(return_value=False)
        self.comm._nblocks, self.comm._nblocks_q = Mock(return_value=4), Mock(return_value=8)
        self.comm._peer_scratch, self.comm._scratch = 10, 11
        self.comm._peer_flags, self.comm._flags = 12, 13
        self.comm._seq, self.comm._locpk = Tensor(pointer=14), Tensor(pointer=15)
        self.comm.slot16, self.comm.nt, self.comm.qnt = 64, 1024, 1024
        self.comm.drain, self.comm.acq = 3, 0
        self.launches = []
        self.comm._ar_exact = Mock(side_effect=lambda *args: self.launches.append((self.cuda.current, args)))
        self.comm._ar_wht6 = Mock(side_effect=lambda *args: self.launches.append((self.cuda.current, args)))

    def test_cpu_and_other_device_inputs_decline_native(self):
        for device in ("cpu", "cuda:0", "cuda:2"):
            with self.subTest(device=device):
                tensor = Tensor(device=device)
                self.assertFalse(self.comm.should_custom_ar(tensor))
                self.assertIsNone(self.comm.custom_all_reduce(tensor))
        self.comm._ar_exact.assert_not_called()

    def test_contiguous_unaligned_views_decline_native(self):
        for offset in (2, 4, 6, 8, 12):
            with self.subTest(offset=offset):
                self.assertFalse(self.comm.should_custom_ar(Tensor(pointer=4096 + offset)))

    def test_aligned_contiguous_dtypes_remain_eligible(self):
        for dtype, width in ((BF16, 2), (FP16, 2), (FP32, 4)):
            with self.subTest(width=width):
                self.assertTrue(self.comm.should_custom_ar(Tensor(dtype=dtype, width=width)))

    def test_existing_size_dtype_and_contiguity_boundaries(self):
        for tensor in (Tensor(count=0), Tensor(count=3), Tensor(count=1024),
                       Tensor(dtype=object()), Tensor(contiguous=False)):
            with self.subTest(metadata=vars(tensor)):
                self.assertFalse(self.comm.should_custom_ar(tensor))

    def test_exact_launch_uses_communicator_device_and_current_stream(self):
        result = self.comm.custom_all_reduce(Tensor())
        device, arguments = self.launches[0]
        self.assertEqual(device, "cuda:1")
        self.assertEqual(arguments[10], 211)
        self.assertIs(result, self.scope["torch"].empty_like.return_value)
        self.assertEqual(self.cuda.current, "cuda:0")

    def test_quant_launch_uses_communicator_device_and_current_stream(self):
        self.comm._quant_ok.return_value = True
        self.comm.custom_all_reduce(Tensor())
        device, arguments = self.launches[0]
        self.assertEqual(device, "cuda:1")
        self.assertEqual(arguments[12], 211)
        self.assertEqual(self.cuda.current, "cuda:0")

    def test_native_exception_restores_callers_device(self):
        self.comm._ar_exact.side_effect = RuntimeError("synthetic launch failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic launch failure"):
            self.comm.custom_all_reduce(Tensor())
        self.assertEqual(self.cuda.current, "cuda:0")


class AllreduceGroupTests(unittest.TestCase):
    def setUp(self):
        self.scope, _ = load()
        self.stock = Mock(return_value=object())
        stock = self.stock

        class Communicator:
            def __init__(self, name, size):
                self.unique_name, self.world_size = name, size
                self.cpu_group, self.device, self.ca_comm = object(), "cuda:1", object()

            def all_reduce(self, value):
                return stock(value)

        self.comm_class = Communicator
        native = types.SimpleNamespace(disabled=False, should_custom_ar=Mock(return_value=True),
                                       custom_all_reduce=Mock(return_value=object()))
        self.factory = Mock(return_value=native)
        self.scope["RadianceAllreduce"] = self.factory
        module = types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
        module.CudaCommunicator = Communicator
        with patch.dict(sys.modules, {module.__name__: module}):
            self.scope["install_custom_ar"]()

    def test_tp2_selects_native_without_replacing_upstream_ca_comm(self):
        comm = self.comm_class("tp:0", 2)
        ca_comm = comm.ca_comm
        result = comm.all_reduce("payload")
        self.assertIs(result, self.factory.return_value.custom_all_reduce.return_value)
        self.assertIs(comm.ca_comm, ca_comm)
        self.factory.assert_called_once_with(comm.cpu_group, comm.device)
        self.stock.assert_not_called()

    def test_tp4_preserves_upstream_fallback(self):
        comm = self.comm_class("tp:0", 4)
        self.assertIs(comm.all_reduce("payload"), self.stock.return_value)
        self.factory.assert_not_called()
        self.stock.assert_called_once_with("payload")

    def test_other_parallel_groups_cannot_attach_tp_kernel(self):
        for name in ("etp:0", "pp:0", "dp:0", "pcp:0", "notp:0", "tp_extra:0"):
            with self.subTest(name=name):
                comm = self.comm_class(name, 2)
                self.assertIs(comm.all_reduce("payload"), self.stock.return_value)
        self.factory.assert_not_called()

    def test_single_rank_preserves_upstream_fallback(self):
        comm = self.comm_class("tp:0", 1)
        self.assertIs(comm.all_reduce("payload"), self.stock.return_value)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
