"""Trace real W4 launch wrappers with device/stream doubles; no GPU kernels run."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_DFLASH_W4_SOURCE", ROOT / "radiance_w4.py"))


class Tensor:
    sequence = 0
    def __init__(self, shape, device=1, dtype="fp16"):
        self.shape, self.device, self.dtype = shape, device, dtype
        Tensor.sequence += 1
        self.pointer = Tensor.sequence
    def data_ptr(self):
        return self.pointer


class Cuda:
    def __init__(self):
        self.current = 0
        self.streams = {index: 1000 + index for index in range(4)}
    @contextmanager
    def device(self, index):
        previous, self.current = self.current, index
        try:
            yield
        finally:
            self.current = previous
    def current_stream(self, device=None):
        return SimpleNamespace(cuda_stream=self.streams[self.current if device is None else device])


class DFlashW4DeviceStreamTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Cuda()
        self.calls = []
        self.failure = None
        def native(name):
            def launch(*args):
                self.calls.append((name, self.cuda.current, args))
                if self.failure == name:
                    raise RuntimeError("synthetic " + name + " failure")
            return launch
        torch = SimpleNamespace(Tensor=Tensor, bfloat16="bf16", float32="fp32", int8="i8", cuda=self.cuda,
                                empty=lambda shape, device, dtype: Tensor(shape, device, dtype))
        nodes = [node for node in ast.walk(ast.parse(SOURCE.read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name in ("w4_gemm", "w4a8_gemm")]
        for node in nodes:
            node.decorator_list = []
        self.scope = dict(torch=torch, _GEMM=native("W4"), _QACT=native("QACT"), _A8=native("W4A8"))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), self.scope)
        self.a, self.wq, self.wsz = Tensor((3, 128)), Tensor((16, 32)), Tensor((16, 1))

    def run_op(self, name="w4_gemm"):
        return self.scope[name](self.a, self.wq, self.wsz, 16, 1, 2, 3, 4, 5)

    def check_launches(self, names):
        self.assertEqual([call[0] for call in self.calls], names)
        for name, device, args in self.calls:
            self.assertEqual(device, self.a.device, name)
            self.assertEqual(args[-1], self.cuda.streams[self.a.device], name)

    def test_w4_uses_tensor_device(self):
        self.run_op()
        self.check_launches(["W4"])
        self.assertEqual(self.cuda.current, 0)

    def test_activation_quantization_and_w4a8_share_tensor_stream(self):
        self.run_op("w4a8_gemm")
        self.check_launches(["QACT", "W4A8"])
        self.assertEqual(self.cuda.current, 0)

    def test_nondefault_stream_on_tensor_device_is_used(self):
        self.cuda.streams[1] = 7731
        self.run_op("w4a8_gemm")
        self.check_launches(["QACT", "W4A8"])

    def test_four_device_metadata_selects_last_device(self):
        self.a.device = self.wq.device = self.wsz.device = 3
        self.run_op()
        self.check_launches(["W4"])

    def test_w4_failure_restores_callers_device(self):
        self.failure = "W4"
        with self.assertRaisesRegex(RuntimeError, "synthetic W4"):
            self.run_op()
        self.check_launches(["W4"])
        self.assertEqual(self.cuda.current, 0)

    def test_activation_quantization_failure_restores_callers_device(self):
        self.failure = "QACT"
        with self.assertRaisesRegex(RuntimeError, "synthetic QACT"):
            self.run_op("w4a8_gemm")
        self.check_launches(["QACT"])
        self.assertEqual(self.cuda.current, 0)

    def test_w4a8_failure_restores_callers_device(self):
        self.failure = "W4A8"
        with self.assertRaisesRegex(RuntimeError, "synthetic W4A8"):
            self.run_op("w4a8_gemm")
        self.check_launches(["QACT", "W4A8"])
        self.assertEqual(self.cuda.current, 0)

    def test_matching_device_preserves_launch_geometry(self):
        self.cuda.current = 1
        output = self.run_op()
        self.check_launches(["W4"])
        self.assertEqual(output.shape, (3, 16))
        self.assertEqual(output.dtype, "bf16")
        self.assertEqual(self.calls[0][2][4:-1], (3, 128, 16, 1, 2, 3, 4, 5))

    def test_nested_device_context_is_restored(self):
        with self.cuda.device(2):
            self.run_op("w4a8_gemm")
            self.check_launches(["QACT", "W4A8"])
            self.assertEqual(self.cuda.current, 2)
        self.assertEqual(self.cuda.current, 0)


if __name__ == "__main__":
    unittest.main()
