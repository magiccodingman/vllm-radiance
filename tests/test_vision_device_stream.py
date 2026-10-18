"""CPU launch-boundary tests with explicit device/stream doubles, not HIP execution."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(os.environ.get("RADIANCE_VISION_DEVICE_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_vit_attn.py")))


class Tensor:
    def __init__(self, pointer, device="cuda:1", shape=(8, 4, 72)):
        self.pointer, self.device, self.shape = pointer, device, shape

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self.shape[0]


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


class VisionDeviceStreamTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Cuda()
        self.output = Tensor(4)
        self.cu = Tensor(5, shape=(2,))
        self.segments = Mock(return_value=self.cu)
        self.launches = []

        def launch(*args):
            self.launches.append((self.cuda.current, args))

        self.native = Mock(side_effect=launch)
        scope = dict(torch=types.SimpleNamespace(cuda=self.cuda, empty_like=Mock(return_value=self.output)),
                     _HEAD_DIM=72, _cu_single=self.segments, _R4D_ATTN=self.native)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_r4d_72"]
        exec(compile(tree, str(SOURCE), "exec"), scope)
        self.forward = scope["_r4d_72"]

    def test_tensor_device_controls_launch_and_nondefault_stream(self):
        tensors = [Tensor(i) for i in (1, 2, 3)]
        self.assertIs(self.forward(*tensors, 0.25), self.output)
        device, args = self.launches[0]
        self.assertEqual(device, "cuda:1")
        self.assertEqual(args, (1, 2, 3, 4, 5, 1, 8, 4, 72, 0.25, 211))
        self.assertEqual(self.cuda.current, "cuda:0")
        self.segments.assert_called_once_with(8, "cuda:1")

    def test_current_tensor_device_keeps_its_current_stream(self):
        self.cuda.current = "cuda:1"
        self.cuda.streams["cuda:1"] = 399
        self.forward(*(Tensor(i) for i in (1, 2, 3)), 0.5)
        self.assertEqual(self.launches[0][0], "cuda:1")
        self.assertEqual(self.launches[0][1][-1], 399)
        self.assertEqual(self.cuda.current, "cuda:1")

    def test_explicit_varlen_segments_keep_bounds_and_device(self):
        cu = Tensor(9, shape=(3,))
        self.forward(*(Tensor(i) for i in (1, 2, 3)), 0.5, cu=cu, max_seqlen=5)
        device, args = self.launches[0]
        self.assertEqual(device, "cuda:1")
        self.assertEqual(args[4:7], (9, 2, 5))
        self.assertEqual(args[-1], 211)
        self.segments.assert_not_called()
        self.assertEqual(self.cuda.current, "cuda:0")

    def test_kernel_failure_restores_caller_device(self):
        self.native.side_effect = RuntimeError("synthetic kernel failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic kernel failure"):
            self.forward(*(Tensor(i) for i in (1, 2, 3)), 0.5)
        self.assertEqual(self.cuda.current, "cuda:0")

    def test_segment_allocation_failure_restores_caller_device(self):
        self.segments.side_effect = RuntimeError("synthetic allocation failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic allocation failure"):
            self.forward(*(Tensor(i) for i in (1, 2, 3)), 0.5)
        self.native.assert_not_called()
        self.assertEqual(self.cuda.current, "cuda:0")


if __name__ == "__main__":
    unittest.main()
