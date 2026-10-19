"""Execute dispatch/launch metadata with doubles; no native GEMM or GPU runs."""
import ast
from contextlib import contextmanager
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('RADIANCE_SKINNY_SOURCE', ROOT / 'radiance_gemm.py'))


class Tensor:
    def __init__(self, shape, device=1, dtype='bf16', contiguous=True):
        self.shape, self.device, self.dtype = tuple(shape), device, dtype
        self.ndim, self.is_cuda, self.contiguous = len(shape), device != 'cpu', contiguous
    def dim(self):
        return self.ndim
    def numel(self):
        return math.prod(self.shape)
    def reshape(self, *shape):
        return Tensor(shape, self.device, self.dtype, self.contiguous)
    def is_contiguous(self):
        return self.contiguous
    def data_ptr(self):
        return id(self)


class Cuda:
    def __init__(self):
        self.current = 0
    @contextmanager
    def device(self, device):
        previous, self.current = self.current, device
        try:
            yield
        finally:
            self.current = previous
    def current_stream(self, device=None):
        return SimpleNamespace(cuda_stream=7000 + (self.current if device is None else device))


class SkinnyGemmContractTests(unittest.TestCase):
    def setUp(self):
        self.cuda, self.calls, self.fail_launch = Cuda(), [], False
        def launch(*args):
            self.calls.append((self.cuda.current, args))
            if self.fail_launch:
                raise RuntimeError('synthetic native launch failure')
        nodes = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('config_for', 'maybe_gemm')]
        torch = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', cuda=self.cuda,
                                empty=lambda shape, device, dtype: Tensor(shape, device, dtype))
        self.scope = dict(torch=torch, sys=SimpleNamespace(stderr=io.StringIO()), _GEMM=launch,
                          M_MIN=6, _M_MAX=64, _CFG={(256, 2048): (1, 16, 1)}, _seen=set())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.scope)
        self.x, self.weight = Tensor((6, 2048)), Tensor((256, 2048))

    def run_op(self):
        return self.scope['maybe_gemm'](self.x, self.weight)

    def decline(self):
        self.assertIsNone(self.run_op())
        self.assertEqual(self.calls, [])

    def test_tensor_device_and_stream_control_launch(self):
        self.run_op()
        self.assertEqual(self.calls[0][0], 1)
        self.assertEqual(self.calls[0][1][-1], 7001)
        self.assertEqual(self.cuda.current, 0)

    def test_four_device_metadata_selects_ordinal_three(self):
        self.x.device = self.weight.device = 3
        self.run_op()
        self.assertEqual(self.calls[0][0], 3)
        self.assertEqual(self.calls[0][1][-1], 7003)

    def test_cpu_operands_decline(self):
        self.x = Tensor((6, 2048), device='cpu')
        self.weight = Tensor((256, 2048), device='cpu')
        self.decline()

    def test_mixed_device_operands_decline(self):
        self.weight.device = 2
        self.decline()

    def test_wrong_dtype_operands_decline(self):
        for target in (self.x, self.weight):
            target.dtype = 'fp32'
            self.decline()
            target.dtype = 'bf16'

    def test_bad_operand_ranks_decline_before_indexing(self):
        self.x = Tensor(())
        self.decline()
        self.x, self.weight = Tensor((6, 2048)), Tensor((2048,))
        self.decline()

    def test_unmeasured_and_noncontiguous_shapes_decline(self):
        self.x = Tensor((5, 2048))
        self.decline()
        self.x = Tensor((6, 2048), contiguous=False)
        self.decline()

    def test_native_exception_restores_nested_caller_device(self):
        self.fail_launch = True
        with self.cuda.device(2):
            with self.assertRaisesRegex(RuntimeError, 'synthetic native'):
                self.run_op()
            self.assertEqual(self.cuda.current, 2)
        self.assertEqual(self.cuda.current, 0)

    def test_supported_leading_dimensions_keep_shape_and_geometry(self):
        self.x = Tensor((2, 3, 2048))
        self.cuda.current = 1
        output = self.run_op()
        self.assertEqual(output.shape, (2, 3, 256))
        self.assertEqual(self.calls[0][1][3:-1], (6, 2048, 256, 1, 16, 1))

    def test_missing_native_kernel_declines(self):
        self.scope['_GEMM'] = None
        self.decline()


if __name__ == '__main__':
    unittest.main()
