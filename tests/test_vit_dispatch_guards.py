"""Execute the actual dispatch functions with CPU metadata doubles, without HIP imports.

These tests prove fallback decisions and argument forwarding, not kernel numerics.
"""

import ast
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BF16 = object()
BOOL = object()
SOURCE = Path(os.environ.get(
    "RADIANCE_VIT_TEST_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_vit_attn.py"),
))


class Tensor:
    def __init__(self, shape=(1, 8, 4, 72), dtype=BF16, device="cuda:0", cuda=True):
        self.shape, self.dtype, self.device, self.is_cuda = shape, dtype, device, cuda

    def dim(self):
        return len(self.shape)

    def __getitem__(self, index):
        return Tensor(self.shape[1:], self.dtype, self.device, self.is_cuda)

    def contiguous(self):
        return self


class ViTDispatchTests(unittest.TestCase):
    def setUp(self):
        wanted = {"_supports_r4d", "apply_flash", "radiance_vit_flash_attn"}
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        self.stock = Mock(return_value=object())
        self.native = Mock(return_value=Mock())
        self.gemma_native = Mock(return_value=object())
        self.stack = Mock(return_value=object())
        self.scope = dict(torch=types.SimpleNamespace(bfloat16=BF16, bool=BOOL, stack=self.stack),
                          _HEAD_DIM=72, _orig_apply_sdpa=self.stock,
                          _r4d_72=self.native, _r4d_bhsd=self.gemma_native)
        exec(compile(tree, str(SOURCE), "exec"), self.scope)
        self.eager = Mock(return_value=(object(), None))
        module = types.ModuleType("transformers.models.gemma4.modeling_gemma4")
        module.eager_attention_forward = self.eager
        self.modules = patch.dict(sys.modules, {module.__name__: module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def apply(self, q=None, k=None, v=None, **kwargs):
        operands = [q or Tensor(), k or Tensor(), v or Tensor()]
        return self.scope["apply_flash"](*operands, **kwargs), operands

    def gemma(self, q=None, k=None, v=None, **kwargs):
        operands = [q or Tensor(), k or Tensor(), v or Tensor()]
        module = types.SimpleNamespace(num_key_value_groups=1)
        result = self.scope["radiance_vit_flash_attn"](module, *operands, **kwargs)
        return result, module, operands

    def test_matching_bf16_self_attention_uses_native(self):
        result, _ = self.apply(scale=0.25)
        self.native.assert_called_once()
        self.assertEqual(self.native.call_args.args[-1], 0.25)
        self.assertIs(result, self.native.return_value.unsqueeze.return_value)
        self.stock.assert_not_called()

    def test_matching_batched_self_attention_uses_native(self):
        tensors = [Tensor((2, 8, 4, 72)) for _ in range(3)]
        result, _ = self.apply(*tensors)
        self.assertEqual(self.native.call_count, 2)
        self.assertIs(result, self.stack.return_value)
        self.stock.assert_not_called()

    def test_key_and_value_dtype_mismatch_falls_back(self):
        for operand in ("k", "v"):
            with self.subTest(operand=operand):
                result, tensors = self.apply(**{operand: Tensor(dtype=object())}, scale=0.3)
                self.assertIs(result, self.stock.return_value)
                self.stock.assert_called_with(*tensors, scale=0.3, enable_gqa=False)
        self.native.assert_not_called()

    def test_key_and_value_shape_mismatch_falls_back(self):
        for operand in ("k", "v"):
            for shape in ((1, 8, 1, 72), (1, 3, 4, 72), (2, 8, 4, 72), (1, 8, 4, 64), (8, 4, 72)):
                with self.subTest(operand=operand, shape=shape):
                    result, _ = self.apply(**{operand: Tensor(shape)})
                    self.assertIs(result, self.stock.return_value)
        self.native.assert_not_called()

    def test_key_and_value_device_mismatch_falls_back(self):
        for operand in ("k", "v"):
            for device, cuda in (("cpu", False), ("cuda:1", True)):
                with self.subTest(operand=operand, device=device):
                    result, _ = self.apply(**{operand: Tensor(device=device, cuda=cuda)})
                    self.assertIs(result, self.stock.return_value)
        self.native.assert_not_called()

    def test_empty_dimensions_fall_back(self):
        for shape in ((0, 8, 4, 72), (1, 0, 4, 72), (1, 8, 0, 72)):
            with self.subTest(shape=shape):
                result, _ = self.apply(*(Tensor(shape) for _ in range(3)))
                self.assertIs(result, self.stock.return_value)
        self.native.assert_not_called()

    def test_existing_unsupported_query_and_gqa_fall_back(self):
        for q in (Tensor(dtype=object()), Tensor(device="cpu", cuda=False), Tensor((1, 8, 4, 64))):
            result, _ = self.apply(q=q)
            self.assertIs(result, self.stock.return_value)
        result, tensors = self.apply(enable_gqa=True)
        self.assertIs(result, self.stock.return_value)
        self.stock.assert_called_with(*tensors, scale=None, enable_gqa=True)
        self.native.assert_not_called()

    def test_gemma_matching_operands_use_native(self):
        result, _, tensors = self.gemma(attention_mask=None, scaling=0.2)
        self.gemma_native.assert_called_once_with(*tensors, 0.2)
        self.assertEqual(result, (self.gemma_native.return_value, None))
        self.eager.assert_not_called()

    def test_gemma_mismatched_operands_use_eager(self):
        for operand in ("k", "v"):
            for bad in (Tensor(dtype=object()), Tensor((1, 1, 4, 72)),
                        Tensor(device="cuda:1"), Tensor(device="cpu", cuda=False)):
                with self.subTest(operand=operand, bad=vars(bad)):
                    result, module, tensors = self.gemma(**{operand: bad}, attention_mask=None,
                                                        scaling=0.2, marker="forwarded")
                    self.assertIs(result, self.eager.return_value)
                    self.eager.assert_called_with(module, *tensors, None, dropout=0.0,
                                                  scaling=0.2, softcap=None, marker="forwarded")
        self.gemma_native.assert_not_called()

    def test_gemma_dropout_uses_eager(self):
        result, module, tensors = self.gemma(attention_mask=None, dropout=0.25)
        self.assertIs(result, self.eager.return_value)
        self.eager.assert_called_once_with(module, *tensors, None, dropout=0.25,
                                          scaling=None, softcap=None)
        self.gemma_native.assert_not_called()

    def test_gemma_mask_and_softcap_keep_eager(self):
        additive_mask = Tensor(shape=(1, 1, 4, 4), dtype=object())
        for kwargs in ({"attention_mask": additive_mask}, {"attention_mask": None, "softcap": 0.5}):
            with self.subTest(kwargs=kwargs):
                result, _, _ = self.gemma(**kwargs)
                self.assertIs(result, self.eager.return_value)
        self.gemma_native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
