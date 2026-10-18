"""Execute the real lazy-head control flow without a GPU or Torch import."""
import ast
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_DRAFT_LAZY_SOURCE", ROOT / "radiance_drafthead.py"))


def functions():
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in ("_apply_head_lazy", "_quantize_head_now")]
    scope = dict(sys=sys, types=types, GROUP=128, BITS=2)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


class Weight:
    def __init__(self, array):
        self.array = array
        self.data = self
        self.shape = array.shape

    def abs(self):
        return Weight(np.abs(self.array))

    def max(self):
        return self.array.max()


class Processor:
    def _apply_head(self, head, hidden, bias):
        self.native_calls += 1
        result = hidden @ head.weight.array.T
        return result if bias is None else result + bias


class DraftHeadLazyFallbackTests(unittest.TestCase):
    def setUp(self):
        self.scope = functions()
        self.processor = Processor()
        self.processor.native_calls = 0
        self.processor._apply_head = types.MethodType(self.scope["_apply_head_lazy"], self.processor)
        self.head = types.SimpleNamespace(weight=Weight(np.arange(21, dtype=np.float32).reshape(3, 7)))
        self.hidden = np.arange(14, dtype=np.float32).reshape(2, 7)
        self.log = io.StringIO()
        self.capture = redirect_stderr(self.log)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def assert_native(self, bias=None):
        expected = self.hidden @ self.head.weight.array.T
        if bias is not None:
            expected += bias
        np.testing.assert_array_equal(self.processor._apply_head(self.head, self.hidden, bias), expected)

    def test_unsupported_hidden_size_uses_native_instead_of_recursing(self):
        # The actual packer rejects K=7 before touching Torch or allocating buffers.
        self.assert_native(np.array([1, 2, 3], dtype=np.float32))
        self.assertEqual(self.processor.native_calls, 1)

    def test_declined_packing_is_not_retried_every_request(self):
        original = self.scope["_quantize_head_now"]
        calls = []
        def decline(*args):
            calls.append(1)
            return original(*args)
        self.scope["_quantize_head_now"] = decline
        self.assert_native()
        self.assert_native()
        self.assertEqual(len(calls), 1)

    def check_incomplete_quarter_groups(self, width):
        self.head.weight = Weight(np.arange(3 * width, dtype=np.float32).reshape(3, width))
        self.hidden = np.arange(2 * width, dtype=np.float32).reshape(2, width)
        # Run the real packer's guard before any optional Torch allocation.
        # K=256/768 passed the old half-plane check; the INT2 kernel's
        # NG=(K//4)//128 then processed zero/only one group per quarter.
        status = self.scope["_quantize_head_now"](self.processor, self.head)
        self.assertEqual(status, f"hidden size {width} not a multiple of 512")
        self.assert_native()
        self.assertNotIn("packing failed", self.log.getvalue())

    def test_int2_rejects_zero_complete_groups_per_quarter(self):
        self.check_incomplete_quarter_groups(256)

    def test_int2_rejects_truncated_groups_per_quarter(self):
        self.check_incomplete_quarter_groups(768)

    def test_initialization_exception_restores_native(self):
        def fail(*args):
            raise MemoryError("synthetic optional packing failure")
        self.scope["_quantize_head_now"] = fail
        self.assert_native()
        self.assert_native()
        self.assertIn("synthetic optional packing failure", self.log.getvalue())

    def test_partial_method_installation_is_reverted_on_setup_error(self):
        def fail(processor, head):
            processor._apply_head = lambda *args: self.fail("partially installed head used")
            raise RuntimeError("synthetic late initialization failure")
        self.scope["_quantize_head_now"] = fail
        self.assert_native()

    def test_empty_shared_head_can_be_initialized_on_a_later_call(self):
        self.head.weight.array.fill(0)
        calls = []
        def arm(processor, head):
            calls.append(1)
            processor._apply_head = lambda head, hidden, bias: np.full((2, 3), 17, dtype=np.float32)
            return "armed"
        self.scope["_quantize_head_now"] = arm
        self.assert_native()
        self.assertEqual(calls, [])
        self.head.weight.array.fill(1)
        np.testing.assert_array_equal(self.processor._apply_head(self.head, self.hidden, None), np.full((2, 3), 17))
        self.assertEqual(len(calls), 1)

    def test_successful_setup_preserves_the_fast_method(self):
        def arm(processor, head):
            processor._apply_head = lambda *args: "fast result"
            return "armed"
        self.scope["_quantize_head_now"] = arm
        self.assertEqual(self.processor._apply_head(self.head, self.hidden, None), "fast result")

    def test_post_setup_kernel_exception_is_not_hidden(self):
        def kernel(*args):
            raise ArithmeticError("synthetic execution error")
        def arm(processor, head):
            processor._apply_head = kernel
            return "armed"
        self.scope["_quantize_head_now"] = arm
        with self.assertRaisesRegex(ArithmeticError, "synthetic execution error"):
            self.processor._apply_head(self.head, self.hidden, None)

    def test_native_fallback_respects_processor_subclass(self):
        class Specialized(Processor):
            def _apply_head(self, head, hidden, bias):
                return "specialized native result"
        processor = Specialized()
        processor._apply_head = types.MethodType(self.scope["_apply_head_lazy"], processor)
        self.assertEqual(processor._apply_head(self.head, self.hidden, None), "specialized native result")


if __name__ == "__main__":
    unittest.main()
