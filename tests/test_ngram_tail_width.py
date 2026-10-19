"""Execute the gather kernel's masked lane algebra on NumPy, without Triton/GPU."""
import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('RADIANCE_NGRAM_SOURCE', ROOT / 'radiance_draft_gpu.py'))


class Pointer:
    def __init__(self, array, offset=0):
        self.array, self.offset = array, offset
    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset)


class Values(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


class Lanes:
    constexpr = object()
    int32 = np.int32
    def program_id(self, axis):
        return self.row
    def arange(self, start, end):
        if start != 0 or end <= 0 or end & (end - 1):
            raise ValueError('Triton arange end must be a power of two')
        return np.arange(end).view(Values)
    def load(self, pointer, mask=True, other=0):
        indices, active = np.broadcast_arrays(pointer.offset, mask)
        result = np.full(indices.shape, other, dtype=pointer.array.dtype)
        result[active] = pointer.array[indices[active]]
        return result.view(Values)
    def store(self, pointer, value, mask=True):
        indices, values, active = np.broadcast_arrays(pointer.offset, value, mask)
        # A padded store without its width mask would overwrite another row or
        # the sentinel. NumPy also raises on an out-of-range final-row access.
        pointer.array[indices[active]] = values[active]
    def sum(self, value, axis):
        return np.sum(value, axis=axis)
    where = staticmethod(np.where)


class NgramTailWidthTests(unittest.TestCase):
    def setUp(self):
        self.tl = Lanes()
        node = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_match_gather')
        node.decorator_list = []
        self.scope = dict(tl=self.tl)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), self.scope)

    def gather(self, contexts, starts, width):
        maxlen = max(map(len, contexts))
        rows = np.full((len(contexts), maxlen), -999, dtype=np.int32)
        for index, context in enumerate(contexts):
            rows[index, :len(context)] = context
        lengths = np.array(list(map(len, contexts)), dtype=np.int32)
        keys = np.array([0 if start is None else (3 << 20) | (start - 1) for start in starts], dtype=np.int32)
        result = np.full(len(contexts) * width + 8, -777, dtype=np.int32)
        counts = np.full(len(contexts), -777, dtype=np.int32)
        fn = self.scope['_match_gather']
        args = [Pointer(rows.ravel()), Pointer(lengths), maxlen, Pointer(keys),
                Pointer(result), Pointer(counts), 20, width]
        if 'BLOCK' in inspect.signature(fn).parameters:
            args.append(1 << (width - 1).bit_length())
        for self.tl.row in range(len(contexts)):
            fn(*args)
        np.testing.assert_array_equal(result[-8:], np.full(8, -777))
        return result[:-8].reshape(len(contexts), width), counts

    def check_reference(self, contexts, starts, width):
        actual, counts = self.gather(contexts, starts, width)
        expected = [([] if start is None else context[start:start + width])
                    for context, start in zip(contexts, starts, strict=True)]
        for index, tokens in enumerate(expected):
            np.testing.assert_array_equal(actual[index], tokens + [-1] * (width - len(tokens)))
        np.testing.assert_array_equal(counts, list(map(len, expected)))

    def test_width_five(self):
        self.check_reference([list(range(12))], [3], 5)

    def test_width_seven(self):
        self.check_reference([list(range(12))], [3], 7)

    def test_width_three_with_adjacent_rows(self):
        self.check_reference([list(range(12)), list(range(20, 32))], [4, 7], 3)

    def test_unmatched_rows_are_filled_with_negative_one(self):
        self.check_reference([list(range(8)), list(range(8))], [None, 4], 5)

    def test_short_tail_is_masked(self):
        self.check_reference([list(range(8))], [6], 7)

    def test_exact_tail_length(self):
        self.check_reference([list(range(12))], [5], 7)

    def test_existing_power_of_two_widths(self):
        for width in (1, 2, 4, 8):
            self.check_reference([list(range(16))], [3], width)

    def test_wrapper_passes_rounded_tile(self):
        tree = ast.parse(SOURCE.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'match_gpu')
        launches = []
        class Launch:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: launches.append((args, kwargs))
        class Buffer:
            def view(self, *shape):
                return shape
        scope = dict(torch=SimpleNamespace(int32='i32', zeros=lambda *a, **k: Buffer(),
                                          empty=lambda *a, **k: Buffer()),
                     triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                            next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
                     _match_scan=Launch(), _match_gather=Launch(), _MAXL=24, _MIN=3, _SH=20)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
        scope['match_gpu'](SimpleNamespace(shape=(2, 12), device='cpu'), Buffer(), 7, 12)
        args, kwargs = launches[-1]
        block = kwargs.get('BLOCK', args[8] if len(args) > 8 else None)
        self.assertEqual(block, 8)
        self.assertEqual(args[7], 7)


if __name__ == '__main__':
    unittest.main()
