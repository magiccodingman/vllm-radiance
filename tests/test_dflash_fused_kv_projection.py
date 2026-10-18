"""CPU algebra and patch migration; native FP8/W4 kernels are not exercised."""
import ast
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_DFLASH_KV_SOURCE", ROOT / "patch_dflash_fused_kv_fp8.py"))
sys.path.insert(0, str(ROOT))


class QuantMethod:
    def __init__(self, effective_weight):
        self.weight = effective_weight
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        return F.linear(x, self.weight, bias)


class Projection:
    def __init__(self, storage="quant", bias=True, skip_bias_add=False):
        self.effective = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 16
        self.weight = (self.effective.clone() if storage == "dense" else
                       torch.empty(0, dtype=torch.bfloat16) if storage == "packed" else
                       torch.zeros((6, 4), dtype=torch.int8))
        self.bias = torch.full((6,), 1e8) if bias else None
        self.orig_dtype = torch.float32
        self.input_size_per_partition = 4
        self.quant_method = QuantMethod(self.effective)
        self.skip_bias_add = skip_bias_add

    def __call__(self, x):
        return (self.quant_method.apply(self, x, None if self.skip_bias_add else self.bias),
                self.bias if self.skip_bias_add else None)


def attention(storage="quant", **kwargs):
    return SimpleNamespace(qkv_proj=Projection(storage, **kwargs), q_size=2)


class DFlashFusedKVProjectionTests(unittest.TestCase):
    def setUp(self):
        self.overlay = runpy.run_path(str(SOURCE))
        self.scope = dict(torch=torch, F=F)
        helper = self.overlay["HELPER_NEW"].split("@support_torch_compile", 1)[0]
        exec(compile(helper, str(SOURCE), "exec"), self.scope)
        self.rows = self.scope["_dflash_kv_weight_rows"]
        exec("def build(self, layers_attn):\n" + self.overlay["SLICE_NEW"], self.scope)
        exec("def project(self, normed_context_states):\n" + self.overlay["LAZY_NEW"] +
             "\n        return all_kv_flat\n", self.scope)

    def test_dense_dtypes_preserve_exact_rows(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                proj = Projection("dense")
                proj.weight = proj.weight.to(dtype)
                torch.testing.assert_close(self.rows(proj, 2), proj.weight[2:], rtol=0, atol=0)
                self.assertEqual(proj.quant_method.calls, 0)

    def test_quantized_bias_is_not_baked_into_weights(self):
        proj = Projection()
        # A large bias also rules out subtracting it after rounded projection output.
        torch.testing.assert_close(self.rows(proj, 2), proj.effective[2:], rtol=0, atol=0)
        x = torch.tensor([[1., -2., 3., -4.], [2., 0., -1., 1.]])
        torch.testing.assert_close(F.linear(x, self.rows(proj, 2), proj.bias[2:]),
                                   F.linear(x, proj.effective, proj.bias)[:, 2:], rtol=0, atol=0)

    def test_unbiased_quantized_projection(self):
        proj = Projection(bias=False)
        torch.testing.assert_close(self.rows(proj, 2), proj.effective[2:], rtol=0, atol=0)

    def test_separate_bias_projection(self):
        proj = Projection(skip_bias_add=True)
        torch.testing.assert_close(self.rows(proj, 2), proj.effective[2:], rtol=0, atol=0)

    def test_packed_bf16_storage_uses_quant_method(self):
        proj = Projection("packed")
        torch.testing.assert_close(self.rows(proj, 2), proj.effective[2:], rtol=0, atol=0)
        self.assertEqual(proj.quant_method.calls, 1)

    def test_any_quantized_or_packed_layer_defers_build(self):
        for layout in (("dense", "quant"), ("quant", "dense"), ("dense", "packed")):
            with self.subTest(layout=layout):
                layers = [attention(storage) for storage in layout]
                state = SimpleNamespace()
                self.scope["build"](state, layers)
                self.assertIsNone(state._fused_kv_weight)
                self.assertTrue(all(a.qkv_proj.quant_method.calls == 0 for a in layers))

    def test_all_dense_layers_keep_eager_build(self):
        layers = [attention("dense"), attention("dense")]
        state = SimpleNamespace()
        self.scope["build"](state, layers)
        torch.testing.assert_close(state._fused_kv_weight,
                                   torch.cat([a.qkv_proj.effective[2:] for a in layers]))

    def test_lazy_mixed_projection_matches_independent_layers_and_is_reused(self):
        layers = [attention("dense", bias=False), attention("quant", bias=False),
                  attention("packed", bias=False)]
        state = SimpleNamespace(_fused_kv_bias=None)
        self.scope["build"](state, layers)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8
        expected = torch.cat([F.linear(x, a.qkv_proj.effective)[:, 2:] for a in layers], dim=1)
        for _ in range(2):
            torch.testing.assert_close(self.scope["project"](state, x), expected, rtol=0, atol=0)
        self.assertEqual([a.qkv_proj.quant_method.calls for a in layers], [0, 1, 1])

    def bake(self, installed):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3_dflash.py"
            path.write_text(installed, encoding="utf-8")
            self.overlay["main"].__globals__["F"] = path
            with contextlib.redirect_stdout(io.StringIO()):
                self.overlay["main"]()
                first = path.read_text(encoding="utf-8")
                self.overlay["main"]()
            self.assertEqual(first, path.read_text(encoding="utf-8"))
        ast.parse(first)
        scope = dict(torch=torch, nn=torch.nn, F=F, support_torch_compile=lambda cls: cls)
        exec(first, scope)
        for storage in ("quant", "packed"):
            proj = Projection(storage)
            torch.testing.assert_close(scope["_dflash_kv_weight_rows"](proj, 2),
                                       proj.effective[2:], rtol=0, atol=0)
        return first

    def test_fresh_patch_is_correct_and_idempotent(self):
        installed = self.overlay["HELPER_OLD"] + "\n    pass\n\nclass Buffers:\n"
        installed += "    def build(self, layers_attn):\n" + self.overlay["SLICE_OLD"]
        installed += "\n    def project(self, normed_context_states):\n" + self.overlay["LAZY_OLD"] + "\n"
        self.bake(installed)

    def test_existing_overlay_is_repaired_and_idempotent(self):
        helper = self.overlay["HELPER_NEW"].replace(
            "weight.dtype in _DFLASH_DENSE and weight.ndim == 2", "weight.dtype in _DFLASH_DENSE"
        ).replace("qkv_proj.quant_method.apply(qkv_proj, eye, bias=None)", "qkv_proj(eye)")
        old_slice = """        self._kv_source_attn = layers_attn
        if layers_attn[0].qkv_proj.weight.dtype not in _DFLASH_DENSE:
            self._fused_kv_weight = None
        else:
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)"""
        installed = helper + "\n    pass\n\nclass Buffers:\n"
        installed += "    def build(self, layers_attn):\n" + old_slice
        installed += "\n    def project(self, normed_context_states):\n" + self.overlay["LAZY_NEW"] + "\n"
        result = self.bake(installed)
        self.assertIn("or a.qkv_proj.weight.ndim != 2 for a in layers_attn)", result)


if __name__ == "__main__":
    unittest.main()
