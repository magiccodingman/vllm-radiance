"""Numerical CPU mask contracts against the declared Transformers mask API and Torch SDPA.

No weights, vLLM, HIP library, image processor or GPU are loaded. The fallback
reference is extracted from the installed Gemma4 source, not reimplemented.
"""
import ast
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
import transformers
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_bidirectional_mask
from vision_fixtures import dispatch, installed


def reference():
    path = Path(transformers.__file__).parent / "models/gemma4/modeling_gemma4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"repeat_kv", "eager_attention_forward"}]
    scope = dict(torch=torch, nn=nn)
    exec(compile(tree, str(path), "exec"), scope)
    return scope["eager_attention_forward"]


class VisionMaskNumericalTests(unittest.TestCase):
    def setUp(self):
        self.scope = dispatch(torch)
        self.eager = reference()
        self.module = types.SimpleNamespace(head_dim=72, num_key_value_groups=1, training=False)
        self.q = torch.zeros((2, 2, 3, 72))
        self.k = torch.zeros_like(self.q)
        self.v = torch.ones_like(self.q)
        self.v[0, :, 2] = 1000
        self.v[1, :, 1:] = -1000
        self.padding = torch.tensor([[True, True, False], [True, False, False]])

    def run_attention(self, mask):
        with installed(self.scope, eager=self.eager):
            return self.scope["radiance_vit_flash_attn"](
                self.module, self.q, self.k, self.v, mask, scaling=0.5)[0]

    def expected(self, mask):
        return nn.functional.scaled_dot_product_attention(
            self.q, self.k, self.v, attn_mask=mask, scale=0.5).transpose(1, 2).contiguous()

    def test_boolean_sdpa_mask_excludes_padding_numerically(self):
        mask = self.padding[:, None, None, :].expand(-1, 1, 3, -1)
        original = mask.clone()
        torch.testing.assert_close(self.run_attention(mask), self.expected(mask))
        self.assertTrue(torch.equal(mask, original))

    def test_two_dimensional_padding_masks_broadcast_per_image(self):
        expected = self.expected(self.padding[:, None, None, :])
        for dtype in (torch.bool, torch.int64, torch.float32):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(self.run_attention(self.padding.to(dtype)), expected)

    def test_additive_attention_bias_is_preserved(self):
        mask = torch.tensor([[[[0.0, -2.0, -10000.0]]], [[[0.0, -10000.0, -10000.0]]]])
        original = mask.clone()
        torch.testing.assert_close(self.run_attention(mask), self.expected(mask))
        self.assertTrue(torch.equal(mask, original))

    def test_unmasked_reference_is_unchanged(self):
        torch.testing.assert_close(self.run_attention(None), self.expected(None))

    def test_repeated_encoder_requests_keep_padding_mask(self):
        # The real encoder builds its mask BEFORE the attention forward switches
        # the shared config. Exercise both the first SDPA mask and later calls.
        with patch.dict(ALL_MASK_ATTENTION_FUNCTIONS._global_mapping), \
                installed(self.scope, eager=self.eager) as gemma, patch("sys.stderr", io.StringIO()):
            self.scope["install_gemma_vision"]()
            config = types.SimpleNamespace(_attn_implementation="sdpa")
            attention = gemma.Gemma4VisionAttention(config)
            for request in range(3):
                with self.subTest(request=request):
                    mask = create_bidirectional_mask(config, self.q[:, 0], self.padding)
                    self.assertIsNotNone(mask, "padded image lost its mask after backend switch")
                    attention.forward()
                    torch.testing.assert_close(self.run_attention(mask),
                                               self.expected(self.padding[:, None, None, :]))

    def test_registered_backend_keeps_dense_mask_elision(self):
        with patch.dict(ALL_MASK_ATTENTION_FUNCTIONS._global_mapping), \
                installed(self.scope, eager=self.eager), patch("sys.stderr", io.StringIO()):
            self.scope["install_gemma_vision"]()
            config = types.SimpleNamespace(_attn_implementation="radiance_vit_flash")
            self.assertIsNone(create_bidirectional_mask(config, self.q[:, 0], torch.ones_like(self.padding)))

    def test_small_real_vision_encoder_matches_sdpa_across_requests(self):
        from transformers.models.gemma4 import modeling_gemma4 as gemma
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

        # Random tiny encoder, no pretrained checkpoint or image understanding
        # claim. This exercises real projections, RoPE, mask creation and layers.
        torch.manual_seed(17)
        config = Gemma4VisionConfig(hidden_size=144, intermediate_size=288,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=72, _attn_implementation="sdpa")
        encoder = gemma.Gemma4VisionEncoder(config).eval()
        inputs = torch.randn(2, 3, 144)
        positions = torch.tensor([[[0, 0], [0, 1], [-1, -1]],
                                  [[0, 0], [-1, -1], [-1, -1]]])
        arguments = dict(inputs_embeds=inputs, attention_mask=self.padding, pixel_position_ids=positions)
        with torch.no_grad():
            expected = encoder(**arguments).last_hidden_state.clone()
        platform = types.ModuleType("vllm.platforms.rocm")
        platform.on_gfx12x = lambda: True
        with patch.dict(ALL_MASK_ATTENTION_FUNCTIONS._global_mapping), \
                patch.dict(gemma.ALL_ATTENTION_FUNCTIONS._global_mapping), \
                patch.dict("sys.modules", {platform.__name__: platform}), \
                patch.object(gemma, "_radiance_vit_installed", False, create=True), \
                patch.object(gemma.Gemma4VisionAttention, "forward", gemma.Gemma4VisionAttention.forward), \
                patch("sys.stderr", io.StringIO()), torch.no_grad():
            self.scope["install_gemma_vision"]()
            self.assertTrue(gemma._radiance_vit_installed)
            for request in range(3):
                with self.subTest(request=request):
                    actual = encoder(**arguments).last_hidden_state
                    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
