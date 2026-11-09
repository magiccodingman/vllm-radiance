"""CPU-only registration contracts, independent of Torch and Transformers imports."""
import io
import types
import unittest
from unittest.mock import Mock, patch
from vision_fixtures import Registry, dispatch, installed


class VisionMaskRegistrationTests(unittest.TestCase):
    def test_mask_backend_is_registered_before_config_switch(self):
        scope, masks, eager = dispatch(), Registry(), object()
        with installed(scope, masks, eager) as gemma, patch("sys.stderr", io.StringIO()):
            scope["install_gemma_vision"]()
            self.assertIs(masks.get("radiance_vit_flash"), eager)
            self.assertIs(gemma.ALL_ATTENTION_FUNCTIONS["radiance_vit_flash"], scope["radiance_vit_flash_attn"])
            config = types.SimpleNamespace(_attn_implementation="sdpa")
            args, kwargs = gemma.Gemma4VisionAttention(config).forward("input", marker=1)
            self.assertEqual(config._attn_implementation, "radiance_vit_flash")
            self.assertEqual((args, kwargs), (("input",), {"marker": 1}))

    def test_install_is_idempotent(self):
        scope, masks = dispatch(), Registry()
        with installed(scope, masks, object()) as gemma, patch("sys.stderr", io.StringIO()):
            scope["install_gemma_vision"]()
            forward = gemma.Gemma4VisionAttention.forward
            scope["install_gemma_vision"]()
            self.assertIs(gemma.Gemma4VisionAttention.forward, forward)
            self.assertIn("radiance_vit_flash", masks)

    def test_other_head_size_keeps_original_backend(self):
        scope = dispatch()
        with installed(scope, Registry(), object()) as gemma, patch("sys.stderr", io.StringIO()):
            scope["install_gemma_vision"]()
            config = types.SimpleNamespace(_attn_implementation="sdpa")
            gemma.Gemma4VisionAttention(config, head_dim=64).forward()
            self.assertEqual(config._attn_implementation, "sdpa")

    def test_no_library_leaves_both_registries_and_forward_alone(self):
        scope, masks = dispatch(), Registry()
        scope["_HAS_R4D"] = False
        with installed(scope, masks, object()) as gemma:
            original = gemma.Gemma4VisionAttention.forward
            scope["install_gemma_vision"]()
            self.assertIs(gemma.Gemma4VisionAttention.forward, original)
            self.assertEqual(masks, {})
            self.assertEqual(gemma.ALL_ATTENTION_FUNCTIONS, {})

    def test_other_gpu_arch_leaves_registries_alone(self):
        scope, masks = dispatch(), Registry()
        with installed(scope, masks, object(), gfx=False) as gemma:
            original = gemma.Gemma4VisionAttention.forward
            scope["install_gemma_vision"]()
            self.assertIs(gemma.Gemma4VisionAttention.forward, original)
            self.assertEqual(masks, {})

    def test_mask_registration_failure_cannot_activate_wrapper(self):
        scope, masks = dispatch(), Registry()
        masks.register = Mock(side_effect=RuntimeError("synthetic registration failure"))
        with installed(scope, masks, object()) as gemma:
            original = gemma.Gemma4VisionAttention.forward
            with self.assertRaisesRegex(RuntimeError, "synthetic registration failure"):
                scope["install_gemma_vision"]()
            self.assertIs(gemma.Gemma4VisionAttention.forward, original)
            self.assertFalse(getattr(gemma, "_radiance_vit_installed", False))


if __name__ == "__main__":
    unittest.main()
