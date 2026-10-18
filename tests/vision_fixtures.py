"""Load the real Vision dispatch/installer without importing the HIP library."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch


SOURCE = Path(os.environ.get("RADIANCE_VISION_TEST_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_vit_attn.py")))


class Registry(dict):
    def register(self, name, function):
        self[name] = function


def dispatch(torch=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"install_gemma_vision", "radiance_vit_flash_attn", "_supports_r4d"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = dict(sys=sys, _HEAD_DIM=72, _HAS_R4D=True, torch=torch, _r4d_bhsd=Mock())
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope


@contextmanager
def installed(scope, mask_registry=None, eager_mask=None, eager=None, gfx=True):
    class Attention:
        def __init__(self, config, head_dim=72):
            self.config, self.head_dim = config, head_dim

        def forward(self, *args, **kwargs):
            return args, kwargs

    gemma = types.ModuleType("transformers.models.gemma4.modeling_gemma4")
    gemma.Gemma4VisionAttention = Attention
    gemma.ALL_ATTENTION_FUNCTIONS = Registry()
    gemma.eager_attention_forward = eager or Mock()
    package = types.ModuleType("transformers.models.gemma4")
    package.modeling_gemma4 = gemma
    platform = types.ModuleType("vllm.platforms.rocm")
    platform.on_gfx12x = lambda: gfx
    modules = {gemma.__name__: gemma, package.__name__: package, platform.__name__: platform}
    for name in ("transformers", "transformers.models", "vllm", "vllm.platforms"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
    if mask_registry is not None:
        masks = types.ModuleType("transformers.masking_utils")
        masks.ALL_MASK_ATTENTION_FUNCTIONS = mask_registry
        masks.eager_mask = eager_mask
        modules[masks.__name__] = masks
    with patch.dict(sys.modules, modules):
        yield gemma
