"""Exercise the real FP8-stream installer on CPU-only structural fixtures."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "radiance_arnq.py"


def load_installer():
    # The installer is ordinary Python. Extract it rather than importing the
    # module's Torch custom-op registrations or substituting its guard logic.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_linear_is_ours", "_consumers_ok", "install"}
    selected = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {
        "ENABLED": True, "GNQ": False, "_log": lambda message: None,
        "_stream_forward": lambda self, *args, **kwargs: "stream",
        "_SiluMulQuant": lambda: object(),
    }
    exec(compile(selected, str(SOURCE), "exec"), namespace)
    return namespace


def linear():
    return types.SimpleNamespace(
        radiance_wref=object(), bias=None, reduce_results=True,
        input_size_per_partition=128,
    )


def layer(kind="full_attention"):
    return types.SimpleNamespace(
        layer_type=kind, layer_scale=False, use_attn_reduce_scatter_for_moe=False,
        self_attn=types.SimpleNamespace(qkv_proj=linear(), o_proj=linear()),
        linear_attn=types.SimpleNamespace(_rad_merged=True, out_proj=linear()),
        mlp=types.SimpleNamespace(
            expert_gate=None, gate_up_proj=linear(), down_proj=linear(), act_fn=object(),
        ),
        forward=lambda *args, **kwargs: "stock",
    )


class FP8StreamInstallTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_installer()
        distributed = types.ModuleType("vllm.distributed")
        distributed.get_pp_group = lambda: types.SimpleNamespace(world_size=1)
        distributed.get_tensor_model_parallel_world_size = lambda: 2
        self.distributed = distributed
        modules = patch.dict(sys.modules, {
            "vllm": types.ModuleType("vllm"), "vllm.distributed": distributed,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def install(self, layers):
        core = types.SimpleNamespace(layers=layers, aux_hidden_state_layers=())
        model = types.SimpleNamespace(modules=lambda: iter([core]))
        self.namespace["install"](model)

    def test_unsupported_layers_keep_stock_forward_and_reductions(self):
        for kind in ("full_attention", "linear_attention"):
            for attribute, value in (
                ("layer_scale", True),
                ("use_attn_reduce_scatter_for_moe", True),
                ("expert_gate", object()),
            ):
                with self.subTest(kind=kind, guard=attribute):
                    layers = [layer(kind) for _ in range(3)]
                    unsupported = layers[1]
                    owner = unsupported.mlp if attribute == "expert_gate" else unsupported
                    setattr(owner, attribute, value)
                    original_forward = unsupported.forward
                    original_act = unsupported.mlp.act_fn
                    self.install(layers)
                    self.assertIs(unsupported.forward, original_forward)
                    self.assertEqual(unsupported.forward(), "stock")
                    self.assertIs(unsupported.mlp.act_fn, original_act)
                    self.assertTrue(unsupported.mlp.down_proj.reduce_results)
                    self.assertTrue(unsupported.self_attn.o_proj.reduce_results)
                    self.assertTrue(unsupported.linear_attn.out_proj.reduce_results)

    def test_previous_layer_does_not_defer_reduction_into_expert_layer(self):
        layers = [layer(), layer()]
        layers[1].mlp.expert_gate = object()
        self.install(layers)
        self.assertTrue(layers[0].mlp.down_proj.reduce_results)
        self.assertFalse(layers[1]._rad_fp8_in)

    def test_supported_pair_streams_and_last_layer_stays_reduced(self):
        layers = [layer(), layer("linear_attention")]
        self.install(layers)
        self.assertFalse(layers[0].mlp.down_proj.reduce_results)
        self.assertTrue(layers[1]._rad_fp8_in)
        self.assertTrue(layers[1].mlp.down_proj.reduce_results)
        for item in layers:
            self.assertIs(item.forward.__func__, self.namespace["_stream_forward"])
            self.assertTrue(item._rad_fp8_mid)

    def test_expert_next_layer_need_not_expose_dense_consumer(self):
        layers = [layer(), layer()]
        layers[1].mlp = types.SimpleNamespace(expert_gate=object())
        original_forward = layers[1].forward
        self.install(layers)
        self.assertTrue(layers[0].mlp.down_proj.reduce_results)
        self.assertIs(layers[1].forward, original_forward)

    def test_default_off_does_not_change_layers(self):
        self.namespace["ENABLED"] = False
        item = layer()
        original_forward = item.forward
        self.install([item])
        self.assertIs(item.forward, original_forward)
        self.assertTrue(item.mlp.down_proj.reduce_results)
        self.assertFalse(hasattr(item, "_rad_fp8_in"))

    def test_unsupported_parallel_topologies_do_not_change_layers(self):
        for tp, pp in ((1, 1), (2, 2)):
            with self.subTest(tp=tp, pp=pp):
                self.distributed.get_tensor_model_parallel_world_size = lambda: tp
                self.distributed.get_pp_group = lambda: types.SimpleNamespace(world_size=pp)
                item = layer()
                original_forward = item.forward
                self.install([item])
                self.assertIs(item.forward, original_forward)
                self.assertTrue(item.mlp.down_proj.reduce_results)


if __name__ == "__main__":
    unittest.main()
