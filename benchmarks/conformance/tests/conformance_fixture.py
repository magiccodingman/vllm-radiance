"""Public, tiny hybrid checkpoint; no production weights or chat fixtures."""

import hashlib
import json

import numpy as np
from safetensors.numpy import save_file

from qwen_r9700_lab.conformance_reference import bf16, reference_contract
from qwen_r9700_lab.conformance_replay import PLAN_SCHEMA, reference_code_identity
from qwen_r9700_lab.diagnostic_contract import digest, seal


def tiny_checkpoint(root):
    root.mkdir()
    c = {
        "model_type": "qwen3_5_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 32,
        "max_position_embeddings": 256,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attn_output_gate": True,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.5,
        },
    }
    rng = np.random.default_rng(90214)
    prefix, weights = "model.language_model.", {}

    def tensor(name, shape, scale=0.1):
        weights[name] = bf16(rng.normal(0, scale, shape).astype(np.float32))

    def quantized(name, n, k=32):
        weights[name + ".weight"] = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        weights[name + ".weight_scale"] = np.full((n, k // 32), 122, np.uint8)

    tensor(prefix + "embed_tokens.weight", (32, 32))
    tensor("lm_head.weight", (32, 32))
    tensor(prefix + "norm.weight", (32,))
    for layer, kind in enumerate(c["layer_types"]):
        base = prefix + f"layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensor(base + name + ".weight", (32,))
        for name, n, k in (("gate_proj", 64, 32), ("up_proj", 64, 32), ("down_proj", 32, 64)):
            quantized(base + "mlp." + name, n, k)
        if kind == "linear_attention":
            base += "linear_attn."
            for name, n in (
                ("in_proj_qkv", 64),
                ("in_proj_z", 32),
                ("in_proj_a", 4),
                ("in_proj_b", 4),
                ("out_proj", 32),
            ):
                quantized(base + name, n)
            tensor(base + "conv1d.weight", (64, 1, 4))
            tensor(base + "norm.weight", (8,), 1)
            tensor(base + "A_log", (4,))
            tensor(base + "dt_bias", (4,))
        else:
            base += "self_attn."
            for name, n in (("q_proj", 64), ("k_proj", 16), ("v_proj", 16), ("o_proj", 32)):
                quantized(base + name, n)
            for name in ("q_norm", "k_norm"):
                tensor(base + name + ".weight", (8,))
    save_file(weights, root / "model-00001.safetensors")
    (root / "config.json").write_text(json.dumps({"text_config": c}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model-00001.safetensors")})
    )
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


def tiny_plan(root):
    files = tiny_checkpoint(root)
    return seal(
        {
            "schema": PLAN_SCHEMA,
            "contract": digest({"fixture": files, "reference": reference_contract()}),
            "execution": digest("CPU fixture"),
            "adapter": digest(reference_code_identity()),
            "checkpoint": str(root),
            "checkpoint_files": files,
            "kv_scales": {"3": [1.0, 1.0]},
            "prefix": [1, 4, 8],
            "forced_tokens": [7, 9, 13, 3],
            "reference_arithmetic": reference_contract(),
        }
    )
