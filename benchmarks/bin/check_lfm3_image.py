#!/usr/bin/env python3
"""Installed LFM3 native registry/config contracts; no GPU computation."""
import argparse
import copy
import json
from pathlib import Path

from vllm.model_executor.models import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.lfm3 import Lfm3Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.model / "config.json").read_text())
    expected = copy.deepcopy(raw)
    config = Lfm3Config.from_dict(copy.deepcopy(raw))
    assert config.model_type == "lfm3"
    assert config.conv_L_cache == 7 and config.sliding_window == 256
    assert [config.layer_types.count(x) for x in ("conv", "full_attention", "sliding_attention")] == [13, 6, 5]
    assert config.rope_parameters["full_attention"]["rope_type"] == "proportional"
    assert config.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert config.use_attention_gate and config.use_sandwich_norm
    # Loading the config must use the vendored native class, with remote code
    # disabled; a registry-only string alias would fail here.
    loaded = get_config(str(args.model), trust_remote_code=False)
    assert isinstance(loaded, Lfm3Config)
    # Load the registered native class without constructing a device-aware
    # ModelConfig. Full ModelConfig/quantization execution is the live gate.
    registered = ModelRegistry.models["Lfm3ForCausalLM"]
    cls = registered.load_model_cls()
    assert cls.__name__ == "Lfm3ForCausalLM"
    assert cls.__module__ == "vllm.model_executor.models.lfm3"
    assert json.loads((args.model / "config.json").read_text()) == expected
    quant = getattr(config, "quantization_config", None)
    if quant is not None:
        assert quant["quant_method"] == "compressed-tensors", quant
        assert any("conv" in rule for rule in quant["ignore"])
        print("Checkpoint quantization: compressed-tensors FP8; convolution exclusions retained")
    print("Installed LFM3 registry/config and proportional RoPE selection: PASS")


if __name__ == "__main__":
    main()
