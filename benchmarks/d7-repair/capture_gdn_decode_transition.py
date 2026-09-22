"""Capture a synthetic first decode transition with the qualified conv repair.

Opt-in diagnostic hooks only. Captures selected state slots, never entire pools
or weights. Install before the conformance worker attaches its observers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

from qwen_r9700_lab.conformance_instrumentation import CallRecorder, HookSet
from qwen_r9700_lab.conformance_radiance import RadianceProbe
from qwen_r9700_lab.diagnostic_contract import seal, write_private

CONV_SHA256 = "da2d1183c29f68497d0166960c3ca7bedd143b90eead1de3d98e90af0c0f8a4a"
LAYER = "target.language_model.model.layers.0.linear_attn"
SELECTED = {
    LAYER,
    *(LAYER + suffix for suffix in (".in_proj_ba", ".in_proj_qkvz", ".norm", ".out_proj")),
}
SITES = {"radiance_gdn.conv_update", "radiance_gdn.recurrent_update", "stock.gdn.packed_decode"}


def selected_pool(pool, indices):
    import torch

    ids = torch.unique(indices.detach().reshape(-1).to(dtype=torch.long), sorted=True)
    if not bool(((ids >= 0) & (ids < pool.shape[0])).all()) or ids.numel() > 8:
        raise ValueError("selected state capture outside admitted slots")
    return {
        "source_shape": list(pool.shape),
        "indices": ids,
        "selected_values": pool.index_select(0, ids),
    }


def validate_selection(layer, consumed):
    if type(layer) is not int or not 0 <= layer < 64:
        raise ValueError("selected decoder layer must be an integer from 0 through 63")
    if type(consumed) is not int or consumed < 1:
        raise ValueError("selected consumed position must be a positive integer")


def selected(recorder, row, logical, *, layer=0, consumed=2050, residuals=False):
    validate_selection(layer, consumed)
    prefix = f"target.language_model.model.layers.{layer}.linear_attn"
    sites = {
        prefix,
        *(prefix + suffix for suffix in (".in_proj_ba", ".in_proj_qkvz", ".norm", ".out_proj")),
    }
    if logical.get("phase") != "step" or logical.get("consumed") != consumed:
        return False
    attention = f"target.language_model.model.layers.{layer}.self_attn"
    if row["site"] == attention or row["site"].startswith(attention + "."):
        # Module arguments are activations, positions and outputs. Do not select
        # unrelated layers, weights, or backend calls with whole cache pools.
        return True
    if residuals and (
        re.fullmatch(
            r"target\.language_model\.model\.layers\.\d+(\.(input_layernorm|post_attention_layernorm))?",
            row["site"],
        )
        or row["site"]
        in {"target.language_model.model.norm", "target.language_model.logits_processor"}
    ):
        return True
    if row["site"] not in sites | SITES:
        return False
    while row is not None:
        if row["site"] == prefix:
            return True
        row = recorder.rows.get(row["parent"])
    return False


def install_transition_hook(conv_path: Path, *, layer=0, consumed=2050, residuals=False):
    validate_selection(layer, consumed)
    if hashlib.sha256(conv_path.read_bytes()).hexdigest() != CONV_SHA256:
        raise ValueError("stock convolution candidate hash mismatch")
    capture, attach, detach = CallRecorder._capture, RadianceProbe.attach, RadianceProbe.detach

    def targeted_capture(self, root, values, logical):
        index = int(root.parent.name.removeprefix("call-"))
        row = self.rows[index]
        if not selected(self, row, logical, layer=layer, consumed=consumed, residuals=residuals):
            return capture(self, root, values, logical)
        transformed = dict(values)
        args, kwargs = list(values.get("args", ())), dict(values.get("kwargs", {}))
        if row["site"] == "radiance_gdn.conv_update":
            if len(args) != 13:
                raise ValueError("unreviewed convolution call ABI")
            args[3] = selected_pool(args[3], args[5])
        elif row["site"] == "radiance_gdn.recurrent_update":
            if len(args) < 16:
                raise ValueError("unreviewed recurrent call ABI")
            args[7] = selected_pool(args[7], args[10])
        elif row["site"] == "stock.gdn.packed_decode":
            if args or "initial_state" not in kwargs or "ssm_state_indices" not in kwargs:
                raise ValueError("unreviewed stock GDN call ABI")
            kwargs["initial_state"] = selected_pool(
                kwargs["initial_state"], kwargs["ssm_state_indices"]
            )
            if root.name == "after" and isinstance(values.get("result"), tuple):
                result = values["result"]
                if len(result) != 2:
                    raise ValueError("unreviewed stock GDN result ABI")
                transformed["result"] = (
                    result[0],
                    selected_pool(result[1], kwargs["ssm_state_indices"]),
                )
        transformed.update(args=tuple(args), kwargs=kwargs)
        with self.lock:
            original = self.mode
            try:
                self.mode = "tensor"
                return capture(self, root, transformed, logical)
            finally:
                self.mode = original

    def targeted_attach(self):
        module = sys.modules["vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"]
        spec = importlib.util.spec_from_file_location("qwen_causal_conv1d_candidate", conv_path)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        spec.loader.exec_module(candidate)
        repairs = HookSet()
        repairs.replace(module, "causal_conv1d_update", candidate.causal_conv1d_update)
        self._decode_transition_repairs = repairs
        attach(self)

        def context():
            if self.positions is None or self.index >= len(self.campaign.expected):
                return None
            expected = self.campaign.expected[self.index]
            return {
                "consumed": expected["consumed"],
                "input_digest": expected["input_digest"],
                "positions": self.positions,
                "phase": expected["phase"],
            }

        for name, site in {
            "causal_conv1d_update": "stock.conv.update",
            "fused_recurrent_gated_delta_rule_packed_decode": "stock.gdn.packed_decode",
            "fused_sigmoid_gating_delta_rule_update": "stock.gdn.sigmoid_update",
        }.items():
            self.calls.bind(module, name, site=site, hooks=self.hooks, context=context)
        write_private(
            self.campaign.root / "decode-transition-capture.json",
            seal(
                {
                    "scope": (
                        f"Layer {layer} at consumed={consumed}; convolution correction. "
                        "The caller separately binds any other repair candidates."
                    ),
                    "selected_layer": layer,
                    "selected_consumed": consumed,
                    "residuals": residuals,
                    "convolution_candidate": CONV_SHA256,
                    "hook": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "selected_tensor_capture": True,
                    "installed_sources_modified": False,
                }
            ),
        )

    def targeted_detach(self):
        try:
            return detach(self)
        finally:
            repairs = getattr(self, "_decode_transition_repairs", None)
            if repairs is not None:
                repairs.close()

    CallRecorder._capture = targeted_capture
    RadianceProbe.attach, RadianceProbe.detach = targeted_attach, targeted_detach
