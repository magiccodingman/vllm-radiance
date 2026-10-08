"""Native Q/K-normalization + rotary cuts, with explicit private KV restoration.

The original compiler fused normalization into rotation. That composite is
replaced at its attention consumer: the common projection is normalized and
rotated by the selected native variant, written using the unchanged KV writer,
then consumed by the unchanged qualified attention implementation.
"""

import copy
import functools
import importlib.util
import os
import sys
from pathlib import Path

import torch

from qwen_r9700_lab.conformance_rotary_repair import patch_source
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def layer_name(value):
    if isinstance(value, str):
        return value
    from vllm.model_executor.layers.attention.attention import _resolve_layer_name

    return _resolve_layer_name(value)


class AttentionStages:
    def __init__(self, history):
        import vllm.model_executor.layers.rotary_embedding.mrope as native

        self.history = history
        self.native = native
        # An isolated module leaves the installed native/eager implementation intact.
        destination = (
            Path(os.environ["QWEN_OPTIMIZED_STARTUP_RECEIPT"]).parent / "tape-rne-mrope.py"
        )
        destination.write_text(patch_source(Path(native.__file__).read_text()))
        name = "vllm.model_executor.layers.rotary_embedding._tape_rne_mrope"
        spec = importlib.util.spec_from_file_location(name, destination)
        self.rne = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.rne
        spec.loader.exec_module(self.rne)
        self.cuts = {}

    def prepare(self, tape, parameters):
        self.cuts = {}
        latest = {}
        for call in tape.calls:
            if call.cut is None:
                continue
            args, kwargs = tape.thaw(call.cut[0])
            if call.name == "radiance.mxfp4_linear.default":
                owners = parameters.get(args[1].untyped_storage().data_ptr(), [])
                if len(owners) == 1 and owners[0].endswith("self_attn.qkv_proj.weight"):
                    after = tape.thaw(call.cut[1])
                    latest = {"raw": after[2], "rotations": [], "norms": []}
            elif call.name.startswith("qwen_d7_qualified.") and ".self_attn." in str(args[-1]):
                latest["norms"].append((call.function, args, kwargs))
            elif call.name.startswith("inductor/") and latest:
                if (
                    len(args) >= 5
                    and isinstance(args[1], torch.Tensor)
                    and isinstance(args[2], torch.Tensor)
                    and args[1].shape == args[2].shape == (8, 32)
                ):
                    latest["rotations"].append((call.function, args, kwargs))
            elif call.name == "vllm.unified_kv_cache_update.default":
                latest["writer"] = call.function
            elif call.name == "vllm.unified_attention_with_output.default":
                name = layer_name(args[4])
                if len(latest.get("rotations", [])) != 2 or len(latest.get("norms", [])) != 2:
                    raise DiagnosticError("native Q/K rotary cut is incomplete")
                if latest["raw"].shape != (8, 14336):
                    raise DiagnosticError("native rotary projection shape changed")
                latest["norms"].sort(key=lambda item: 0 if ".q_norm" in item[1][-1] else 1)
                latest["rotations"].sort(key=lambda item: -item[1][0].shape[1])
                if [item[1][0].shape[1] for item in latest["rotations"]] != [24, 4]:
                    raise DiagnosticError("native rotation roles are ambiguous")
                self.cuts[name] = latest
                latest = {}

    def original(self, arm, name, cut):
        events = self.history.catalogs[arm]["events"]
        key = name.rsplit(".attn", 1)[0] + ".q_norm.weight"
        # The attention layer registration may omit the final '.attn'.
        if key not in self.history.norms[arm]:
            key = name + ".q_norm.weight"
        index = self.history.norms[arm][key]
        rotate, reduce = events[index], events[index - 1]
        group = 1 if arm == "m1" else 8
        cos, sin = cut["rotations"][0][1][1:3]
        q, k = [], []
        qw = cut["norms"][0][1][1]
        kw = cut["norms"][1][1][1]
        for start in range(0, 8, group):
            raw = cut["raw"][start : start + group]
            qr = torch.empty((group, 24, 1), dtype=torch.float32, device=raw.device)
            kr = torch.empty((group, 4, 1), dtype=torch.float32, device=raw.device)
            qo = torch.empty((group, 24, 256), dtype=raw.dtype, device=raw.device)
            ko = torch.empty((group, 4, 256), dtype=raw.dtype, device=raw.device)
            self.history.launch(reduce, raw, qr, kr, group * 24, group * 4)
            self.history.launch(
                rotate,
                raw,
                kr,
                kw,
                cos[start : start + group],
                sin[start : start + group],
                qr,
                qw,
                ko[:, :, :64],
                ko[:, :, 64:],
                qo[:, :, :64],
                qo[:, :, 64:],
                group * 256,
                group * 768,
                group * 1536,
                group * 4608,
            )
            q.append(qo)
            k.append(ko)
        return torch.cat(q), torch.cat(k)

    def normalized(self, cut, group):
        result = []
        for fn, args, kwargs in cut["norms"]:
            outputs = [
                fn(args[0][start : start + group], *args[1:], **kwargs)
                for start in range(0, 8, group)
            ]
            result.append(torch.cat(outputs))
        return result

    def fixed(self, variant, name, cut):
        group = 1 if variant == "fix1_m1" else 8
        normalized = self.normalized(cut, group)
        cos, sin = cut["rotations"][0][1][1:3]
        if variant.endswith("eager_m8"):
            module = self.rne if variant.startswith("final") else self.native
            q, k = module.triton_mrope(
                normalized[0].reshape(8, -1),
                normalized[1].reshape(8, -1),
                torch.stack([cos] * 3),
                torch.stack([sin] * 3),
                [11, 11, 10],
                256,
                64,
                True,
                True,
            )
            return q.reshape(8, 24, 256), k.reshape(8, 4, 256)
        result = []
        archived = None
        if variant.startswith("fix1"):
            events = self.history.catalogs[variant]["events"]
            # Actual generated rotations directly preceding this layer's KV writer.
            writers = [
                i
                for i, e in enumerate(events)
                if e["operation"] == "vllm.unified_kv_cache_update.default"
            ]
            layer = int(name.split(".layers.")[1].split(".")[0])
            writer = writers[layer // 4]
            archived = events[writer - 2 : writer]
            if any("compiled" not in e for e in archived):
                raise DiagnosticError("Fix 1 rotary dispatch catalog changed")
            archived = sorted(archived, key=lambda event: -event["args"][0]["shape"][1])
        for role, (fn, args, kwargs) in enumerate(cut["rotations"]):
            full = torch.empty(
                (8, 24 if role == 0 else 4, 256), dtype=normalized[role].dtype, device=cos.device
            )
            for start in range(0, 8, group):
                out = full[start : start + group]
                if archived is None:
                    values = self.rotation_arguments(
                        args,
                        normalized[role][start : start + group],
                        cos[start : start + group],
                        sin[start : start + group],
                        out,
                    )
                    fn(*values, **kwargs)
                else:
                    event = archived[role]
                    self.history.launch(
                        event,
                        *self.rotation_arguments(
                            event["args"],
                            normalized[role][start : start + group],
                            cos[start : start + group],
                            sin[start : start + group],
                            out,
                        ),
                    )
            result.append(full)
        return tuple(result)

    @staticmethod
    def rotation_arguments(template, normalized, cosine, sine, out):
        values = [normalized, cosine, sine]
        for value in template[3:]:
            shape = (
                value.get("shape")
                if isinstance(value, dict)
                else (list(value.shape) if isinstance(value, torch.Tensor) else None)
            )
            if shape is None:
                values.append(value)
            elif shape[-1] == 256:
                values.append(out)
            elif shape[-1] == 64:
                values.append(out[:, :, :64])
            elif shape[-1] == 192:
                values.append(out[:, :, 64:])
            else:
                raise DiagnosticError("unknown rotary output layout")
        return values

    def composite(self, function, variant, cut, *args, **kwargs):
        name = layer_name(args[4])
        if variant.startswith("old"):
            q, k = self.original("m1" if variant.endswith("m1") else "m8", name, cut)
        else:
            q, k = self.fixed(variant, name, cut)
        # Mutate private call arguments so the cut checker also observes Q/K,
        # not just coincidentally equal attention output after an altered cache.
        args[0].copy_(q)
        args[1].copy_(k)
        cut["writer"](args[1], args[2], args[4])
        return function(*args, **kwargs)

    def kv_serial(self, function, *args, **kwargs):
        from vllm.forward_context import get_forward_context, override_forward_context

        context = get_forward_context()
        name = layer_name(args[2])
        for row in range(8):
            one = copy.copy(context)
            one.slot_mapping = dict(context.slot_mapping)
            one.slot_mapping[name] = context.slot_mapping[name][row : row + 1]
            with override_forward_context(one):
                result = function(
                    args[0][row : row + 1], args[1][row : row + 1], *args[2:], **kwargs
                )
        return result

    def cases(self, call):
        from native_d7_stage_matrix import shared_versions

        args, _ = call.cut[0][0].thaw(call.cut[0])
        if call.name == "vllm.unified_kv_cache_update.default":
            yield (
                "Attention KV write",
                layer_name(args[2]),
                shared_versions(call.function, functools.partial(self.kv_serial, call.function)),
            )
        elif call.name == "vllm.unified_attention_with_output.default":
            name = layer_name(args[4])
            cut = self.cuts[name]
            versions = {
                variant: functools.partial(self.composite, call.function, variant, cut)
                for variant in (
                    "old_m1",
                    "old_m8",
                    "fix1_m1",
                    "fix1_m8",
                    "fix1_eager_m8",
                    "final_m8",
                    "final_eager_m8",
                )
            }
            yield "Attention Q/K normalization, RoPE and layout", name, versions
