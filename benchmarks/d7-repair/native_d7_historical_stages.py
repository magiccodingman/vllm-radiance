"""Invoke the actual archived compiled normalization and pointwise kernels.

Catalogs come from observed native calls, not inferred function names. This
module never decodes fixture tokens. A normalization cut includes any retained
residual terms that the historical compiler fused into that operation.
"""

import hashlib
import importlib.util
from pathlib import Path

import torch

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, private_json


class HistoricalStages:
    def __init__(self, model, catalogs):
        self.model = model
        self.parameters = dict([*model.named_parameters(), *model.named_buffers()])
        self.catalogs = {}
        self.modules = {}
        self.norms = {}
        self.pointwise = {}
        for arm, path in catalogs.items():
            document = private_json(Path(path))
            authenticate(document)
            if document["rows"] != (1 if arm.endswith("m1") else 8):
                raise DiagnosticError("historical compiled catalog width changed")
            self.catalogs[arm] = document
            self.norms[arm] = {}
            self.pointwise[arm] = {}
            for index, event in enumerate(document["events"]):
                if "compiled" not in event:
                    continue
                weights = [
                    w
                    for arg in event["args"]
                    if isinstance(arg, dict)
                    for w in arg.get("parameters", [])
                    if w.endswith("norm.weight")
                ]
                for name in weights:
                    self.norms[arm][name] = index
                if "silu_slice" in event["operation"]:
                    self.pointwise[arm].setdefault("silu", []).append(event)
                if "sigmoid_view" in event["operation"]:
                    self.pointwise[arm].setdefault("sigmoid", []).append(event)

    def kernel(self, event):
        desc = event["compiled"]
        path = Path(desc["source"])
        digest = desc["source_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise DiagnosticError("archived compiled source changed")
        if digest not in self.modules:
            spec = importlib.util.spec_from_file_location("qwen_archived_" + digest, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[digest] = module
        return getattr(self.modules[digest], desc["name"])

    def launch(self, event, *args):
        if len(args) != len(event["args"]):
            raise DiagnosticError(
                f"archived ABI argument count changed at event {event['index']}: "
                f"{len(args)} != {len(event['args'])}"
            )
        return self.kernel(event).run(*args, stream=torch.cuda.current_stream().cuda_stream)

    def norm(self, arm, key, args, prior=None, embedding_ids=None):
        events = self.catalogs[arm]["events"]
        event = events[self.norms[arm][key + ".weight"]]
        group = 1 if arm.endswith("m1") else 8
        result = []
        residuals = []
        x = args[0]
        weight = args[-3]  # gemma: x,w,eps,key; residual: x,r,w,eps,key
        if ".linear_attn.norm" in key:
            x, z, w, _eps, _ = args
            mean_event = events[event["index"] - 1]
            if "mean" not in mean_event["operation"] and "fused_" not in mean_event["operation"]:
                raise DiagnosticError("historical GDN norm reduction is missing")
            for row in range(0, 8, group):
                a = x.reshape(8, 48, 128)[row : row + group]
                gate = z.reshape(8, 48, 128)[row : row + group]
                mean = torch.empty((group * 48, 1), dtype=torch.float32, device=x.device)
                out = torch.empty((group, 6144), dtype=x.dtype, device=x.device)
                self.launch(mean_event, a, mean, group * 48, 128)
                self.launch(event, a, mean, w, gate, out, group, group * 6144)
                result.append(out.reshape(group * 48, 128))
            return torch.cat(result)
        for row in range(0, 8, group):
            a = x[row : row + group]
            norm = torch.empty_like(a)
            carry = torch.empty_like(a)
            if key.endswith("layers.0.input_layernorm"):
                if embedding_ids is None:
                    raise DiagnosticError("embedding cut lacks token indices")
                names = [
                    p
                    for arg in event["args"]
                    if isinstance(arg, dict)
                    for p in arg.get("parameters", [])
                    if "embed_tokens" in p
                ]
                if len(names) != 1:
                    raise DiagnosticError("historical embedding parameter is ambiguous")
                self.launch(
                    event,
                    embedding_ids[row : row + group],
                    self.parameters[names[0]],
                    weight,
                    carry,
                    norm,
                    group,
                    5120,
                )
                if not torch.equal(carry, a):
                    raise DiagnosticError(
                        "historical embedding does not reproduce the common input"
                    )
            elif ".post_attention_layernorm" in key:
                self.launch(event, a, args[1][row : row + group], weight, norm, group, 5120)
                carry = (a.float() + args[1][row : row + group].float()).bfloat16()
            else:
                if prior is None:
                    raise DiagnosticError("retained residual terms are missing")
                previous_attention, previous_residual = prior
                left = previous_attention[row : row + group]
                right = previous_residual[row : row + group]
                if event["compiled"]["arg_names"][0] == "in_out_ptr0":
                    norm.copy_(a)
                    self.launch(event, norm, left, right, weight, group, 5120)
                    carry = (a.float() + (left.float() + right.float())).bfloat16()
                else:
                    outputs = [carry, norm]
                    if len(event["args"]) == 9:
                        desc = event["args"][6]
                        outputs.append(
                            torch.empty(
                                (group, 5120),
                                device=x.device,
                                dtype=getattr(torch, desc["dtype"].split(".")[-1]),
                            )
                        )
                    self.launch(event, a, left, right, weight, *outputs, group, 5120)
            result.append(norm)
            residuals.append(carry)
        normalized = torch.cat(result)
        return normalized if len(args) == 4 else (normalized, torch.cat(residuals))

    def point(self, arm, kind, args, instance=0):
        events = self.pointwise[arm][kind]
        if len(events) != (64 if kind == "silu" else 16):
            raise DiagnosticError("compiled pointwise layer inventory changed")
        event = events[instance]
        group = 1 if arm.endswith("m1") else 8
        if kind == "silu":
            x, out, _count = args
            if x.shape != (8, 34816) or out.shape != (8, 17408):
                raise DiagnosticError("SiLU cut shape changed")
            for row in range(0, 8, group):
                self.launch(event, x[row : row + group], out[row : row + group], group * 17408)
        elif kind == "sigmoid":
            x, gate, out, _count = args
            if x.shape != (8, 24, 256) or gate.shape != out.shape or out.shape != (8, 6144):
                raise DiagnosticError("sigmoid cut shape changed")
            for row in range(0, 8, group):
                self.launch(
                    event,
                    x[row : row + group],
                    gate[row : row + group],
                    out[row : row + group],
                    group * 6144,
                )
        else:
            raise DiagnosticError("unqualified pointwise stage")
        return None
