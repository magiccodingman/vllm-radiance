"""Explicit target-only arithmetic controls for isolated D7 qualification.

The serial fallback executes the unchanged pinned target operator once per row.
It is a correctness fallback, not a claim of an inexpensive implementation.
Large prefill batches keep their existing implementation; no drafter is patched.
"""

import functools
import re

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class TargetArithmetic:
    def __init__(self, model, hooks, *, norm=False, head=False, norm_build=None, head_group_size=1):
        import torch

        if type(head_group_size) is not int or head_group_size not in (1, 2, 4, 5):
            raise DiagnosticError("target head grouping is outside the qualified small-batch path")
        self.norm_calls = self.norm_rows = self.head_calls = self.head_rows = 0
        self.gdn_norm_calls = self.gdn_norm_rows = 0
        self.norm_mode = "serial-m1" if norm else "unchanged"
        self.head_mode = (
            ("serial-m1" if head_group_size == 1 else f"groups-{head_group_size}")
            if head
            else "unchanged"
        )
        self.norm_build = None
        candidate = None
        gdn_candidate = None
        if norm:
            from stock_m1_gdn_norm import StockM1GdnNorm

            gdn_candidate = StockM1GdnNorm()
        if norm_build is not None:
            from stock_m1_norm import StockM1Norm

            if not norm:
                raise DiagnosticError("normalization build requires the normalization repair")
            candidate = StockM1Norm(norm_build)
            self.norm_build = candidate.manifest["sha256"]
            self.norm_mode = "fixed-m1-kernel"

        def norm_forward(module, original, attention=False):
            @functools.wraps(original)
            def forward(x, residual=None):
                valid = (
                    x.ndim == 3 and x.shape[1:] in ((24, 256), (4, 256)) and residual is None
                    if attention
                    else x.ndim == 2 and x.shape[-1] == 5120
                )
                if not valid:
                    raise DiagnosticError("target normalization shape differs from the contract")
                if x.shape[0] > 8 or x.shape[0] == 0:
                    return original(x, residual)
                self.norm_calls += 1
                self.norm_rows += x.shape[0]
                if x.shape[0] == 1:
                    return original(x, residual)
                if candidate is not None:
                    return candidate(x, residual, module.weight, module.variance_epsilon)
                outputs = [
                    original(x[i : i + 1], residual[i : i + 1] if residual is not None else None)
                    for i in range(x.shape[0])
                ]
                if residual is None:
                    return torch.cat(outputs)
                return tuple(torch.cat([out[j] for out in outputs]) for j in (0, 1))

            return forward

        def head_apply(original):
            @functools.wraps(original)
            def apply(lm_head, hidden_states, embedding_bias=None):
                if hidden_states.ndim != 2 or hidden_states.shape[-1] != 5120:
                    raise DiagnosticError("target head shape differs from the contract")
                if lm_head.tp_size != 1:
                    raise DiagnosticError("serial target head requires TP1")
                if hidden_states.shape[0] > 8 or hidden_states.shape[0] == 0:
                    return original(lm_head, hidden_states, embedding_bias)
                self.head_calls += 1
                self.head_rows += hidden_states.shape[0]
                if hidden_states.shape[0] == 1:
                    return original(lm_head, hidden_states, embedding_bias)
                return torch.cat(
                    [
                        original(lm_head, hidden_states[i : i + head_group_size], embedding_bias)
                        for i in range(0, hidden_states.shape[0], head_group_size)
                    ]
                )

            return apply

        def gdn_norm_forward(module, original):
            @functools.wraps(original)
            def forward(x, z=None):
                if x.ndim != 2 or x.shape[1] != 128 or x.shape[0] % 48:
                    raise DiagnosticError(
                        "target GDN normalization shape differs from the contract"
                    )
                if x.shape[0] == 0 or x.shape[0] == 48 or x.shape[0] > 384:
                    return original(x, z)
                self.gdn_norm_calls += 1
                self.gdn_norm_rows += x.shape[0] // 48
                return gdn_candidate(x, z, module.weight, module.eps)

            return forward

        norms = attention_norms = gdn_norms = heads = 0
        for name, module in model.named_modules():
            if norm and re.fullmatch(
                r"(?:language_model\.)?model\.(?:layers\.\d+\."
                r"(?:input_layernorm|post_attention_layernorm)|norm)",
                name,
            ):
                if type(module).__name__ != "GemmaRMSNorm":
                    raise DiagnosticError("target normalization implementation changed")
                if module.weight.shape != (5120,):
                    raise DiagnosticError("target Gemma normalization width changed")
                hooks.replace(module, "forward", norm_forward(module, module.forward))
                norms += 1
            if norm and re.fullmatch(
                r"(?:language_model\.)?model\.layers\.\d+\.self_attn\.[qk]_norm", name
            ):
                if type(module).__name__ != "GemmaRMSNorm" or module.weight.shape != (256,):
                    raise DiagnosticError("target attention normalization contract changed")
                hooks.replace(module, "forward", norm_forward(module, module.forward, True))
                attention_norms += 1
            if norm and re.fullmatch(
                r"(?:language_model\.)?model\.layers\.\d+\.linear_attn\.norm", name
            ):
                if (
                    type(module).__name__ != "RMSNormGated"
                    or module.weight.shape != (128,)
                    or module.bias is not None
                    or module.group_size is not None
                    or module.norm_before_gate is not True
                    or module.activation != "silu"
                    or module._forward_method.__name__ != "forward_hip"
                ):
                    raise DiagnosticError("target GDN normalization implementation changed")
                hooks.replace(module, "forward", gdn_norm_forward(module, module.forward))
                gdn_norms += 1
            if head and name.endswith("logits_processor"):
                hooks.replace(module, "_apply_head", head_apply(module._apply_head))
                heads += 1
        if (norm and (norms != 129 or attention_norms != 32 or gdn_norms != 48)) or (
            head and heads != 1
        ):
            raise DiagnosticError(
                "target arithmetic module inventory differs from the pinned model"
            )

    def receipt(self):
        return {
            "norm_mode": self.norm_mode,
            "norm_build": self.norm_build,
            "norm_calls": self.norm_calls,
            "norm_rows": self.norm_rows,
            "gdn_norm_mode": "one-row-workgroups" if self.norm_mode != "unchanged" else "unchanged",
            "gdn_norm_calls": self.gdn_norm_calls,
            "gdn_norm_rows": self.gdn_norm_rows,
            "head_mode": self.head_mode,
            "head_calls": self.head_calls,
            "head_rows": self.head_rows,
            "scope": "Target batches of 1 through 8; larger prefill batches unchanged.",
        }
