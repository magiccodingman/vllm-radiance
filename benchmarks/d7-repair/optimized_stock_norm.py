"""Opaque compiler bindings for the already qualified native norm repairs.

No arithmetic is rewritten here. Fake implementations describe fresh outputs;
the real implementations call the exact source/binary-bound eager candidates.
Unsupported shapes retain the loaded model's original operator. Registration
and module replacement must precede the first Dynamo trace and graph capture.
"""

import functools
import re
from collections import Counter

import torch

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def prepare_native_gdn_norms(model):
    """Preserve the qualified eager numerical operator under compiled dispatch.

    vLLM normally selects forward_native for Inductor. That is a different
    finite-precision implementation from the pinned FLA/HIP M1 reference.
    Select that reference explicitly before RuntimeRepairs validates/binds it.
    """
    selected = []
    for name, module in model.named_modules():
        if not re.fullmatch(r"(?:language_model\.)?model\.layers\.\d+\.linear_attn\.norm", name):
            continue
        if type(module).__name__ != "RMSNormGated" or module._forward_method.__name__ not in (
            "forward_native",
            "forward_hip",
        ):
            raise DiagnosticError("compiled GDN norm entry point is outside the pinned contract")
        selected.append((name, module, module._forward_method.__name__))
    if len(selected) != 48:
        raise DiagnosticError("compiled GDN norm inventory changed")
    for _, module, _ in selected:
        module._forward_method = module.forward_hip
    return {
        "modules": len(selected),
        "previous_entry_points": dict(Counter(p for _, _, p in selected)),
        "selected": "forward_hip",
        "reason": "preserve qualified numerical reference",
    }


def install_compiled_norms(model, repairs, *, residual_build=None):
    from stock_m1_gdn_norm import StockM1GdnNorm
    from stock_m1_norm import StockM1Norm

    candidate = StockM1Norm(repairs.manifest["norm_build"])
    residual_candidate = StockM1Norm(residual_build) if residual_build is not None else candidate
    gdn = StockM1GdnNorm()
    # Keep a separate restoration layer above the eager repair bindings.
    hooks = HookSet()
    repairs.compiled_norm_hooks = hooks
    originals = {}
    calls = Counter()

    @torch.library.custom_op("qwen_d7_qualified::gemma", mutates_args=())
    def gemma(x: torch.Tensor, weight: torch.Tensor, eps: float, key: str) -> torch.Tensor:
        calls[f"gemma/{x.shape[0]}"] += 1
        if 1 < x.shape[0] <= 8:
            return candidate(x, None, weight, eps)
        return originals[key](x, None)

    @gemma.register_fake
    def gemma_fake(x, weight, eps, key):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    @torch.library.custom_op("qwen_d7_qualified::gemma_residual", mutates_args=())
    def gemma_residual(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, key: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls[f"residual/{x.shape[0]}"] += 1
        if 1 < x.shape[0] <= 8:
            return residual_candidate(x, residual, weight, eps)
        return originals[key](x, residual)

    @gemma_residual.register_fake
    def gemma_residual_fake(x, residual, weight, eps, key):
        return (
            torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(residual, memory_format=torch.contiguous_format),
        )

    @torch.library.custom_op("qwen_d7_qualified::gdn", mutates_args=())
    def gdn_norm(
        x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, key: str
    ) -> torch.Tensor:
        calls[f"gdn/{x.shape[0]}"] += 1
        if 48 < x.shape[0] <= 384:
            return gdn(x, z, weight, eps)
        return originals[key](x, z)

    @gdn_norm.register_fake
    def gdn_fake(x, z, weight, eps, key):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    def wrap_gemma(module, key):
        original = originals[key]

        @functools.wraps(original)
        def forward(x, residual=None):
            # vLLM can reuse one symbolic trace without evaluating shape guards.
            # Select the numerical path inside the opaque runtime operator.
            if residual is None:
                return gemma(x, module.weight, module.variance_epsilon, key)
            return gemma_residual(x, residual, module.weight, module.variance_epsilon, key)

        return forward

    def wrap_gdn(module, key):
        original = originals[key]

        @functools.wraps(original)
        def forward(x, z=None):
            return gdn_norm(x, z, module.weight, module.eps, key)

        return forward

    norms = gdns = 0
    for name, module in model.named_modules():
        norm = re.fullmatch(
            r"(?:language_model\.)?model\.(?:layers\.\d+\."
            r"(?:input_layernorm|post_attention_layernorm|self_attn\.[qk]_norm)|norm)",
            name,
        )
        gated = re.fullmatch(r"(?:language_model\.)?model\.layers\.\d+\.linear_attn\.norm", name)
        if not (norm or gated):
            continue
        if not hasattr(module.forward, "__wrapped__"):
            raise DiagnosticError("compiled norm binding requires the installed eager repair")
        originals[name] = module.forward.__wrapped__
        hooks.replace(
            module, "forward", wrap_gemma(module, name) if norm else wrap_gdn(module, name)
        )
        norms += bool(norm)
        gdns += bool(gated)
    if (norms, gdns) != (161, 48):
        raise DiagnosticError("compiled normalization inventory changed")
    return {
        "norm_modules": norms,
        "gdn_norm_modules": gdns,
        "norm_build": candidate.manifest["sha256"],
        "residual_norm_build": residual_candidate.manifest["sha256"],
        "arithmetic": "qualified M1 arithmetic contract; original large-prefill operators",
        "compiler_binding": "torch.library.custom_op with fresh fake outputs",
        "runtime_calls_including_graph_capture": calls,
    }
