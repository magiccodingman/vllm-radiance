"""Run synthetic startup batches without broadening live qualification.

GDN and attention are explicit piecewise graph splits. Their noncaptured
startup calls may use the original multi-sequence operators, including the
mixed speculative/non-speculative batches used by vLLM's kernel warm-up.
Restore every binding before the worker admits real requests.
"""

import contextlib

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.conformance_topk import require


@contextlib.contextmanager
def startup_prefill_compatibility(repairs, torch, attention_impl=None, *, max_tokens=2048):
    prefill = repairs.prefill
    native = prefill.native
    counts = {"synthetic_gdn_calls": 0}
    hooks = HookSet()

    def uncaptured():
        require(
            not torch.cuda.is_current_stream_capturing(),
            "unrepaired startup operators must not enter a captured graph",
        )

    # forward_core_fused resolves these names dynamically. Calling its original
    # while leaving either repair installed still rejects vLLM's two-request
    # decode warm-up. Recover the exact originals from the repair's binding map.
    originals = {}
    for name in ("conv_update", "recurrent_update"):
        entries = [e for e in repairs.hooks.entries if e[0] is native and e[1] == name]
        require(len(entries) == 1, "startup requires the exact GDN repair bindings")
        originals[name] = entries[0][2]

    def synthetic_operator(name, original):
        def call(*args, **kwargs):
            uncaptured()
            counts[name] = counts.get(name, 0) + 1
            return original(*args, **kwargs)

        return call

    def forward(layer, mixed_qkv, b, a, output):
        require(0 < mixed_qkv.shape[0] <= max_tokens, "unexpected synthetic graph warm-up batch")
        uncaptured()
        counts["synthetic_gdn_calls"] += 1
        return prefill.original(layer, mixed_qkv, b, a, output)

    for name, original in originals.items():
        hooks.replace(native, name, synthetic_operator(name, original))
    hooks.replace(native, "forward_core_fused", forward)
    if attention_impl is not None:
        repaired_attention = attention_impl.forward
        require(hasattr(repaired_attention, "__wrapped__"), "attention repair binding missing")
        # A stage-qualified performance adapter can be layered above the repair.
        # Follow the original binding map, not the outermost wrapper's predecessor.
        entries = [e for e in repairs.hooks.entries if e[0] is attention_impl and e[1] == "forward"]
        require(len(entries) == 1, "startup requires the exact attention repair binding")
        original_attention = entries[0][2]
        counts["synthetic_attention_calls"] = 0

        def attention(impl, layer, query, *args, **kwargs):
            require(
                0 < query.shape[0] <= max_tokens, "unexpected synthetic attention warm-up batch"
            )
            uncaptured()
            counts["synthetic_attention_calls"] += 1
            return original_attention(impl, layer, query, *args, **kwargs)

        hooks.replace(attention_impl, "forward", attention)
    try:
        yield counts
    finally:
        hooks.close()
