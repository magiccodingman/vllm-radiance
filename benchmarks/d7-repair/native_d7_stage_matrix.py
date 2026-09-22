"""Common-input native stage comparisons followed by the validated reference suffix.

Only supported concrete cuts are emitted. Equal local tensors and hidden state
may reuse an already checked full-vector result; any unequal cut is replayed all
the way to the actual vocabulary head. Multiple layer instances are never
changed together. This is diagnostic evidence, not a release timing.
"""

import functools
import json
import os
import re
from collections import defaultdict

import torch

from qwen_r9700_lab.conformance_topk import compare_rows, summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

PAIRS = {
    "old": ("old_m1", "old_m8"),
    "fixed": ("fix1_m1", "fix1_m8"),
    "fix1_modes": ("fix1_eager_m8", "fix1_m8"),
    "final_modes": ("final_eager_m8", "final_m8"),
}


def serial(function, args, kwargs, width, chunk=1):
    """Rows are independent here; callers must never use this for recurrence."""

    def row(value, start):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == width:
            return value[start : start + chunk]
        return value

    results = [
        function(*(row(v, i) for v in args), **{k: row(v, i) for k, v in kwargs.items()})
        for i in range(0, width, chunk)
    ]
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results)
    if isinstance(results[0], tuple):
        return tuple(torch.cat([r[i] for r in results]) for i in range(len(results[0])))
    raise DiagnosticError("serial stage has an unsupported output interface")


def rows(logits):
    values = logits.float().cpu().numpy()
    return [summarize_logits(row) for row in values]


def shared_versions(reference, m1=None):
    m1 = m1 or reference
    return {
        "old_m1": m1,
        "old_m8": reference,
        "fix1_m1": m1,
        "fix1_m8": reference,
        "fix1_eager_m8": reference,
        "final_m8": reference,
        "final_eager_m8": reference,
    }


class StageMatrix:
    def __init__(self, model, *, runner=None, repairs=None):
        self.model = model
        self.parameters = {}
        for name, value in [*model.named_parameters(), *model.named_buffers()]:
            self.parameters.setdefault(value.untyped_storage().data_ptr(), []).append(name)
        self.inventory = defaultdict(set)
        self.history = None
        if os.environ.get("QWEN_D7_HISTORICAL_CATALOGS"):
            from native_d7_historical_stages import HistoricalStages

            self.history = HistoricalStages(
                model, json.loads(os.environ["QWEN_D7_HISTORICAL_CATALOGS"])
            )
        self.filters = os.environ.get("QWEN_D7_STAGE_FILTER", "").split("|")
        self.norm_cuts = {}
        self.embedding_ids = None
        self.silu_index = 0
        self.state_stages = None
        self.attention_stages = None
        if os.environ.get("QWEN_D7_ATTENTION_STAGES") == "1":
            from native_d7_attention_stages import AttentionStages

            self.attention_stages = AttentionStages(self.history)
        if os.environ.get("QWEN_D7_STATE_STAGES") == "1":
            from native_d7_state_stages import StateStages

            self.state_stages = StateStages(runner, repairs)

    def capture_filter(self, name, args, kwargs):
        if self.attention_stages is not None and name.startswith(("vllm.", "inductor/")):
            return True
        if self.state_stages is not None and name in (
            "vllm.qwen_gdn_attention_core.default",
            "vllm.unified_attention_with_output.default",
        ):
            return True
        return (
            name.startswith(("qwen_d7_qualified.", "radiance.mxfp4_linear."))
            or name == "target.full_bf16_head"
            or "embedding" in name
            or "silu_slice" in name
            or "sigmoid_view" in name
        )

    def cut_cases(self, call):
        args, _kwargs = call.cut[0][0].thaw(call.cut[0])
        fn = call.function
        if self.attention_stages is not None and call.name.startswith("vllm."):
            yield from self.attention_stages.cases(call)
        if self.state_stages is not None and call.name.startswith("vllm."):
            yield from self.state_stages.cases(call)
            return
        if call.name == "radiance.mxfp4_linear.default":
            weights = self.parameters.get(args[1].untyped_storage().data_ptr(), [])
            if len(weights) != 1:
                raise DiagnosticError("projection identity is ambiguous")
            name = weights[0]
            match = re.search(r"layers\.(\d+)\.(.*)", name)
            if match is None:
                raise DiagnosticError("non-target projection in target tape")
            layer = int(match[1])
            role = match[2]
            if "gate_up_proj" in role:
                stage = "MLP gate/up"
            elif "down_proj" in role:
                stage = "MLP down"
            elif "linear_attn" in role:
                stage = "GDN output" if "out_proj" in role else "GDN input"
            elif "self_attn" in role:
                stage = "Attention input" if "qkv_proj" in role else "Attention output"
            else:
                raise DiagnosticError("unknown target projection role")

            def quant_m1(*a, **kw):
                from vllm import _custom_ops as ops

                original = ops.scaled_fp8_quant

                def quant(x, *qa, **qkw):
                    if x.shape[0] != 8:
                        return original(x, *qa, **qkw)
                    outputs = [original(x[i : i + 1], *qa, **qkw) for i in range(8)]
                    return tuple(torch.cat([r[j] for r in outputs]) for j in (0, 1))

                ops.scaled_fp8_quant = quant
                try:
                    return fn(*a, **kw)
                finally:
                    ops.scaled_fp8_quant = original

            # The same FP8 quantizer is executed in all versions. Serializing
            # only that call leaves the M8 projection unchanged.
            yield stage + " activation FP8 quantization", str(layer), shared_versions(fn, quant_m1)

            def project_m1(*a, **kw):
                import radiance_mxfp4 as native

                original = native._ext.launch
                observed = []

                def launch(x, w, ws, wr, xs, out, m, n, k, stream):
                    if m != 8:
                        return original(x, w, ws, wr, xs, out, m, n, k, stream)
                    observed.append(m)
                    for i in range(8):
                        original(x + i * k, w, ws, wr, xs + i * 4, out + i * n * 2, 1, n, k, stream)

                native._ext.launch = launch
                try:
                    result = fn(*a, **kw)
                    if observed != [8]:
                        raise DiagnosticError(
                            "serial projection did not intercept the actual M8 dispatch"
                        )
                    return result
                finally:
                    native._ext.launch = original

            yield stage + " projection", str(layer), shared_versions(fn, project_m1)
        elif call.name.startswith("qwen_d7_qualified."):
            key = args[-1]
            if not isinstance(key, str):
                raise DiagnosticError("normalization identity missing")
            match = re.search(r"layers\.(\d+)\.(.*)", key)
            layer = "final" if match is None else str(int(match[1]))
            if call.name == "qwen_d7_qualified.gdn.default":
                stage = "GDN output gated normalization"
                chunk = 48
                width = 384
            else:
                chunk = 1
                width = 8
                if match is None:
                    stage = "Final normalization/layout"
                elif "post_attention_layernorm" in key:
                    stage = "Post-attention/GDN residual/normalization"
                elif "input_layernorm" in key:
                    stage = (
                        "Embedding + first input normalization"
                        if layer == "0"
                        else "Layer input residual/normalization"
                    )
                else:
                    stage = "Attention Q/K normalization"
                    layer += ":q" if ".q_norm" in key else ":k"

            def m1(*a, **kw):
                return serial(fn, a, kw, width, chunk)

            versions = shared_versions(fn, m1)
            # The original compiled norm used different fused generated IR.
            # It is deliberately absent until that exact IR is admitted.
            versions.pop("old_m1")
            versions.pop("old_m8")
            if self.history is not None and stage != "Attention Q/K normalization":
                prior = None
                if stage in ("Final normalization/layout", "Layer input residual/normalization"):
                    previous = 63 if layer == "final" else int(layer) - 1
                    prefix = (
                        key.split(".layers.")[0] if ".layers." in key else key.rsplit(".", 1)[0]
                    )
                    previous_key = prefix + f".layers.{previous}.post_attention_layernorm"
                    previous_args, _ = self.norm_cuts[previous_key][0].thaw(
                        self.norm_cuts[previous_key]
                    )
                    prior = previous_args[:2]

                def historical(arm, *a, **kw):
                    return self.history.norm(
                        arm, key, a, prior=prior, embedding_ids=self.embedding_ids
                    )

                versions["old_m1"] = functools.partial(historical, "m1")
                versions["old_m8"] = functools.partial(historical, "m8")
            yield stage, layer, versions
        elif (
            "silu_slice" in call.name or "sigmoid_view" in call.name
        ) and self.history is not None:
            kind = "silu" if "silu_slice" in call.name else "sigmoid"
            layer = str(self.silu_index if kind == "silu" else self.sigmoid_index * 4 + 3)
            if kind == "silu":
                self.silu_index += 1
            else:
                self.sigmoid_index += 1

            def compiled(arm, *a, **kw):
                instance = int(layer) if kind == "silu" else int(layer) // 4
                return self.history.point(arm, kind, a, instance)

            def eager(*a, **kw):
                if kind == "silu":
                    torch.ops._C.silu_and_mul(a[1], a[0])
                else:
                    a[2].copy_(a[0].reshape(8, 6144) * torch.sigmoid(a[1]))
                return None

            old_m1 = functools.partial(compiled, "m1")
            old_m8 = functools.partial(compiled, "m8")
            versions = {
                "old_m1": old_m1,
                "old_m8": old_m8,
                "fix1_m1": functools.partial(compiled, "fix1_m1"),
                "fix1_m8": functools.partial(compiled, "fix1_m8"),
                "fix1_eager_m8": eager,
                "final_m8": fn,
                "final_eager_m8": eager,
            }
            yield (
                "MLP SiLU and gating" if kind == "silu" else "Attention output gating",
                layer,
                versions,
            )
        elif call.name == "target.full_bf16_head":
            processors = [
                m for n, m in self.model.named_modules() if n.endswith("logits_processor")
            ]
            if len(processors) != 1:
                raise DiagnosticError("target head identity changed")
            processor = processors[0]

            def old(*a, **kw):
                from vllm.model_executor.layers.utils import rocm_unquantized_gemm

                original = processor._apply_head

                def apply(lm_head, hidden, embedding_bias=None):
                    return rocm_unquantized_gemm(None, hidden, lm_head.weight, embedding_bias)

                processor._apply_head = apply
                try:
                    return fn(*a, **kw)
                finally:
                    processor._apply_head = original

            def m1(*a, **kw):
                return serial(old, a, kw, 8)

            versions = shared_versions(fn, m1)
            versions["old_m8"] = old
            yield "Full BF16 target head", "head", versions

    def run(self, tape, logits):
        if not tape.qualified:
            raise DiagnosticError("unqualified reference suffix")
        reference = rows(logits)
        records = []
        self.silu_index = 0
        self.sigmoid_index = 0
        self.norm_cuts = {}
        if self.attention_stages is not None:
            self.attention_stages.prepare(tape, self.parameters)
        for call in tape.calls:
            if call.cut is None:
                continue
            a, _ = tape.thaw(call.cut[0])
            if call.name.startswith("qwen_d7_qualified."):
                self.norm_cuts[a[-1]] = call.cut[0]
            if "embedding" in call.name:
                self.embedding_ids = a[0]
        for index, call in enumerate(tape.calls):
            if call.cut is None:
                continue
            for stage, instance, versions in self.cut_cases(call):
                if self.filters != [""] and not any(token in stage for token in self.filters):
                    continue
                values = {}
                evidence = {}
                for fn in versions.values():
                    # Shared source/dispatch identity is explicit. Execute each
                    # unique actual callable on the common cut once.
                    if fn in values:
                        continue
                    exact, output_exact, state_exact = tape.check_cut(index, fn)
                    values[fn] = reference if exact else rows(tape.replay((index, fn)))
                    evidence[fn] = {
                        "local_output_exact": output_exact,
                        "local_state_exact": state_exact,
                        "suffix": "validated reference reuse"
                        if exact
                        else "native full suffix replay",
                    }
                comparisons = {}
                for column, (left, right) in PAIRS.items():
                    if left not in versions or right not in versions:
                        continue
                    comparisons[column] = [
                        compare_rows(a, b)
                        for a, b in zip(
                            values[versions[left]], values[versions[right]], strict=True
                        )
                    ]
                records.append(
                    {
                        "stage": stage,
                        "instance": instance,
                        "call_index": index,
                        "comparisons": comparisons,
                        "variants": {name: evidence[fn] for name, fn in versions.items()},
                    }
                )
                self.inventory[stage].add(instance)
        return records
