"""Explicitly armed, source-bound V2 Radiance adapter; imports no GPU code at rest.

This is an isolated diagnostic worker adapter, never a production extension.
Its captures synchronize GPU state and are deliberately unsuitable for timing
qualification. Native execution remains UNPROVED until its GPU fault campaign.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import re
import traceback
from pathlib import Path

import numpy as np

from qwen_r9700_lab.conformance_artifacts import capture_runtime
from qwen_r9700_lab.conformance_boundaries import BoundaryRecorder
from qwen_r9700_lab.conformance_dispatch import DispatchRecorder
from qwen_r9700_lab.conformance_model import state_names
from qwen_r9700_lab.conformance_reference import bf16, kv_scaling, reference_precision
from qwen_r9700_lab.conformance_replay import (
    OUTPUT_COMPONENTS,
    CampaignWriter,
    observation_domain,
    validate_plan,
)
from qwen_r9700_lab.conformance_state import FrameWriter
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    digest,
    integer,
    private_json,
    seal,
    write_private,
)


def temporal_column(running_column, accepted_count):
    return running_column + accepted_count - 1


def convolution_offset(accepted_count):
    return accepted_count - 1


def gather_paged(cache, block_table, consumed, *, block_size, kv_heads, head_dim):
    """Normalize R4D HND physical pages to logical [token, head, channel]."""
    integer(consumed)
    if cache.ndim != 4 or tuple(cache.shape[1:]) != (kv_heads, block_size, 2 * head_dim):
        raise DiagnosticError("unsupported R4D cache layout")
    table = np.asarray(block_table)
    count = (consumed + block_size - 1) // block_size
    if table.ndim != 1 or table.dtype.kind not in "iu" or len(table) < count:
        raise DiagnosticError("incomplete logical page table")
    ids = table[:count]
    if np.any(ids < 0) or np.any(ids >= cache.shape[0]) or len(set(ids.tolist())) != count:
        raise DiagnosticError("invalid or aliased pages within a writable sequence")
    ordered = (
        np.asarray(cache[ids]).transpose(0, 2, 1, 3).reshape(-1, kv_heads, 2 * head_dim)[:consumed]
    )
    return ordered[..., :head_dim].copy(), ordered[..., head_dim:].copy()


def gather_hybrid(
    conv, temporal, table, *, running_column, accepted_count, history_width, dim_first
):
    integer(running_column)
    if not 1 <= integer(accepted_count) <= 8:
        raise DiagnosticError("unsupported accepted-state width")
    column = temporal_column(running_column, accepted_count)
    if column >= len(table):
        raise DiagnosticError("missing accepted recurrent-state version")
    conv_id, temporal_id = int(table[running_column]), int(table[column])
    if not 0 <= conv_id < conv.shape[0] or not 0 <= temporal_id < temporal.shape[0]:
        raise DiagnosticError("recurrent state references a nonexistent block")
    window = np.asarray(conv[conv_id])
    if not dim_first:
        window = window.T
    start = convolution_offset(accepted_count)
    if window.ndim != 2 or window.shape[1] < start + history_width:
        raise DiagnosticError("convolution state misses the committed history window")
    return np.asarray(temporal[temporal_id]).copy(), window[:, start : start + history_width].copy()


def verify_sources(package_root: Path, binding: dict):
    authenticate(binding)
    if binding.get("schema") != "urn:qwen:radiance-native-binding:v1" or not binding.get("files"):
        raise DiagnosticError("missing pinned native source binding")
    for name, expected in binding["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise DiagnosticError("unsafe native binding path")
        path = package_root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise DiagnosticError("native source changed; adapter binding needs review")
    return binding["sha256"]


_KV_FORMATS = {
    "bf16": ({"auto", "bfloat16"}, {"torch.bfloat16"}),
    "fp8_e4m3fn": ({"fp8", "fp8_e4m3"}, {"torch.uint8", "torch.float8_e4m3fn"}),
}


def validate_native_kv_encoding(cache_dtype, storage_dtype, reference_encoding):
    # The source-bound R4D backend interprets fp8/fp8_e4m3 as E4M3FN even
    # though vLLM allocates its backing tensors as uint8. Other byte-backed
    # formats (E5M2, per-token scaling, etc.) are not interchangeable.
    formats, storage_types = _KV_FORMATS.get(reference_encoding, (set(), set()))
    if cache_dtype not in formats or str(storage_dtype) not in storage_types:
        raise DiagnosticError(
            f"native KV encoding differs from the declared reference: "
            f"cache={cache_dtype}, storage={storage_dtype}, reference={reference_encoding}"
        )


def as_cpu(tensor, *, storage=False, kv_encoding=None):
    # Importing this module on the host never imports torch or queries a device.
    import torch

    if storage:
        declared = kv_encoding or {
            "torch.float8_e4m3fn": "fp8_e4m3fn",
            "torch.bfloat16": "bf16",
        }.get(str(tensor.dtype))
        storage_types = _KV_FORMATS.get(declared, (set(), set()))[1]
        if str(tensor.dtype) not in storage_types:
            raise DiagnosticError("native KV storage must be declared BF16 or E4M3FN")
        encoding = torch.uint8 if declared == "fp8_e4m3fn" else torch.uint16
        return tensor.detach().view(encoding).contiguous().cpu().numpy().copy()
    return tensor.detach().float().contiguous().cpu().numpy().copy()


def export_call_tensor(tensor):
    """Raw storage for semantic call evidence; preserve integer and FP8 bits."""
    import torch

    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    elif str(value.dtype).startswith("torch.float8"):
        value = value.view(torch.uint8)
    return value.cpu().numpy().copy()


class ConformanceWorkerExtension:
    """Named worker RPCs using vLLM's standard serialization contract."""

    def qwen_conformance_install(self, plan_path: str, output_path: str, binding_path: str):
        return install(self, plan_path, output_path, binding_path)

    def qwen_conformance_finish(self):
        return finish(self)


def preserve_native_failure(root, plan, binding, stage, error):
    """Preserve the worker-side cause before the transport reduces it to engine death."""
    document = seal(
        {
            "schema": "urn:qwen:native-worker-failure:v1",
            "plan": plan["sha256"],
            "binding": binding["sha256"],
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    # Preserve the first cause, including during failed teardown.
    with contextlib.suppress(FileExistsError):
        write_private(root / "worker-error.json", document)


def diagnosed_native_call(function, root, plan, binding, stage):
    from functools import wraps

    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            preserve_native_failure(root, plan, binding, stage, error)
            raise

    return call


def install(worker, plan_path: str, output_path: str, binding_path: str):
    """Entry point for LLM.collective_rpc, only in the isolated GPU worker."""
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("GPU instrumentation is not armed")
    runner = worker.model_runner
    package = Path(inspect.getfile(type(runner))).resolve().parents[5]
    # .../site-packages/vllm/v1/worker/gpu/model_runner.py
    if package.name != "site-packages":
        package = next(
            (
                p
                for p in Path(inspect.getfile(type(runner))).resolve().parents
                if p.name == "site-packages"
            ),
            None,
        )
    if package is None:
        raise DiagnosticError("cannot locate the pinned runtime source")
    binding = private_json(Path(binding_path))
    verify_sources(package, binding)
    if getattr(runner, "_qwen_conformance_probe", None) is not None:
        raise DiagnosticError("native conformance probe is already installed")
    probe = RadianceProbe(
        runner, validate_plan(private_json(Path(plan_path))), Path(output_path), binding
    )
    try:
        probe.attach()
        # Capture at the real worker entry points, outside all nested observation
        # hooks. A transport-level EngineDeadError is insufficient for a negative control.
        for method in ("execute_model", "sample_tokens"):
            probe.hooks.replace(
                runner,
                method,
                diagnosed_native_call(
                    getattr(runner, method), probe.campaign.root, probe.plan, binding, method
                ),
            )
    except BaseException as error:
        try:
            preserve_native_failure(probe.campaign.root, probe.plan, binding, "install", error)
        finally:
            probe.detach()
        raise
    runner._qwen_conformance_probe = probe
    return {"installed": True, "binding": binding["sha256"], "gpu_qualification": "UNPROVED"}


def finish(worker):
    runner = worker.model_runner
    try:
        return runner._qwen_conformance_probe.finish()
    finally:
        del runner._qwen_conformance_probe


def capture_committed_state(
    worker, *, output_path, request_id, expected, plan_path, binding_path, quiescent=False
):
    """Explicit worker RPC for live/restored V2 state, independent of forced replay.

    The caller must hold scheduler admission and wait for any connector restore
    or bank handover to finish before setting quiescent. A GPU synchronization
    alone cannot establish that CPU-side ownership will stay unchanged. The
    requested prefix/count/pending identity is checked before a frame is saved.
    No generation or model output is forced and no snapshot is loaded/saved here.
    """
    from types import SimpleNamespace

    from qwen_r9700_lab.conformance_state import read_frame

    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1" or quiescent is not True:
        raise DiagnosticError("native state capture requires an armed, quiescent worker boundary")
    runner = worker.model_runner
    package = next(
        (
            p
            for p in Path(inspect.getfile(type(runner))).resolve().parents
            if p.name == "site-packages"
        ),
        None,
    )
    if package is None:
        raise DiagnosticError("cannot locate pinned native state producer")
    binding = private_json(Path(binding_path))
    verify_sources(package, binding)
    plan = validate_plan(private_json(Path(plan_path)))
    precision = reference_precision(plan.get("reference_profile", "radiance-fp8"))
    if Path(runner.model_config.model).resolve() != Path(plan["checkpoint"]).resolve():
        raise DiagnosticError("native state producer loaded a different checkpoint path")
    config = runner.model_config.hf_text_config.to_dict()
    kv_scaling(config, plan["kv_scales"], precision["profile"])
    validate_native_kv_encoding(
        runner.cache_config.cache_dtype, runner.kv_cache_dtype, precision["kv_encoding"]
    )
    parallel = runner.vllm_config.parallel_config
    if (
        config["model_type"] != "qwen3_5_text"
        or runner.cache_config.mamba_cache_mode != "align"
        or parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
    ):
        raise DiagnosticError("unadmitted native restore layout")
    if set(expected) != {"consumed", "pending", "input_digest"}:
        raise DiagnosticError("recovery capture requires exact position, pending token and prefix")
    index = runner.req_states.req_id_to_index.get(request_id)
    if index is None:
        raise DiagnosticError("requested native sequence is not resident")
    output = Path(output_path)
    # Reuse the same extraction code qualified by the independent replay, while
    # keeping its source and evidence identities distinct from forced execution.
    probe = RadianceProbe.__new__(RadianceProbe)
    probe.runner, probe.config, probe.plan = runner, config, plan
    probe.adapter = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    probe.execution = digest(
        {
            "plan": plan["execution"],
            "source": binding["sha256"],
            "capture": "quiescent-native-state",
        }
    )
    probe.campaign = SimpleNamespace(
        root=output.parent, coverage=state_names(config), record=lambda _: None
    )
    probe.capture(
        SimpleNamespace(idx_mapping_np=[index]),
        {**expected, "phase": "restore", "name": output.name},
        None,
        mode="native_recovery_capture",
    )
    if runner.req_states.req_id_to_index.get(request_id) != index:
        raise DiagnosticError("sequence ownership changed during native state capture")
    frame = read_frame(output)
    return {
        "frame": frame["sha256"],
        "consumed": frame["consumed"],
        "native_extraction": "UNPROVED",
        "quiescence": "ASSUMED: caller-held admission",
    }


class RadianceProbe:
    def __init__(self, runner, plan, root, binding):
        self.runner, self.plan, self.binding = runner, plan, binding
        precision = reference_precision(plan.get("reference_profile", "radiance-fp8"))
        c = runner.model_config.hf_text_config
        self.config = c.to_dict()
        kv_scaling(self.config, plan["kv_scales"], precision["profile"])
        parallel = runner.vllm_config.parallel_config
        if (
            c.model_type != "qwen3_5_text"
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise DiagnosticError("native adapter admits one dense Qwen3.5 TP1/DP1 worker")
        if (
            not runner.model_config.enforce_eager
            or runner.vllm_config.scheduler_config.async_scheduling
        ):
            raise DiagnosticError("full tensor capture requires explicitly serialized eager replay")
        if runner.cache_config.mamba_cache_mode != "align":
            raise DiagnosticError(
                "native hybrid adapter requires the reviewed align state convention"
            )
        if runner.vllm_config.kv_transfer_config is not None:
            raise DiagnosticError("replay must start independently with no snapshot connector")
        validate_native_kv_encoding(
            runner.cache_config.cache_dtype, runner.kv_cache_dtype, precision["kv_encoding"]
        )
        self.campaign = CampaignWriter(
            root,
            plan,
            coverage=state_names(self.config) + OUTPUT_COMPONENTS,
            backend="radiance-v2-native-candidate",
        )
        self.adapter = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.artifacts = capture_runtime()
        write_private(root / "runtime-before.json", self.artifacts)
        self.execution = digest(
            {
                "plan_execution": plan["execution"],
                "binding": binding["sha256"],
                "adapter": self.adapter,
                "schedule": "serialized-eager",
                "runtime_artifacts": self.artifacts["sha256"],
            }
        )
        positions, inputs = observation_domain(plan)
        self.boundaries = BoundaryRecorder(
            root / "boundaries",
            contract=plan["contract"],
            execution=self.execution,
            adapter=self.adapter,
            positions=positions,
            layers=c.num_hidden_layers,
            input_digests=inputs,
        )
        self.pending_capture, self.logits, self.batch = None, None, None
        self.index, self.cursor, self.handles = 0, 0, []
        self.positions = None
        from qwen_r9700_lab.conformance_instrumentation import CallRecorder, HookSet

        self.hooks = HookSet()
        self.calls = CallRecorder(
            root / "calls",
            contract=plan["contract"],
            execution=self.execution,
            adapter=self.adapter,
            tensor_export=export_call_tensor,
            is_tensor=lambda value: hasattr(value, "detach") and hasattr(value, "dtype"),
            mode=os.environ.get("QWEN_CONFORMANCE_CALL_MODE", "tensor"),
        )
        self.dispatch = None

    def detach(self):
        try:
            for handle in reversed(self.handles):
                handle.remove()
        finally:
            self.handles.clear()
            try:
                if getattr(self, "dispatch", None) is not None:
                    self.dispatch.close()
            finally:
                self.hooks.close()

    def attach(self):
        from qwen_r9700_lab.conformance_faults import install_experiment

        install_experiment(self)
        runner = self.runner
        original_sample, original_post = runner.sample, runner.postprocess_sampled
        original_logits = runner.model.compute_logits
        original_prepare = runner.prepare_inputs

        def prepare(*args, **kwargs):
            batch = original_prepare(*args, **kwargs)
            if batch.num_reqs != 1:
                raise DiagnosticError("native replay observed an unadmitted concurrent request")
            self.positions = batch.positions[: batch.num_tokens].detach().cpu().numpy().tolist()
            return batch

        def logits(*args, **kwargs):
            out = original_logits(*args, **kwargs)
            self.logits = as_cpu(out)
            return out

        def sample(hidden_states, batch, grammar_output):
            if grammar_output is not None:
                raise DiagnosticError(
                    "grammar and parser semantics need a separate checked protocol adapter"
                )
            result, ns, nr = original_sample(hidden_states, batch, grammar_output)
            n = int(ns[0].item())
            if not n:  # intermediate prefill chunk: no pending output yet
                return result, ns, nr
            if self.index >= len(self.campaign.expected):
                raise DiagnosticError("native backend exceeded its replay schedule")
            expected = self.campaign.expected[self.index]
            drafts = int(batch.num_draft_tokens)
            widths = self.plan.get("accepted_widths", [0] * (len(self.plan["forced_tokens"]) - 1))
            accepted = widths[self.index - 1] if self.index else 0
            if accepted > drafts or (drafts and drafts != 7):
                raise DiagnosticError("native verifier width is outside the admitted D7 schedule")
            # Capture genuine target results before injecting the diagnostic
            # accept boundary. The forced decision is never sold as a sample.
            if self.logits is None or self.logits.shape[0] != drafts + 1:
                raise DiagnosticError("missing causal target verification rows")
            natural = self.logits[accepted].copy()
            begin = self.cursor + (1 if self.index else 0)
            count = accepted + 1 if self.index else 1
            forced = self.plan["forced_tokens"][begin : begin + count]
            if len(forced) != count:
                raise DiagnosticError("missing forced output suffix")
            result.sampled_token_ids.fill_(-1)
            for j, token in enumerate(forced):
                result.sampled_token_ids[0, j] = token
            ns.fill_(count)
            nr.fill_(drafts - accepted)
            self.cursor = expected["consumed"] - len(self.plan["prefix"])
            self.next_proposal_accept = widths[self.index] if self.index < len(widths) else 0
            self.pending_capture = (batch, expected, natural)
            return result, ns, nr

        def post(*args, **kwargs):
            result = original_post(*args, **kwargs)
            if self.pending_capture is not None:
                batch, expected, target_logits = self.pending_capture
                self.capture(batch, expected, target_logits)
                self.pending_capture = None
                self.index += 1
            return result

        self.hooks.replace(runner, "prepare_inputs", prepare)
        self.hooks.replace(runner, "sample", sample)
        self.hooks.replace(runner, "postprocess_sampled", post)
        self.hooks.replace(runner.model, "compute_logits", logits)
        if runner.speculator is not None:
            original_propose = runner.speculator.propose

            def propose(*args, **kwargs):
                out = original_propose(*args, **kwargs)
                if out.ndim != 2 or out.shape[0] != 1 or out.shape[1] != 7:
                    raise DiagnosticError("unexpected native drafter representation")
                # Proposals after the pending token. A rejected suffix beyond
                # the plan uses public token 0, never another session's data.
                for j in range(7):
                    p = self.cursor + 1 + j
                    out[0, j] = (
                        self.plan["forced_tokens"][p] if p < len(self.plan["forced_tokens"]) else 0
                    )
                    # Perturb only proposals that the NEXT forced commit will
                    # reject. Accepted prefix and pending-token inputs stay fixed.
                    next_width = getattr(self, "next_proposal_accept", 0)
                    if j >= next_width and hasattr(self, "rejected_suffix_token"):
                        out[0, j] = self.rejected_suffix_token
                return out

            self.hooks.replace(runner.speculator, "propose", propose)

        layers = [
            (int(m.group(1)), module)
            for name, module in runner.model.named_modules()
            if (m := re.search(r"(?:^|\.)layers\.(\d+)$", name))
        ]
        if len(layers) != self.config["num_hidden_layers"]:
            raise DiagnosticError("native decoder inventory is incomplete")
        for layer, module in layers:
            for name, stage in (
                ("input_layernorm", "input_norm"),
                ("post_attention_layernorm", "post_attention_norm"),
            ):
                child = getattr(module, name)

                def observed(mod, args, output, layer=layer, stage=stage):
                    value = output[0] if isinstance(output, tuple) else output
                    self.record_boundary(layer, stage, as_cpu(value))

                self.handles.append(child.register_forward_hook(observed))

            def layer_output(mod, args, output, layer=layer):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise DiagnosticError("native residual boundary changed")
                value = bf16(as_cpu(output[0]) + as_cpu(output[1]))
                self.record_boundary(layer, "output", value)

            self.handles.append(module.register_forward_hook(layer_output))

        def call_context():
            if self.positions is None or self.index >= len(self.campaign.expected):
                return None
            expected = self.campaign.expected[self.index]
            if not any(
                p < expected["consumed"] and p in self.boundaries.position_set
                for p in self.positions
            ):
                return None
            return {
                "consumed": expected["consumed"],
                "input_digest": expected["input_digest"],
                "positions": self.positions,
                "phase": expected["phase"],
            }

        # Observe actual target module calls and numerical glue. Aliases hidden
        # inside compiled operators remain unqualified; calls.json lists what
        # really executed. No import-time or production hook installation.
        for name, module in runner.model.named_modules():
            site = "target." + (name or "root")
            self.calls.bind(module, "forward", site=site, hooks=self.hooks, context=call_context)
            if re.search(r"(?:^|\.)layers\.\d+$", name):
                self.calls.required.add(site)
        import sys

        # An explicit reviewed export inventory is required. Absence is recorded
        # as a gap; never infer complete device coverage from module hooks.
        entries = self.binding.get("native_entrypoints", [])
        if entries:
            self.dispatch = DispatchRecorder(
                self.campaign.root / "dispatch", execution=self.execution
            )
            for entry in entries:
                name = entry["binding"]["module"]
                module = sys.modules.get(name)
                aliases = [sys.modules.get(alias) for alias in entry["aliases"]]
                if module is None or any(alias is None for alias in aliases):
                    raise DiagnosticError("native dispatch binding module or alias is not loaded")
                self.dispatch.bind(module, entry["binding"], aliases=aliases)

        for module_name, functions in {
            "radiance_gdn": (
                "conv_prep",
                "conv_update",
                "fused_update",
                "kkt_solve",
                "output_norm",
                "recurrent_update",
                "fused_prefill",
            ),
            "radiance_mxfp4": ("mxfp4_linear", "mxfp4_linear_pq"),
        }.items():
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name in functions:
                self.calls.bind(
                    module,
                    name,
                    site=module_name + "." + name,
                    hooks=self.hooks,
                    context=call_context,
                )

    def record_boundary(self, layer, stage, values):
        if self.positions is None or values.shape[0] != len(self.positions):
            raise DiagnosticError("native boundary has no exact token positions")
        for row, position in enumerate(self.positions):
            if (
                self.index < len(self.campaign.expected)
                and position < self.campaign.expected[self.index]["consumed"]
            ):
                self.boundaries.record(position, layer, stage, values[row])

    def capture(self, batch, expected, target_logits, *, mode="forced_token_replay"):
        import torch
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        torch.cuda.synchronize()  # diagnostics only, not a production hook
        runner, c = self.runner, self.config
        precision = reference_precision(self.plan.get("reference_profile", "radiance-fp8"))
        index = int(batch.idx_mapping_np[0])
        consumed = int(runner.req_states.num_computed_tokens.gpu[index].item())
        pending = int(runner.req_states.last_sampled_tokens[index, 0].item())
        tokens = runner.req_states.all_token_ids.gpu[index, :consumed].cpu().numpy().astype("<i4")
        if (
            consumed != expected["consumed"]
            or pending != expected["pending"]
            or digest(tokens.tolist()) != expected["input_digest"]
        ):
            raise DiagnosticError(
                "native committed position/prefix differs from the forced schedule"
            )
        root = self.campaign.root / expected["name"]
        writer = FrameWriter(
            root,
            contract=self.plan["contract"],
            execution=self.execution,
            adapter=self.adapter,
            input_digest=expected["input_digest"],
            phase=expected["phase"],
            consumed=consumed,
            pending=pending,
            logical={"execution_mode": mode},
            expected=self.campaign.coverage,
        )
        writer.array("sequence.tokens", tokens)
        writer.array("sequence.position", np.asarray([consumed], dtype="<i8"))
        context = runner.compilation_config.static_forward_context
        groups = runner.kv_cache_config.kv_cache_groups
        seen = set()
        target_modules = {id(module) for _, module in runner.model.named_modules()}
        for gid, group in enumerate(groups):
            table = runner.block_tables.block_tables[gid].gpu[index].cpu().numpy()
            for name in group.layer_names:
                module = context[name]
                if id(module) not in target_modules:
                    # Drafter cache groups are not target state. Their physical
                    # layer numbers may coincide with the target's layer IDs.
                    continue
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match is None:
                    raise DiagnosticError("unmapped native cache owner")
                layer = int(match.group(1))
                if layer in seen:
                    raise DiagnosticError("duplicate native logical layer")
                seen.add(layer)
                prefix = f"layer.{layer:03d}."
                if c["layer_types"][layer] == "linear_attention":
                    running = int(runner.model_state._mamba_state_idx_gpu[index].item())
                    accepted = int(runner.model_state.num_accepted_tokens_gpu[index].item())
                    conv, state = module.kv_cache
                    column = temporal_column(running, accepted)
                    if running < 0 or column >= len(table):
                        raise DiagnosticError("invalid native committed state version")
                    # Copy only the selected bank/window, not the whole GPU pool.
                    selected_conv = as_cpu(conv[int(table[running])])
                    if not is_conv_state_dim_first():
                        selected_conv = selected_conv.T
                    offset = convolution_offset(accepted)
                    history = selected_conv[:, offset : offset + c["linear_conv_kernel_dim"] - 1]
                    if history.shape[-1] != c["linear_conv_kernel_dim"] - 1:
                        raise DiagnosticError("native convolution window is incomplete")
                    writer.array(prefix + "gdn", as_cpu(state[int(table[column])]))
                    writer.array(prefix + "conv", history)
                else:
                    cache = module.kv_cache
                    heads, width = c["num_key_value_heads"], c["head_dim"]
                    if cache.ndim != 4 or cache.shape[1] != heads or cache.shape[3] != width * 2:
                        raise DiagnosticError("native R4D HND cache layout changed")
                    block_size = cache.shape[2]
                    needed = (consumed + block_size - 1) // block_size
                    ids = table[:needed]
                    if (
                        len(ids) != needed
                        or np.any(ids < 0)
                        or np.any(ids >= cache.shape[0])
                        or len(set(ids.tolist())) != needed
                    ):
                        raise DiagnosticError("invalid native page ownership")
                    parts = [
                        as_cpu(
                            cache[int(b)], storage=True, kv_encoding=precision["kv_encoding"]
                        ).transpose(1, 0, 2)
                        for b in ids
                    ]
                    logical = np.concatenate(parts, axis=0)[:consumed]
                    for name, data in (
                        ("keys", logical[..., :width]),
                        ("values", logical[..., width:]),
                    ):
                        writer.add(
                            prefix + name,
                            np.ascontiguousarray(data).tobytes(),
                            dtype=precision["kv_encoding"],
                            shape=data.shape,
                        )
                    scales = np.asarray(
                        [module._k_scale_float, module._v_scale_float]
                        if precision["kv_fp8"]
                        else [1.0, 1.0],
                        dtype="<f4",
                    )
                    if not np.array_equal(
                        scales,
                        np.asarray(self.plan["kv_scales"].get(str(layer), [1.0, 1.0]), dtype="<f4"),
                    ):
                        raise DiagnosticError("native KV quantizer differs from the reference")
                    writer.array(prefix + "kv_scales", scales)
        if seen != set(range(c["num_hidden_layers"])):
            raise DiagnosticError("native cache observation omitted a layer")
        if target_logits is not None:
            if target_logits.shape != (c["vocab_size"],) or not np.isfinite(target_logits).all():
                raise DiagnosticError("full-vocabulary comparison requires the full target head")
            writer.array("output.logits", target_logits.astype("<f4"))
            writer.array("output.greedy", np.asarray([np.argmax(target_logits)], dtype="<i4"))
        writer.finish()
        self.campaign.record(root)

    def finish(self):
        try:
            return self._finish_capture()
        except Exception as error:
            preserve_native_failure(self.campaign.root, self.plan, self.binding, "finish", error)
            raise
        finally:
            self.detach()

    def _finish_capture(self):
        self.boundaries.finish()
        calls = self.calls.finish()
        dispatch = self.dispatch.finish() if self.dispatch is not None else None
        result = self.campaign.finish()
        write_private(self.campaign.root / "native-binding.json", self.binding)
        artifacts_after = capture_runtime()
        write_private(self.campaign.root / "runtime-after.json", artifacts_after)
        receipt = seal(
            {
                "schema": "urn:qwen:native-capture-receipt:v1",
                "schedule": result["sha256"],
                "semantic_calls": calls["sha256"],
                "semantic_call_mode": calls["mode"],
                "native_entrypoints": dispatch["sha256"]
                if dispatch
                else "UNPROVED: no reviewed export binding",
                "source_binding": self.binding["sha256"],
                "adapter": self.adapter,
                "runtime_before": self.artifacts["sha256"],
                "runtime_after": artifacts_after["sha256"],
                "exact_device_binary_attested": False,
                "forced_acceptance": True,
                "sampled_distribution": "UNPROVED",
                "asynchronous_schedule": "UNPROVED",
                "native_state_extraction": "UNPROVED",
            }
        )
        write_private(self.campaign.root / "native-receipt.json", receipt)
        return receipt
