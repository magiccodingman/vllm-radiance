"""Diagnostic native-program replay with explicit cache restoration.

The pilot checks a complete reference replay and an injected head discrepancy.
It does not qualify any unimplemented alternative stage or claim that captured
activation equality is a vocabulary top-k measurement.
"""

import functools
import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from execution_mode_d7_worker import ExecutionModeProbe, ExecutionModeWorker
from native_d7_replay_tape import NativeTape, TapeDispatch
from optimized_d7_worker import GraphObservation

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


class CacheRegion:
    def __init__(self, regions):
        self.regions = regions
        self.before = self.after = None

    def snapshot(self):
        return [
            bank.index_select(0, indices).to("cpu", copy=True) for bank, indices in self.regions
        ]

    def capture_before(self):
        self.before = self.snapshot()

    def capture_after(self):
        self.after = self.snapshot()

    def restore(self, values):
        for (bank, indices), value in zip(self.regions, values, strict=True):
            bank.index_copy_(0, indices, value.to(bank.device))

    def unchanged(self):
        return all(
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            for a, b in zip(self.snapshot(), self.after, strict=True)
        )

    matches_after = unchanged

    @contextmanager
    def replay(self):
        self.restore(self.before)
        try:
            yield
        finally:
            self.restore(self.after)


class TapeObserver:
    def __init__(self, runner, probe, observation, root, repairs=None):
        self.runner, self.probe, self.observation = runner, probe, observation
        self.root = Path(root)
        self.hooks = HookSet()
        self.tape = None
        self.context = None
        self.regions = {}
        self.batches = []
        self.forward_hidden = None
        self.groups = int(os.environ.get("QWEN_D7_TAPE_GROUPS", "1"))
        self.matrix = None
        if os.environ.get("QWEN_D7_STAGE_MATRIX") == "1":
            from native_d7_stage_matrix import StageMatrix

            self.matrix = StageMatrix(runner.model, runner=runner, repairs=repairs)

    def state_factory(self, name, args, kwargs):
        if name not in (
            "vllm.qwen_gdn_attention_core.default",
            "vllm.unified_kv_cache_update.default",
            "vllm.unified_attention_with_output.default",
        ):
            return None
        layer = kwargs.get("layer_name")
        if layer is None:
            layer = args[2] if "kv_cache_update" in name else args[4]
        if not isinstance(layer, str):
            from vllm.model_executor.layers.attention.attention import _resolve_layer_name

            layer = _resolve_layer_name(layer)
        identity = (name, layer)
        if identity in self.regions:
            return self.regions[identity]
        module = self.context.no_compile_layers[layer]
        if "gdn" in name:
            metadata = self.context.attn_metadata[layer]
            if metadata.num_prefills or metadata.num_actual_tokens != 8:
                raise DiagnosticError("native tape admits one eight-position decode only")
            slots = metadata.spec_state_indices_tensor
            if slots is None or slots.ndim != 2 or slots.shape[0] != 1:
                raise DiagnosticError("native tape requires explicit speculative state slots")
            recurrent = slots.flatten().to(dtype=torch.long)
            recurrent = recurrent[recurrent >= 0].unique()
            conv = slots[:, 0].to(dtype=torch.long)
            banks = module.kv_cache
            regions = [(banks[0], conv), (banks[1], recurrent)]
        else:
            bank = module.kv_cache
            if bank.ndim != 4 or bank.shape[1] != 4 or bank.shape[-1] != 512:
                raise DiagnosticError("native tape encountered an unqualified KV layout")
            slots = self.context.slot_mapping[layer]
            indices = (slots[slots >= 0] // bank.shape[2]).to(dtype=torch.long).unique()
            if not 1 <= indices.numel() <= 2:
                raise DiagnosticError("native tape has an unexpected KV write footprint")
            regions = [(bank, indices)]
        state = CacheRegion(regions)
        self.regions[identity] = state
        return state

    def attach(self):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from vllm.forward_context import get_forward_context, override_forward_context

        original_run = CachingAutotuner.run

        @functools.wraps(original_run)
        def kernel_run(kernel, *args, **kwargs):
            if self.tape is None:
                return original_run(kernel, *args, **kwargs)
            return self.tape.invoke(
                "inductor/" + kernel.fn.__name__,
                functools.partial(original_run, kernel),
                args,
                kwargs,
            )

        self.hooks.replace(CachingAutotuner, "run", kernel_run)
        model = self.runner.model
        original_forward = model.forward
        original_logits = model.compute_logits

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            positions = self.probe.pending_positions
            start = len(self.probe.schedule.prefix)
            expected = start + 8 * len(self.batches)
            if (
                self.observation.draft
                or len(self.batches) >= self.groups
                or positions != list(range(expected, expected + 8))
            ):
                return original_forward(*args, **kwargs)
            if self.tape is not None:
                raise DiagnosticError("native pilot attempted a duplicate group")
            free, _ = torch.cuda.mem_get_info()
            reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if free + reusable < 256 << 20:
                raise DiagnosticError(
                    "native tape pilot needs 256 MiB of available allocator space"
                )
            self.context = get_forward_context()
            self.tape = NativeTape(
                [*model.parameters(), *model.buffers()],
                state_factory=self.state_factory,
                context=lambda: override_forward_context(self.context),
                capture_filter=self.matrix.capture_filter if self.matrix is not None else None,
            )
            with TapeDispatch(self.tape):
                result = original_forward(*args, **kwargs)
            self.forward_hidden = result[0] if isinstance(result, tuple) else result
            return result

        @functools.wraps(original_logits)
        def logits(*args, **kwargs):
            if self.tape is None:
                return original_logits(*args, **kwargs)
            # The runner may select/copy prediction rows outside model.forward.
            # Admit only the all-eight-row identity bridge. Otherwise a tape
            # could replay a frozen head input and conceal every earlier fault.
            supplied = args[0] if args else kwargs["hidden_states"]
            hidden = self.forward_hidden
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.shape != supplied.shape
                or hidden.dtype != supplied.dtype
                or hidden.stride() != supplied.stride()
                or not torch.equal(hidden.view(torch.uint8), supplied.view(torch.uint8))
            ):
                raise DiagnosticError("native tape cannot bind the runner's head-input selection")
            if args:
                args = (hidden, *args[1:])
            else:
                kwargs = {**kwargs, "hidden_states": hidden}
            # Keep the native full head opaque; its HIP binding bypasses ATen.
            result = self.tape.invoke("target.full_bf16_head", original_logits, args, kwargs)
            self.tape.finish(result)
            self.tape.validate(result)
            index = len(self.tape.calls) - 1

            def wrong_head(*a, **kw):
                out = original_logits(*a, **kw).clone()
                out[:, 0] += 1
                return out

            bad = self.tape.replay((index, wrong_head))
            if torch.equal(bad, result):
                raise DiagnosticError("native output discrepancy was not detected")
            head_ref = self.tape.calls[index].args[0][0]
            producer = next(
                i for i in range(index - 1, -1, -1) if head_ref in self.tape.calls[i].result[0]
            )
            original_producer = self.tape.calls[producer].function
            output_leaf = self.tape.calls[producer].result[0].index(head_ref)

            def wrong_hidden(*a, **kw):
                from torch.utils._pytree import tree_flatten

                out = original_producer(*a, **kw)
                leaves, _ = tree_flatten(out)
                leaves[output_leaf].zero_()
                return out

            bad_prefix = self.tape.replay((producer, wrong_hidden))
            if torch.equal(bad_prefix, result):
                raise DiagnosticError("native replay concealed a corrupted head-input producer")
            early = next(
                i
                for i, call in enumerate(self.tape.calls)
                if call.name == "radiance.mxfp4_linear.default"
            )
            early_producer = self.tape.calls[early].function

            def wrong_early(*a, **kw):
                out = early_producer(*a, **kw)
                out.zero_()
                return out

            if torch.equal(self.tape.replay((early, wrong_early)), result):
                raise DiagnosticError("native replay concealed a corrupted first-layer projection")
            if not all(state.unchanged() for state in self.regions.values()):
                raise DiagnosticError("native tape failed to restore authoritative cache state")
            stage_records = self.matrix.run(self.tape, result) if self.matrix is not None else None
            if not all(state.unchanged() for state in self.regions.values()):
                raise DiagnosticError("isolated stage execution failed to restore cache state")
            record = seal(
                {
                    "status": "REFERENCE_REPLAY_CHECKED",
                    "positions": 8,
                    "first_absolute_position": len(self.probe.schedule.prefix)
                    + 8 * len(self.batches),
                    "diagnostic_sources": {
                        module.__name__: hashlib.sha256(
                            Path(module.__file__).read_bytes()
                        ).hexdigest()
                        for name, module in list(sys.modules.items())
                        if name.startswith("native_d7_") and getattr(module, "__file__", None)
                    },
                    "calls": len(self.tape.calls),
                    "operations": sorted({call.name for call in self.tape.calls}),
                    "cache_regions": len(self.regions),
                    "external_storage_bytes": self.tape.external_bytes,
                    "reference_full_logits_exact": True,
                    "injected_output_fault_detected": True,
                    "injected_prefix_fault_detected": True,
                    "injected_early_projection_fault_detected": True,
                    "authoritative_cache_restored": True,
                    "scope": "Native reference replay and isolated calls on correct inputs"
                    if self.matrix is not None
                    else "Native tape pilot, not an isolated-stage comparison.",
                    "stages": stage_records,
                }
            )
            self.batches.append(record)
            write_private(self.root / f"native-tape-group-{len(self.batches) - 1:03d}.json", record)
            if self.groups == 1:
                write_private(self.root / "native-tape-pilot.json", record)
            self.tape.close()
            self.tape = None
            self.regions = {}
            return result

        self.hooks.replace(model, "forward", forward)
        self.hooks.replace(model, "compute_logits", logits)


class NativeTapeWorker(ExecutionModeWorker):
    def qwen_optimized_begin(self, private, profile=False, task_path=None):
        if profile or task_path is None:
            raise DiagnosticError("native tape requires an untimed forced replay")
        self._qwen_observation = GraphObservation(self.model_runner, private, False)
        self._qwen_forced = ExecutionModeProbe(self.model_runner, private_json(Path(task_path)))
        self._tape_observer = TapeObserver(
            self.model_runner,
            self._qwen_forced,
            self._qwen_observation,
            private,
            repairs=self._qwen_persistent_repairs,
        )
        # Install inside the forced-token recorder so shadow calls cannot append
        # duplicate rows or consume a saved token.
        self._tape_observer.attach()
        self._qwen_forced.attach()
        return {"installed": True, "native_tape": "UNQUALIFIED"}

    def qwen_optimized_finish(self):
        try:
            forced = self._qwen_forced.finish()
        finally:
            # Unwind in the reverse order of installation. GraphObservation's
            # forward wrapper is below both the tape and the forced recorder.
            try:
                self._qwen_forced.close()
            finally:
                try:
                    self._tape_observer.hooks.close()
                finally:
                    observation = self._qwen_observation.close()
                    del self._qwen_forced, self._qwen_observation
        if observation["counts"].get("target_graph_replays", 0):
            raise DiagnosticError("graph replay occurred in the no-graph diagnostic")
        result = {
            "observation": observation,
            "forced": forced,
            "performance": self._qwen_performance_repairs.receipt()
            if self._qwen_performance_repairs
            else None,
            "release_speed_measurement": False,
        }
        if len(self._tape_observer.batches) != self._tape_observer.groups:
            raise DiagnosticError("native tape pilot did not observe its complete position group")
        result["native_tape"] = {
            "positions": 8 * len(self._tape_observer.batches),
            "groups": [b["sha256"] for b in self._tape_observer.batches],
        }
        return result
