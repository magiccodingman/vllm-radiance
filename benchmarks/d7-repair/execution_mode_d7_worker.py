"""Controlled eager/compiled diagnostics with the final D7 repair integration.

Uses OptimizedWorker.load_model unchanged in both modes, including the opaque
normalization bindings and qualified performance adapters. No diagnostic result
from this worker is a release timing or a proof of isolated-stage equivalence.
"""

import hashlib
from pathlib import Path

import torch
from benchmark_d7_equivalence import EquivalenceProbe
from isolated_d7_capture import BoundaryCapture
from optimized_d7_worker import GraphObservation, OptimizedWorker
from vllm.v1.worker.gpu_worker import Worker

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


class ExecutionModeProbe(EquivalenceProbe):
    @staticmethod
    def validate_execution(runner):
        c = runner.vllm_config
        eager = runner.model_config.enforce_eager
        if eager != (int(c.compilation_config.mode) == 0):
            raise DiagnosticError("eager and compiler configuration disagree")
        if str(c.compilation_config.cudagraph_mode).split(".")[-1] != "NONE":
            raise DiagnosticError("execution-mode diagnostic requires graphs disabled")
        if c.scheduler_config.async_scheduling:
            raise DiagnosticError("execution-mode replay requires ordered sampling")


class ExecutionModeWorker(OptimizedWorker):
    def compile_or_warm_up_model(self):
        if not self.model_config.enforce_eager:
            return super().compile_or_warm_up_model()
        # Retain the same temporary startup adapters, but there are no compiled
        # graph boundaries to validate in the explicit eager control.
        from optimized_d7_startup import startup_prefill_compatibility

        repairs = self._qwen_persistent_repairs
        self._qwen_startup_compatibility = None
        if repairs is None or repairs.prefill is None:
            return Worker.compile_or_warm_up_model(self)
        import radiance_r4d_attn as attention

        with startup_prefill_compatibility(
            repairs,
            torch,
            attention.R4DAttentionImpl,
            max_tokens=self.scheduler_config.max_num_batched_tokens,
        ) as counts:
            result = Worker.compile_or_warm_up_model(self)
        self._qwen_startup_compatibility = {**counts, "binding_restored_before_requests": True}
        return result

    def qwen_optimized_begin(self, private, profile=False, task_path=None):
        if profile or task_path is None:
            raise DiagnosticError("execution-mode diagnostic requires forced replay, not timing")
        if hasattr(self, "_qwen_observation"):
            raise DiagnosticError("previous execution-mode probe is active")
        self._qwen_observation = GraphObservation(self.model_runner, private, False)
        try:
            self._qwen_forced = ExecutionModeProbe(self.model_runner, private_json(Path(task_path)))
            self._qwen_forced.attach()
        except BaseException:
            if hasattr(self, "_qwen_forced"):
                self._qwen_forced.close()
                del self._qwen_forced
            self._qwen_observation.close()
            del self._qwen_observation
            raise
        return {"installed": True, "release_speed_measurement": False}

    def qwen_optimized_finish(self):
        try:
            forced = self._qwen_forced.finish()
        finally:
            self._qwen_forced.close()
            observation = self._qwen_observation.close()
            del self._qwen_forced, self._qwen_observation
        if observation["counts"].get("target_graph_replays", 0):
            raise DiagnosticError("graph replay occurred in the no-graph diagnostic")
        return {
            "observation": observation,
            "forced": forced,
            "performance": self._qwen_performance_repairs.receipt()
            if self._qwen_performance_repairs
            else None,
            "release_speed_measurement": False,
        }


class ModeBoundaryCapture(BoundaryCapture):
    """Keep decode groups plus nine sampled prompt positions, without full KV copies.

    First eight prompt rows can expose a prefill error before it accumulates;
    the final prompt row connects to the separately recorded prefill prediction.
    This is sampled prefill coverage, not every prompt position or full state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.probe.schedule.prefix)
        self.prefill_positions = set(range(min(8, n))) | {n - 1}
        self.prefill_batches = []
        self.parameter_storage = {}
        model = self.probe.runner.model
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            self.parameter_storage.setdefault(value.untyped_storage().data_ptr(), []).append(name)

    def positions(self):
        if self.observation.draft or self.busy:
            return None
        positions = self.probe.pending_positions
        if not positions:
            return None
        if positions[-1] < len(self.probe.schedule.prefix):
            selected = tuple(p for p in positions if p in self.prefill_positions)
            return selected or None
        return super().positions()

    def save_tree(self, value, key, rows, saved):
        if isinstance(value, torch.Tensor):
            if value.untyped_storage().data_ptr() in self.parameter_storage:
                return
            actual = self.probe.pending_positions
            if actual[-1] < len(self.probe.schedule.prefix) and value.ndim > 0:
                width = len(actual)
                if value.shape[0] not in (width, width * 48):
                    return
                if value.numel() > width * 65536:
                    return
                selected = [i for i, p in enumerate(actual) if p in self.prefill_positions]
                if value.shape[0] == width * 48:
                    value = value.reshape(width, 48, *value.shape[1:])
                value = value[selected]
                if value.ndim > 2 and value.shape[1] == 48:
                    value = value.reshape(len(selected) * 48, *value.shape[2:])
        super().save_tree(value, key, rows, saved)

    def invoke(self, name, function, args, kwargs, schema=None, signature=None):
        active = self.positions() is not None
        identities = set()
        if active:
            for value in (*args, *kwargs.values()):
                if isinstance(value, str) and value.startswith(("model.", "language_model.model.")):
                    identities.add(value)
                elif isinstance(value, torch.Tensor):
                    identities.update(
                        self.parameter_storage.get(value.untyped_storage().data_ptr(), ())
                    )
        result = super().invoke(name, function, args, kwargs, schema=schema, signature=signature)
        if active:
            self.events[-1]["logical_identities"] = sorted(identities)
        return result

    def flush(self):
        prompt = self.batch_positions is not None and self.batch_positions[-1] < len(
            self.probe.schedule.prefix
        )
        super().flush()
        if prompt:
            self.prefill_batches.append(self.batches.pop())
            # Base filenames use len(batches); retain a separate prompt name
            # so the first decode group cannot overwrite its captured tensors.
            record = self.prefill_batches[-1]
            old = self.root / record["file"]
            new = self.root / f"prefill-{len(self.prefill_batches) - 1:04d}.pt"
            old.rename(new)
            old.with_suffix(".json").rename(new.with_suffix(".json"))
            doc = private_json(new.with_suffix(".json"))
            doc.pop("sha256")
            doc["tensor_file"] = new.name
            doc = seal(doc)
            # Replacing our own unpublished metadata; never overwrite evidence
            # from another invocation or an already admitted capture.
            from qwen_r9700_lab.conformance_queue import replace_private

            replace_private(new.parent, new.with_suffix(".json").name, doc)
            record["file"], record["sha256"] = new.name, doc["sha256"]

    def finish(self):
        result = super().finish()
        observed = [p for b in self.prefill_batches for p in b["positions"]]
        if len(observed) != len(self.prefill_positions) or set(observed) != self.prefill_positions:
            raise DiagnosticError("sampled prefill boundary coverage is incomplete")
        record = seal(
            {
                "schema": "qwen.execution-mode-prefill-capture.v1",
                "positions": sorted(self.prefill_positions),
                "batches": self.prefill_batches,
                "decode_capture": result["sha256"],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "sampled prompt activations; full recurrent/KV state is not captured",
            }
        )
        write_private(self.root / "prefill-manifest.json", record)
        return {**result, "prefill_capture": record["sha256"]}


class ExecutionModeCaptureWorker(ExecutionModeWorker):
    def qwen_optimized_begin(self, private, profile=False, task_path=None):
        result = super().qwen_optimized_begin(private, profile, task_path)
        self._mode_capture = ModeBoundaryCapture(
            self._qwen_forced,
            self._qwen_observation,
            Path(private) / "isolated-boundaries",
            require_compiled=not self.model_config.enforce_eager,
        )
        self._mode_capture.attach()
        return result

    def qwen_optimized_finish(self):
        try:
            capture = self._mode_capture.finish()
        except BaseException:
            self._qwen_forced.close()
            self._qwen_observation.close()
            raise
        finally:
            self._mode_capture.hooks.close()
        result = super().qwen_optimized_finish()
        return {**result, "isolated_capture": capture, "release_output_bridge": "UNQUALIFIED"}
