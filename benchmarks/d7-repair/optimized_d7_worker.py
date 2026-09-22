"""Worker-side observation for the actual compiled, graph-enabled D7 path.

Repairs are installed before memory profiling, compilation and graph capture.
Performance observations do not copy activations or override sampled tokens.
The separate correctness probe forces identical tokens and retains private rows.
"""

import contextlib
import functools
import hashlib
import os
from collections import Counter
from pathlib import Path

import torch
from benchmark_d7_equivalence import EquivalenceProbe
from vllm.v1.worker.gpu_worker import Worker

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


class CompiledEquivalenceProbe(EquivalenceProbe):
    @staticmethod
    def validate_execution(runner):
        config = runner.vllm_config
        require(not runner.model_config.enforce_eager, "compiled replay cannot be eager")
        require(
            not config.scheduler_config.async_scheduling, "forced replay needs ordered sampling"
        )
        require(int(config.compilation_config.mode) != 0, "compilation was disabled")
        require(
            str(config.compilation_config.cudagraph_mode).split(".")[-1] != "NONE",
            "graphs disabled",
        )


class GraphObservation:
    def __init__(self, runner, root, profile=False):
        self.hooks = HookSet()
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        self.counts = Counter()
        self.head_shapes = Counter()
        self.draft = False
        self.profiler = None
        self.profile_requested = profile
        self.profile_active = False
        original_replay = torch.cuda.CUDAGraph.replay

        def replay(graph):
            self.counts["draft_graph_replays" if self.draft else "target_graph_replays"] += 1
            return original_replay(graph)

        self.hooks.replace(torch.cuda.CUDAGraph, "replay", replay)
        original_logits = runner.model.compute_logits

        def logits(*args, **kwargs):
            with self.scope("target_vocabulary_head"):
                result = original_logits(*args, **kwargs)
            self.head_shapes[str(tuple(result.shape))] += 1
            return result

        self.hooks.replace(runner.model, "compute_logits", logits)
        original_forward = runner.model.forward

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            self.counts["target_forward_calls"] += 1
            with self.scope("target_body"):
                return original_forward(*args, **kwargs)

        self.hooks.replace(runner.model, "forward", forward)
        if runner.speculator is not None:
            original_propose = runner.speculator.propose

            def propose(*args, **kwargs):
                self.draft = True
                self.counts["draft_calls"] += 1
                try:
                    with self.scope("drafter"):
                        return original_propose(*args, **kwargs)
                finally:
                    self.draft = False

            self.hooks.replace(runner.speculator, "propose", propose)

    def scope(self, name):
        if not self.profile_active:
            return contextlib.nullcontext()
        return torch.profiler.record_function("qwen_d7_stage/" + name)

    def start_profile(self):
        require(self.profile_requested and self.profiler is None, "invalid profile start")
        torch.cuda.synchronize()
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        self.profile_active = True

    def stop_profile(self):
        require(self.profile_active, "profile was not active")
        torch.cuda.synchronize()
        self.profiler.stop()
        self.profile_active = False
        self.profiler.export_chrome_trace(str(self.root / "profile-trace.json"))

    def close(self):
        self.hooks.close()
        if self.profile_active:
            self.stop_profile()
        return {"counts": dict(self.counts), "head_shapes": dict(self.head_shapes)}


class OptimizedWorker(Worker):
    def compile_or_warm_up_model(self):
        from optimized_d7_startup import startup_prefill_compatibility

        repairs = self._qwen_persistent_repairs
        self._qwen_startup_compatibility = None
        if repairs is None or repairs.prefill is None:
            return super().compile_or_warm_up_model()
        splits = self.vllm_config.compilation_config.splitting_ops or []
        require(
            any("qwen_gdn_attention_core" in name for name in splits),
            "GDN must be an explicit piecewise graph boundary",
        )
        require(
            any("unified_attention_with_output" in name for name in splits),
            "attention must be an explicit piecewise graph boundary",
        )
        import radiance_r4d_attn as attention

        with startup_prefill_compatibility(
            repairs,
            torch,
            attention.R4DAttentionImpl,
            max_tokens=self.scheduler_config.max_num_batched_tokens,
        ) as counts:
            result = super().compile_or_warm_up_model()
        self._qwen_startup_compatibility = {**counts, "binding_restored_before_requests": True}
        return result

    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        self._qwen_persistent_repairs = None
        self._qwen_compiled_dispatch = None
        self._qwen_performance_repairs = None
        manifest = os.environ.get("QWEN_OPTIMIZED_REPAIR")
        if manifest:
            from optimized_stock_norm import install_compiled_norms, prepare_native_gdn_norms
            from stock_gdn_runtime import RuntimeRepairs

            dispatch = prepare_native_gdn_norms(self.model_runner.model)
            self._qwen_persistent_repairs = RuntimeRepairs(
                manifest, target_model=self.model_runner.model
            )
            performance = os.environ.get("QWEN_OPTIMIZED_PERFORMANCE")
            if performance:
                from optimized_d7_performance import PerformanceRepairs

                self._qwen_performance_repairs = PerformanceRepairs(
                    performance, self.model_runner.model, self._qwen_persistent_repairs
                )
            residual = (
                self._qwen_performance_repairs.manifest["stages"]
                .get("residual_norm", {})
                .get("build")
                if self._qwen_performance_repairs
                else None
            )
            self._qwen_compiled_dispatch = install_compiled_norms(
                self.model_runner.model,
                self._qwen_persistent_repairs,
                residual_build=residual,
            )
            self._qwen_compiled_dispatch["gdn_reference_dispatch"] = dispatch
        marker = Path(os.environ["QWEN_OPTIMIZED_STARTUP_RECEIPT"])
        write_private(
            marker,
            seal(
                {
                    "installed_before_compile_and_capture": True,
                    "repair": self._qwen_persistent_repairs.receipt() if manifest else None,
                    "dispatch": self._qwen_compiled_dispatch,
                    "performance": self._qwen_performance_repairs.receipt()
                    if self._qwen_performance_repairs
                    else None,
                }
            ),
        )

    def qwen_optimized_metadata(self):
        from qwen_r9700_lab.conformance_artifacts import capture_runtime

        c = self.vllm_config.compilation_config
        return {
            "enforce_eager": self.model_config.enforce_eager,
            "compilation_mode": int(c.mode),
            "backend": c.backend,
            "graph_mode": str(c.cudagraph_mode),
            "capture_sizes": c.cudagraph_capture_sizes,
            "async_scheduling": self.scheduler_config.async_scheduling,
            "effective_capacity": {
                "max_num_seqs": self.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
                "max_model_len": self.model_config.max_model_len,
                "block_size": self.vllm_config.cache_config.block_size,
                "cache_dtype": self.vllm_config.cache_config.cache_dtype,
                "num_gpu_blocks": self.vllm_config.cache_config.num_gpu_blocks,
            },
            "diagnostic_sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "optimized_d7_worker.py",
                    "optimized_stock_norm.py",
                    "execution_mode_d7_worker.py",
                    "isolated_d7_capture.py",
                )
            },
            "repair": self._qwen_persistent_repairs.receipt()
            if self._qwen_persistent_repairs
            else None,
            "runtime": capture_runtime(),
            "compiled_dispatch": self._qwen_compiled_dispatch,
            "startup_compatibility": self._qwen_startup_compatibility,
            "performance": self._qwen_performance_repairs.receipt()
            if self._qwen_performance_repairs
            else None,
        }

    def qwen_optimized_begin(self, root, profile=False, task_path=None):
        require(not hasattr(self, "_qwen_observation"), "previous optimized probe is active")
        metadata = self.qwen_optimized_metadata()
        require(
            not metadata["enforce_eager"] and metadata["compilation_mode"] != 0,
            "benchmark silently fell back to eager",
        )
        self._qwen_observation = GraphObservation(self.model_runner, root, profile)
        self._qwen_forced = None
        if task_path is not None:
            task = private_json(Path(task_path))
            authenticate(task)
            require(task.get("repair_manifest") is None, "repairs must precede graph capture")
            self._qwen_forced = CompiledEquivalenceProbe(self.model_runner, task)
            self._qwen_forced.attach()
        return {"installed": True}

    def qwen_optimized_finish(self):
        forced = None
        try:
            if self._qwen_forced is not None:
                forced = self._qwen_forced.finish()
        finally:
            if self._qwen_forced is not None:
                self._qwen_forced.close()
            observation = self._qwen_observation.close()
            del self._qwen_observation, self._qwen_forced
        require(
            observation["counts"].get("target_graph_replays", 0) > 0,
            "no target graph replay was observed",
        )
        return {
            "observation": observation,
            "forced": forced,
            "performance": self._qwen_performance_repairs.receipt()
            if self._qwen_performance_repairs
            else None,
        }

    def qwen_optimized_profile(self, start):
        if start:
            self._qwen_observation.start_profile()
        else:
            self._qwen_observation.stop_profile()
        return {"profile_active": self._qwen_observation.profile_active}
