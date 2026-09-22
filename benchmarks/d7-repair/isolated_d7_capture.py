"""Observe compiled launches after compilation, without changing the fused IR.

This diagnostic disables graph replay so Python can copy operation boundaries.
It must pass an output bridge against the uninstrumented graph-enabled release.
Tensor contents remain private. None of its timings qualify as release timings.
"""

import functools
import hashlib
from collections import Counter
from pathlib import Path

import torch
from optimized_d7_worker import CompiledEquivalenceProbe, GraphObservation, OptimizedWorker
from torch.utils._python_dispatch import TorchDispatchMode

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


def row_tensor(value, rows):
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float8_e4m3fn,
    ):
        return False
    # Projection outputs, activations and the flattened GDN norm are bounded;
    # full vocabulary weights and cache banks are never copied accidentally.
    return value.ndim > 0 and value.shape[0] in (rows, rows * 48) and value.numel() <= rows * 65536


class DispatchCapture(TorchDispatchMode):
    def __init__(self, capture):
        super().__init__()
        self.capture = capture

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if self.capture.busy or name.startswith(
            ("aten.empty", "aten.as_strided", "aten.view", "aten.reshape", "aten.detach")
        ):
            return func(*args, **kwargs)
        return self.capture.invoke(name, func, args, kwargs, schema=func._schema)


class BoundaryCapture:
    def __init__(self, probe, observation, root, *, require_compiled=True):
        self.probe, self.observation, self.root = probe, observation, Path(root)
        self.require_compiled = require_compiled
        self.root.mkdir(mode=0o700)
        self.batch_positions = None
        self.events, self.tensors, self.batches = [], {}, []
        self.counts = Counter()
        self.busy = False
        self.hooks = HookSet()

    def positions(self):
        if self.observation.draft or self.busy:
            return None
        positions = self.probe.pending_positions
        if positions is None:
            return None
        start = len(self.probe.schedule.prefix)
        end = start + len(self.probe.schedule.output) - 1
        if not any(start <= p < end for p in positions):
            return None
        if len(positions) not in (1, 8) or not all(start <= p < end for p in positions):
            raise DiagnosticError("capture requires complete M1/M8 decode groups")
        return tuple(positions)

    def save_tree(self, value, key, rows, saved):
        if row_tensor(value, rows):
            # Float8 storage is saved as bytes; its exact dtype/shape is retained.
            v = value.detach()
            dtype = str(v.dtype)
            if v.dtype == torch.float8_e4m3fn:
                v = v.view(torch.uint8)
            self.tensors[key] = v.to(device="cpu", copy=True).contiguous()
            saved.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": dtype,
                    "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(),
                }
            )
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                self.save_tree(child, f"{key}.{i}", rows, saved)
        elif isinstance(value, dict):
            for name, child in value.items():
                self.save_tree(child, f"{key}.{name}", rows, saved)

    def invoke(self, name, function, args, kwargs, schema=None, signature=None):
        positions = self.positions()
        if positions is None:
            return function(*args, **kwargs)
        if self.batch_positions != positions:
            self.flush()
            self.batch_positions = positions
        index = len(self.events)
        event = {"index": index, "operation": name, "before": [], "after": []}
        self.busy = True
        try:
            # Copies are ordered on the same stream and never alter arguments.
            self.save_tree(args, f"{index}.before.args", len(positions), event["before"])
            self.save_tree(kwargs, f"{index}.before.kwargs", len(positions), event["before"])
            result = function(*args, **kwargs)
            self.save_tree(result, f"{index}.after.result", len(positions), event["after"])
            if schema is not None:
                for i, arg in enumerate(schema.arguments):
                    if arg.alias_info is not None and arg.alias_info.is_write:
                        value = args[i] if i < len(args) else kwargs.get(arg.name)
                        self.save_tree(
                            value,
                            f"{index}.after.mutable.{arg.name}",
                            len(positions),
                            event["after"],
                        )
            if signature is not None:
                for i, key in enumerate(signature):
                    if "out" in str(key) and i < len(args):
                        self.save_tree(
                            args[i], f"{index}.after.{key}", len(positions), event["after"]
                        )
            self.events.append(event)
            self.counts[name] += 1
            return result
        finally:
            self.busy = False

    def attach(self):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        original = CachingAutotuner.run

        @functools.wraps(original)
        def run(kernel, *args, **kwargs):
            names = list(kernel.triton_meta.get("signature", {}))
            if names and isinstance(names[0], int):
                names = list(kernel.fn.arg_names)
            return self.invoke(
                "inductor/" + kernel.fn.__name__,
                functools.partial(original, kernel),
                args,
                kwargs,
                signature=names,
            )

        self.hooks.replace(CachingAutotuner, "run", run)
        owner = self.probe.runner.model
        original_forward = owner.forward

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            with DispatchCapture(self):
                return original_forward(*args, **kwargs)

        self.hooks.replace(owner, "forward", forward)

    def flush(self):
        if self.batch_positions is None:
            return
        if not self.events or not self.tensors:
            raise DiagnosticError("no native boundaries captured for a decode group")
        filename = f"group-{len(self.batches):04d}.pt"
        path = self.root / filename
        torch.save(self.tensors, path)
        with path.open("rb") as source:
            tensor_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        metadata = seal(
            {
                "positions": list(self.batch_positions),
                "events": self.events,
                "tensor_file": filename,
                "tensor_sha256": tensor_sha256,
            }
        )
        write_private(path.with_suffix(".json"), metadata)
        self.batches.append(
            {
                "positions": list(self.batch_positions),
                "sha256": metadata["sha256"],
                "file": filename,
                "events": len(self.events),
            }
        )
        self.events, self.tensors = [], {}
        self.batch_positions = None

    def finish(self):
        self.hooks.close()
        self.flush()
        expected = len(self.probe.schedule.output) - 1
        start = len(self.probe.schedule.prefix)
        observed = [p for b in self.batches for p in b["positions"]]
        if observed != list(range(start, start + expected)):
            raise DiagnosticError("incomplete captured position domain")
        if self.require_compiled and not any(name.startswith("inductor/") for name in self.counts):
            raise DiagnosticError("the compiled launch observer captured no compiled kernels")
        result = seal(
            {
                "schema": (
                    "qwen.compiled-launch-capture.v2"
                    if self.require_compiled
                    else "qwen.eager-launch-capture.v1"
                ),
                "positions": expected,
                "counts": dict(self.counts),
                "batches": self.batches,
                "numerical_reference_status": "UNQUALIFIED until release-output bridge passes",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        write_private(self.root / "manifest.json", result)
        return {"sha256": result["sha256"], "positions": expected, "operations": len(self.counts)}


class CaptureProbe(CompiledEquivalenceProbe):
    @staticmethod
    def validate_execution(runner):
        config = runner.vllm_config
        if runner.model_config.enforce_eager or not int(config.compilation_config.mode):
            raise DiagnosticError("the capture bridge must use compiled execution")
        if str(config.compilation_config.cudagraph_mode).split(".")[-1] != "NONE":
            raise DiagnosticError("host tensor capture cannot occur inside GPU graph replay")
        if config.scheduler_config.async_scheduling:
            raise DiagnosticError("isolated capture requires an ordered diagnostic worker")


class IsolatedCaptureWorker(OptimizedWorker):
    def qwen_optimized_begin(self, private, profile=False, task_path=None):
        if profile or task_path is None:
            raise DiagnosticError("capture requires a forced correctness request, not timing")
        if hasattr(self, "_qwen_observation"):
            raise DiagnosticError("previous capture is active")
        self._qwen_observation = GraphObservation(self.model_runner, private, False)
        self._qwen_forced = CaptureProbe(self.model_runner, private_json(Path(task_path)))
        self._qwen_forced.attach()
        self._isolated_capture = BoundaryCapture(
            self._qwen_forced, self._qwen_observation, Path(private) / "isolated-boundaries"
        )
        self._isolated_capture.attach()
        return {"installed": True, "release_output_bridge": "UNQUALIFIED"}

    def qwen_optimized_finish(self):
        try:
            capture = self._isolated_capture.finish()
            forced = self._qwen_forced.finish()
        finally:
            self._isolated_capture.hooks.close()
            self._qwen_forced.close()
            observation = self._qwen_observation.close()
            del self._qwen_forced, self._qwen_observation
        if observation["counts"].get("target_graph_replays", 0):
            raise DiagnosticError("unexpected graph replay during host tensor capture")
        return {
            "observation": observation,
            "forced": forced,
            "isolated_capture": capture,
            "qualification": "UNQUALIFIED until release-output bridge passes",
            "performance": self._qwen_performance_repairs.receipt()
            if self._qwen_performance_repairs
            else None,
        }
