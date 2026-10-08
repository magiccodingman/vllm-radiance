"""Capture actual compiled call interfaces after compilation, without tensor data."""

import functools
import hashlib
from pathlib import Path

import torch
from execution_mode_d7_worker import ExecutionModeWorker
from torch.utils._python_dispatch import TorchDispatchMode

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


class CatalogDispatch(TorchDispatchMode):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        return self.catalog.invoke(str(function), function, args, kwargs or {})


class Catalog:
    def __init__(self, runner, probe, observation, root):
        self.probe, self.observation = probe, observation
        self.root = Path(root)
        self.hooks = HookSet()
        self.active = False
        self.busy = False
        self.events = []
        self.parameters = {}
        self.refs = {}
        self.next_ref = 0
        self.done = False
        self.rows = 0
        for name, value in [*runner.model.named_parameters(), *runner.model.named_buffers()]:
            pointer = value.untyped_storage().data_ptr()
            if pointer:
                self.parameters.setdefault(pointer, []).append(name)
        self.model = runner.model

    def describe(self, value):
        import weakref

        if isinstance(value, torch.Tensor):
            old = self.refs.get(id(value))
            if old is None or old[0]() is not value:
                self.refs[id(value)] = (weakref.ref(value), self.next_ref)
                self.next_ref += 1
            return {
                "tensor": self.refs[id(value)][1],
                "shape": list(value.shape),
                "strides": list(value.stride()),
                "dtype": str(value.dtype),
                "parameters": self.parameters.get(value.untyped_storage().data_ptr(), []),
            }
        if isinstance(value, (list, tuple)):
            return [self.describe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.describe(v) for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__name__}

    def invoke(self, name, function, args, kwargs, kernel=None):
        if not self.active or self.busy:
            return function(*args, **kwargs)
        self.busy = True
        try:
            record = {
                "index": len(self.events),
                "operation": name,
                "args": self.describe(args),
                "kwargs": self.describe(kwargs),
            }
            if kernel is not None:
                fn = kernel.fn.fn
                path = Path(fn.__code__.co_filename)
                record["compiled"] = {
                    "source": str(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "name": kernel.fn.__name__,
                    "arg_names": kernel.fn.arg_names,
                }
            result = function(*args, **kwargs)
            record["result"] = self.describe(result)
            self.events.append(record)
            return result
        finally:
            self.busy = False

    def attach(self):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        original = CachingAutotuner.run

        @functools.wraps(original)
        def run(kernel, *args, **kwargs):
            return self.invoke(
                "inductor/" + kernel.fn.__name__,
                functools.partial(original, kernel),
                args,
                kwargs,
                kernel,
            )

        self.hooks.replace(CachingAutotuner, "run", run)
        forward = self.model.forward

        @functools.wraps(forward)
        def wrapped(*args, **kwargs):
            positions = self.probe.pending_positions
            start = len(self.probe.schedule.prefix)
            if self.done or self.observation.draft or not positions or positions[0] != start:
                return forward(*args, **kwargs)
            self.active = True
            self.rows = len(positions)
            try:
                with CatalogDispatch(self):
                    result = forward(*args, **kwargs)
            finally:
                self.active = False
            self.done = True
            write_private(
                self.root / "native-call-catalog.json",
                seal(
                    {
                        "schema": "qwen.native-call-catalog.v1",
                        "rows": self.rows,
                        "events": self.events,
                        "scope": "Interfaces and actual compiled source, no tensor contents",
                    }
                ),
            )
            return result

        self.hooks.replace(self.model, "forward", wrapped)


class CatalogWorker(ExecutionModeWorker):
    def qwen_optimized_begin(self, private, profile=False, task_path=None):
        result = super().qwen_optimized_begin(private, profile, task_path)
        self._catalog = Catalog(
            self.model_runner, self._qwen_forced, self._qwen_observation, private
        )
        self._catalog.attach()
        return result

    def qwen_optimized_finish(self):
        self._catalog.hooks.close()
        if not self._catalog.done:
            raise DiagnosticError("compiled catalog was not observed")
        result = super().qwen_optimized_finish()
        result["catalog"] = {"rows": self._catalog.rows, "events": len(self._catalog.events)}
        return result
