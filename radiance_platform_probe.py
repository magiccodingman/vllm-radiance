"""Opt-in worker-extension inspection for resident-platform qualification.

No hooks or model replacements. RPCs run only on explicit qualification calls.
The dev RPC endpoint must be bound to localhost and disabled in distribution use.
"""
import inspect
import hashlib
from pathlib import Path


def identity(value):
    cls = value if isinstance(value, type) else type(value)
    path = inspect.getsourcefile(cls)
    return {"class": cls.__qualname__, "module": cls.__module__,
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if path and Path(path).is_file() else None}


class PlatformProbe:
    def platform_snapshot(self):
        import torch
        runner = self.model_runner
        if type(runner).__module__ != "vllm.v1.worker.gpu.model_runner":
            raise RuntimeError(f"Expected Runner V2, got {type(runner)}")
        model = runner.get_model()
        speculator = getattr(runner, "speculator", None)
        spec_config = self.vllm_config.speculative_config
        if spec_config is not None and speculator is None:
            raise RuntimeError("Configured resident speculator was not instantiated")
        def inventory(root):
            return self._platform_inventory(root)
        owners = inventory(model)
        draft_model = getattr(speculator, "model", None)
        return {"runner": identity(runner), "model": identity(model),
                "speculator": identity(speculator) if speculator is not None else None,
                "draft_model": identity(draft_model) if draft_model is not None else None,
                "draft_owners": inventory(draft_model) if draft_model is not None else [],
                "speculative_method": spec_config.method if spec_config is not None else None,
                "num_speculative_steps": getattr(runner, "num_speculative_steps", 0),
                "rank": self.rank, "owners": owners,
                "graph": str(self.vllm_config.compilation_config.cudagraph_mode),
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "free_total": torch.cuda.mem_get_info()}

    def _platform_inventory(self, model):
        owners = []
        for name, module in model.named_modules():
            row = {"name": name, **identity(module)}
            if hasattr(module, "radiance_w4a8_ok"):
                row["radiance_w4a8_ok"] = module.radiance_w4a8_ok
                if not module.radiance_w4a8_ok:
                    raise RuntimeError(f"Required resident W4A8 fell back: {name}")
            for attr in ("quant_method", "kernel", "attn_backend", "impl"):
                value = getattr(module, attr, None)
                if value is not None:
                    row[attr] = identity(value)
                    for nested in ("kernel", "scheme", "quant_method"):
                        obj = getattr(value, nested, None)
                        if obj is not None:
                            row[attr][nested] = identity(obj)
            if len(row) > 4 or "GatedDelta" in row["class"]:
                owners.append(row)
        return owners

    def platform_profile_start(self):
        import torch
        if getattr(self, "_platform_profiler", None) is not None:
            raise RuntimeError("platform profiler already active")
        self._platform_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False, profile_memory=False,
            with_flops=False)
        self._platform_profiler.start()
        return {"rank": self.rank, "started": True}

    def platform_profile_stop(self):
        import json
        profiler = self._platform_profiler
        if profiler is None:
            raise RuntimeError("platform profiler not active")
        profiler.stop()
        self._platform_profiler = None
        rows = [{"name": e.key, "count": e.count,
                 "cpu_us": e.cpu_time_total,
                 "device_us": e.device_time_total} for e in profiler.key_averages()]
        path = Path(f"/evidence/platform-kernels.rank{self.rank}.json")
        path.write_text(json.dumps(rows, indent=2))
        return {"rank": self.rank, "path": str(path), "operations": len(rows)}
