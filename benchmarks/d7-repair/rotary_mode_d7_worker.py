"""Isolated eager M8 worker with explicit BF16 RoPE product rounding."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

from execution_mode_d7_worker import ExecutionModeCaptureWorker, ExecutionModeWorker

from qwen_r9700_lab.conformance_rotary_repair import SOURCE_SHA256, patch_source
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import seal, write_private


class RotaryRneMixin:
    def load_model(self, **kwargs):
        import vllm.model_executor.layers.rotary_embedding.mrope as native

        require(self.model_config.enforce_eager, "rotary intervention requires the eager arm")
        root = Path(os.environ["QWEN_OPTIMIZED_STARTUP_RECEIPT"]).parent
        destination = root / "rotary-rne-module.py"
        destination.write_text(patch_source(Path(native.__file__).read_text()))
        name = "vllm.model_executor.layers.rotary_embedding._diagnostic_rne_mrope"
        spec = importlib.util.spec_from_file_location(name, destination)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self._rotary_module = module
        self._rotary_calls = 0
        original_run = module._triton_mrope_forward.run

        def observe(*args, **kw):
            self._rotary_calls += 1
            return original_run(*args, **kw)

        module._triton_mrope_forward.run = observe
        native._triton_mrope_forward = module._triton_mrope_forward
        self._rotary_identity = seal(
            {
                "schema": "qwen.rotary-rne-worker.v1",
                "native_source": SOURCE_SHA256,
                "patched_source": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "worker_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "patcher_source": hashlib.sha256(
                    Path(sys.modules[patch_source.__module__].__file__).read_bytes()
                ).hexdigest(),
                "installed_before_load": True,
                "scope": "Eager native MRoPE product rounding only; no installed files changed.",
            }
        )
        write_private(root / "rotary-intervention.json", self._rotary_identity)
        return super().load_model(**kwargs)

    def rotary_observation(self):
        import vllm.model_executor.layers.rotary_embedding.mrope as native

        require(
            native._triton_mrope_forward is self._rotary_module._triton_mrope_forward,
            "native rotary binding changed during replay",
        )
        return {"identity": self._rotary_identity, "calls": self._rotary_calls}

    def qwen_optimized_metadata(self):
        result = super().qwen_optimized_metadata()
        result["rotary_intervention"] = self.rotary_observation()
        return result

    def qwen_optimized_finish(self):
        result = super().qwen_optimized_finish()
        observation = self.rotary_observation()
        require(observation["calls"] > 0, "no modified rotary kernel executed")
        result["rotary_intervention"] = observation
        return result


class RotaryRneWorker(RotaryRneMixin, ExecutionModeWorker):
    pass


class RotaryRneCaptureWorker(RotaryRneMixin, ExecutionModeCaptureWorker):
    pass
