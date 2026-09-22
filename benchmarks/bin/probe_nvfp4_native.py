#!/usr/bin/env python3
"""Small installed-loader/native conversion gate, not model quality qualification."""
import argparse
import hashlib
import json
import resource
import tempfile
from pathlib import Path

import torch
import radiance_nvfp4 as nv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(a.device)
    from vllm.distributed import (init_distributed_environment,
        initialize_model_parallel, destroy_model_parallel,
        destroy_distributed_environment)
    rendezvous = tempfile.TemporaryDirectory(prefix="nvfp4-native-")
    init_distributed_environment(world_size=1, rank=0, local_rank=a.device,
        distributed_init_method="file://" + rendezvous.name + "/group")
    initialize_model_parallel()
    torch.manual_seed(170)
    cls = nv.scheme_class()
    scheme = cls(use_a16=False, source_id="synthetic-nvfp4-merged-v1",
                 metadata={"format": "nvfp4-pack-quantized", "group_size": 16})
    layer = torch.nn.Module()
    with torch.device("cuda"):
        scheme.create_weights(layer, [128, 128, 128], 256, torch.bfloat16,
                              lambda *args, **kwargs: None)
    with torch.no_grad():
        layer.weight_packed.copy_(torch.randint(0, 256, layer.weight_packed.shape,
                                                device="cuda", dtype=torch.uint8))
        layer.weight_scale.copy_(torch.rand(layer.weight_scale.shape, device="cuda") + 0.25)
        layer.weight_global_scale.copy_(torch.tensor([1., 2., 4.], device="cuda"))
        layer.input_global_scale.fill_(1)
    original = layer.weight_packed.detach().clone()
    source_scales = layer.weight_scale.detach().clone()
    # Prove repack failure cannot install a half-converted parameter dictionary.
    original_process = scheme.kernel.process_weights_after_loading
    def reject(staged):
        staged.weight = torch.nn.Parameter(torch.zeros_like(staged.weight), requires_grad=False)
        raise RuntimeError("deliberate repack fault")
    scheme.kernel.process_weights_after_loading = reject
    try:
        scheme.process_weights_after_loading(layer)
        raise AssertionError("injected failure missing")
    except RuntimeError as exc:
        assert str(exc) == "deliberate repack fault"
    assert torch.equal(layer.weight_packed, original) and not hasattr(layer, "weight")
    scheme.kernel.process_weights_after_loading = original_process
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    scheme.process_weights_after_loading(layer)
    peak = torch.cuda.max_memory_allocated()
    reference = torch.cat([nv.dequant_nvfp4(original[i*128:(i+1)*128].cpu(),
                          source_scales[i*128:(i+1)*128].cpu(), 2**i) for i in range(3)])
    converted_p, converted_s, _ = nv.convert(original, source_scales,
        torch.tensor([1.,2.,4.]), [128,128,128], source_id="fixture", metadata={})
    reference_mx = nv.dequant_mxfp4(converted_p.cpu(), converted_s.cpu()).to("cuda", torch.bfloat16)
    conversion_rel = float((reference_mx.cpu().float()-reference).norm()/reference.norm())
    assert conversion_rel < 0.3
    cases = []
    for m in (1, 2, 17, 65):
        x = torch.randn((m,256), device="cuda", dtype=torch.bfloat16)
        y = scheme.apply_weights(layer, x)
        repeat = scheme.apply_weights(layer, x)
        changed = scheme.apply_weights(layer, x * 0.5)
        assert torch.equal(y, repeat) and torch.isfinite(y).all() and not torch.equal(y, changed)
        ref = x.float() @ reference_mx.float().T
        rel = float((y.float()-ref).norm()/ref.norm())
        assert rel < 0.15, (m, rel)  # W4A8 activation quantization, not byte parity to BF16.
        cases.append({"m": m, "relative_error_to_converted_bf16": rel,
                      "sha256": hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()})
    # Selected donor delta: the TP1 gate/up width exceeds the old32768 limit.
    wide = torch.nn.Module()
    wide.weight = torch.nn.Parameter(torch.randint(0,256,(34816,128), device="cuda",
                                                   dtype=torch.uint8), requires_grad=False)
    wide.weight_scale = torch.nn.Parameter(torch.full((34816,8),127, device="cuda",
                                                     dtype=torch.uint8), requires_grad=False)
    wide_ref = nv.dequant_mxfp4(wide.weight.detach(), wide.weight_scale.detach())
    scheme.kernel.process_weights_after_loading(wide)
    assert wide.radiance_w4a8_ok
    xwide = torch.randn((1,256), device="cuda", dtype=torch.bfloat16)
    ywide = scheme.kernel.apply_weights(wide, xwide)
    refwide = xwide.float() @ wide_ref.T
    wide_error = float((ywide.float()-refwide).norm()/refwide.norm())
    assert ywide.shape == (1,34816) and torch.isfinite(ywide).all() and wide_error < 0.15
    result = {"status": "NATIVE_FIXTURE_PASS_NOT_MODEL_QUALITY", "device": a.device,
              "arch": torch.cuda.get_device_properties(a.device).gcnArchName,
              "receipt": layer._radiance_nvfp4_receipt, "cases": cases,
              "reconstruction_relative_error": conversion_rel,
              "conversion_gpu_peak_increment": peak-before,
              "host_peak_rss_increment": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss_before,
              "rollback": "PASS", "kernel": type(scheme.kernel).__qualname__}
    result["tp1_wide_decode"] = {"shape": [1,34816,256], "relative_error": wide_error}
    (a.out / "result.json").write_text(json.dumps(result, indent=2))
    destroy_model_parallel()
    destroy_distributed_environment()
    rendezvous.cleanup()


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        main()
