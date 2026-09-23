"""Explicit faults in owned native replay workers; never imported by production.

Faults alter the actual selected device allocation before the normal extractor
reads it. A receipt records application separately from detection. A failed
startup can therefore never count as a successful corruption negative control.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, write_private

FAULTS = ("kv", "gdn", "conv", "pending", "position", "version", "missing_observation")


def install_experiment(probe):
    path = os.environ.get("QWEN_CONFORMANCE_NATIVE_EXPERIMENT")
    if not path:
        return
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("native fault experiment is not armed")
    spec = private_json(Path(path))
    if set(spec) - {"fault", "step", "rejected_suffix_token"}:
        raise DiagnosticError("unsupported native fault configuration")
    fault, step = spec.get("fault"), spec.get("step", 0)
    if fault is not None and fault not in FAULTS:
        raise DiagnosticError("unknown native corruption control")
    if type(step) is not int or step < 0 or step >= len(probe.campaign.expected):
        raise DiagnosticError("fault target is outside the replay schedule")
    suffix = spec.get("rejected_suffix_token")
    if suffix is not None:
        if type(suffix) is not int or not 0 <= suffix < probe.config["vocab_size"]:
            raise DiagnosticError("invalid rejected-suffix token")
        probe.rejected_suffix_token = suffix
    if fault is None:
        return
    original = probe.capture
    applied = False

    def capture(batch, expected, logits, **kwargs):
        nonlocal applied
        if not applied and probe.index == step:
            component = mutate_device(probe, batch, fault)
            applied = True
            write_private(
                probe.campaign.root / "fault-applied.json",
                {
                    "fault": fault,
                    "step": step,
                    "component": component,
                    "consumed": expected["consumed"],
                    "actual_device_allocation": fault != "missing_observation",
                },
            )
            if fault == "missing_observation":
                return
        return original(batch, expected, logits, **kwargs)

    probe.hooks.replace(probe, "capture", capture)


def mutate_device(probe, batch, fault):
    if os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        raise DiagnosticError("native device mutation is not armed")
    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    from qwen_r9700_lab.conformance_radiance import convolution_offset, temporal_column

    torch.cuda.synchronize()
    runner, c = probe.runner, probe.config
    index = int(batch.idx_mapping_np[0])
    if fault == "missing_observation":
        return "observation.frame"
    if fault == "position":
        runner.req_states.num_computed_tokens.gpu[index] += 1
        return "sequence.position"
    if fault == "pending":
        value = runner.req_states.last_sampled_tokens[index, 0]
        value.copy_((value + 1) % c["vocab_size"])
        return "sequence.pending"
    if fault == "version":
        # Invalid version, rather than a version that happens to contain equal data.
        runner.model_state._mamba_state_idx_gpu[index] = -1
        return "gdn.version"
    targets = {id(m) for _, m in runner.model.named_modules()}
    context = runner.compilation_config.static_forward_context
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        table = runner.block_tables.block_tables[gid].gpu[index]
        for name in group.layer_names:
            module = context[name]
            if id(module) not in targets:
                continue
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None:
                raise DiagnosticError("unmapped fault target")
            layer = int(match.group(1))
            linear = c["layer_types"][layer] == "linear_attention"
            if fault == "kv" and not linear:
                # First logically consumed key byte. Do not reshape a
                # noncontiguous view: reshape could mutate a disposable copy.
                selected = module.kv_cache[int(table[0]), 0, 0, 0:1]
            elif fault in {"gdn", "conv"} and linear:
                running = int(runner.model_state._mamba_state_idx_gpu[index])
                accepted = int(runner.model_state.num_accepted_tokens_gpu[index])
                conv, state = module.kv_cache
                if fault == "gdn":
                    selected = state[int(table[temporal_column(running, accepted)])]
                    selected = selected[(0,) * (selected.ndim - 1) + (slice(0, 1),)]
                else:
                    offset = convolution_offset(accepted)
                    selected = conv[int(table[running])]
                    selected = (
                        selected[0, offset : offset + 1]
                        if is_conv_state_dim_first()
                        else selected[offset, 0:1]
                    )
            else:
                continue
            if selected.numel() != 1 or not selected.is_contiguous():
                raise DiagnosticError("native fault target is not an actual scalar view")
            selected.view(torch.uint8)[0].bitwise_xor_(1)
            torch.cuda.synchronize()
            return f"layer.{layer:03d}.{'keys' if fault == 'kv' else fault}"
    raise DiagnosticError("required native corruption target was not found")
