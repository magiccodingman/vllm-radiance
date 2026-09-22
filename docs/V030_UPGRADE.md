# vLLM 0.30 resident platform migration

Status: **IN PROGRESS — NOT HARDWARE QUALIFIED**.

## Source and scope

This branch starts from main `285ac78e7f19bc1e1b5b09b25f865aea0c6d9754`.
MR !37 remains Draft/open/unmerged at research report
`b5208d32b8c024181a182b7e8fbdf5dff368bcf2`; its freeze note is
[note 2990](https://gitlab.sayou.io/lance-wright/vllm-radiance/-/merge_requests/37#note_2990).
Its behavior tests and hardware history are reference evidence, not the upgrade base.

Target: vLLM 0.30.0, immutable commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
GGZ14 donor frozen at `31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`
from https://github.com/GGZ14/vllm-mxfp4 on 2026-09-22.

Production remains stopped under the owner's standing instruction. This work
uses separate candidate images and isolated resident-model validation. No merge,
tiered-v2 implementation, device-managed pool or publication benchmark is included.

## Release semantics and expected impact

The [v0.30 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
changes Runner V2 graph capture and speculative execution, online/partial
quantization, parser/serving dependencies, and Qwen QSA/PLE execution.
It removes deprecated environment variables, deprecates Mamba cache `all`, and
changes scale-out endpoint configuration. NVIDIA-specific compile removal is
not a ROCm policy. Runner V2 selection and graph mode require independent checks.

Migration areas: runner/sampling, GDN metadata and prefix state, FP8/MXFP4
dispatch, attention, TP collectives, speculative hooks, parser/XGrammar,
CPU KV transfer and graph capture. Every registered overlay must receive an
evidence-backed disposition in the existing maintenance registry.

## Stack comparison

The target's `docker/Dockerfile.rocm_base` pins the same AMD Torch
`6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`, Triton
`f0b55c07da61c71775bef6d1a15ebf846430ac75`, and torchvision 0.27.1
lineages as Radiance. Preserve these unless an actual compatibility issue requires
a change. Upstream uses ROCm 7.2.3 packaging and AITER `v0.1.21.post2`;
Radiance starts from qualified ROCm 7.14 and AITER 0.1.20. AITER API comparison
and native qualification are pending; no whole-stack replacement is assumed.

## Qualification ledger

Build/import, overlay dispositions, Runner V2 path evidence, resident FP8 and
Quark MXFP4 gates, NVFP4 conversion, parser/structured output, vision and bounded
performance sanity are pending. No historical test count is attributed to this
candidate. The existing full source/contract suite remains required; publication
BetterBench and long-context qualification are excluded by this mission.
