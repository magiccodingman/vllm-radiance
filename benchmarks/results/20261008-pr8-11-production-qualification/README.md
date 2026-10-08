# PR8–11 qualification on dual R9700, 2026-10-08

The GDN repair is validated against an independent FP64 defining recurrence, including whole-model inference. The historical main-branch comparison **failed** and is retained: 7/8 responses changed because the old scan corrupted real prefill state. This is not a claim that all outputs stayed unchanged.

## Identity and scope

- Main control: `c09ee5db08226895e924983fe2f1495e34ebc758`.
- Integrated candidate: `010bbad79255a7188a4c327398d3c814621448c0`; PR8 `e578d070`, PR9 runtime `e1dcf1c`, PR10 `d97fe11e`, PR11 `1015925c`. Later commits add evidence/docs only.
- Supported `Dockerfile.patch` rebuild over `magiccodingman/vllm-radiance:1.0.387`; full libr4d and MXFP4 extensions rebuilt for gfx1201. Existing pinned Torch/vLLM foundation retained.
- vLLM 0.30.0, ROCm 7.14, Torch 2.12.0, Triton 3.7.1; two AMD Radeon AI PRO R9700 cards.
- Qwen3.8-27B Quark AWQ MXFP4 target. Saved production environment enables DFlash-7 with the tcclaviger FP8 drafter, TP2, 131072 context and 28 GiB native KV offload. The effective engine mode was verified in server logs; client `spec=off` labels did not disable that server-side mode.

## GDN causality and independent reference

1. Two main controls, including an explicit full target head, produce identical responses on all eight fixed prompts.
2. Corrected inference repeats all eight responses exactly, including with the diagnostic observer installed.
3. Real prefill decay spans reach approximately 375; this is not only a synthetic extreme-input issue.
4. In the same candidate process, switching only the scan function to the old library restores **all eight** main responses. The old scan's maximum measured output/state relative errors against the defining recurrence reach **33.8% / 30.3%** on those real inputs. The corrected maximum errors are **0.332% / 0.283%**.
5. An independent CPU FP64 sequential recurrence supplies output/state for affected sequence/head pairs. Unaffected pairs retain their original fast output. **All eight complete responses and output lengths exactly match the published candidate run**, using the unchanged `verify_outputs.py` gate and original prompts/seeds. No mismatching prompt was discarded.

The scripts in this evidence capsule are opt-in diagnostic worker extensions, not serving defaults or a new supported benchmark framework. They retain aggregate measurements, not hidden tensors or conversations. Bind the dev RPC endpoint to localhost. The old-scan comparison loads the exact control's library as `/probe/control-r4d.so`; reference generation independently computes the recurrence from actual q/k/v/g/beta and initial state, without using native output as its answer.

The historical-main failure remains `historical-main-equivalence.json`. The passing mathematical-reference comparison is `independent-reference-equivalence.json`. These represent different controls and must not be conflated.

## Other gates

- Automatic CPU release checks and candidate-image parser, xgrammar, open-object, FP8 calibration and TunableOp contracts: pass.
- Verify-head CPU/native suite: **138 passed, no skips**. Both documented opt-in flags are now enforced.
- Native GDN: **18 cases per GPU**, including the repair boundary, mixed heads/sequences, output guards and three changed-input graph replays. All corrected cases pass the existing 1% limit. Old-library defect reproduced. Below-threshold errors match the control exactly.
- Installed upstream independent-noise sampler: 200000 draws per arm across three positions; corrected distribution and greedy checks pass, biased shared-noise negative control reproduces.
- Conformance/D7 CPU suites in declared locked Torch 2.14 CPU environment: **428 passed, one skip** for an absent retained compiler artifact. Their old-stack native adapters are not claimed qualified on vLLM 0.30.
- Standard BetterBench: ten passes in each of eight categories, 24 requests each at c1/c2/c4/c8, and three prefill targets. All measured requests succeed.
- Matched full-head weighted decode: control **50.303**, candidate **50.256 tok/s** (−0.094%, one comparison, no speedup claim). Largest prefill target: about **4331 vs 3984 prompt tok/s**; the correction has a measured prefill cost.
- Saved DFlash-7 production profile: smoke, **30/30** tool-schema and **40/40** streaming/non-streaming open-object requests pass.
- Capacity waves all complete without failed requests: **8K×8, 16K×7, 32K×5, 64K×3, 131K×1**. This does not certify eight concurrent 131K requests.

## Promotion boundaries

PR9's TP1 global shortlist is approximate. A public 1024×512 negative control drops the true winning token even at global-256. It remains experimental; TP2 uses the full-head guard. PR10/11 add research/qualification tools, not a deployed v0.30 D7 arithmetic repair. No production restart, PR merge or image publication was performed as part of these tests.
