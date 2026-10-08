# Current-stack verify-head contracts

Radiance 1.0.387 / vLLM 0.30 / PyTorch 2.12 / ROCm 7.14 on R9700.
Baseline PR checks pass. CPU dispatch tests: 117 passed, 19 native opt-in skips.
Native suite: 136 passed, zero skips on GPU 0.

This covers synthetic selection, dispatch, full-head fallbacks and a TP2 eligibility guard; it does not claim distributed TP2 numerical execution, complete candidate recall, serving throughput or end-to-end output qualification. Historical benchmark receipts remain separate.

## Deeper qualification

The public hook now enforces both documented opt-in flags before packing or rebinding the target head. The updated full candidate image passes 138 CPU/native tests with no skips.

A public 1024-by-512 BF16 negative control gives every row an identical INT2 projection but makes an excluded row the unique full-head winner. Global-256 drops that winner (reference token 256, approximate token 0). See `global-256-negative.json`. This confirms that the TP1 global path is experimental; passing the capacity fixtures is not an exactness certificate. The TP2 guard uses the full head.

The combined PR8–11 candidate also failed the strict whole-model output-equivalence gate on 7 of 8 fixed prompts. The GDN investigation proved that the old scan caused the differences: replacing only the scan restored all eight main responses, while the candidate exactly matched all eight responses from an independent FP64 corrected reference. The saved TP2 DFlash-7 production profile passed its tool and capacity gates through 131K. These receipts validate the TP2 full-head guard, not the approximate TP1 path. See the [full qualification capsule](https://github.com/Terrydaktal/vllm-radiance/blob/c0abf245573ccff5ce96d6fdeeed4611cffab555/benchmarks/results/20261008-pr8-11-production-qualification/README.md).
