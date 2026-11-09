# Bounded native LFM3 FP8 serving qualification

Exact candidate image `sha256:b2c1f2fe8e3f323e90b017890843ef614a6cfc5e50498a8eb6e242ea11e60f83`, source `451d1eb9acc722b76ae2039e3fb5d8af309ff6fe`.
GPU0 only, TP1/V2, native compressed-tensors FP8 checkpoint, BF16 compute/KV, Triton attention, APC off, 4K/2seq, no speculation.

Seven wire gates pass; exact single-versus-batch greedy token equivalence **fails**. Original requests, responses, SSE and logprobs are retained without edits. Both duplicate batch members agree and a subsequent single request reproduces its original trajectory. No exact batch-invariance claim or tolerance waiver is made.

The attempted upstream batch-invariant diagnostic rejects SHORT_CONV_ATTN at startup. That negative control is retained. Forced named tools intentionally return finish_reason=stop under native vLLM resolve_finish_reason; the harness was corrected after consulting that implementation, retaining tool-name/arguments/SSE/result checks.

The broader final-image CPU/native evidence is under the adjacent image-cpu/final-b2 and installed-native/final-b2 directories. In particular all326 unmodified upstream convolution tests pass.

This is optional model support, not release, TP2, performance or capacity qualification. See docs/LFM3.md.
