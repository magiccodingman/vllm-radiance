# vLLM 0.34 upgrade qualification

The candidate uses vLLM 0.34.0 at `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`
and Transformers 5.18.0. The original Transformers 5.16.1 pin conflicted with
vLLM's Hugging Face Hub 2.x requirement. The candidate retains ROCm 10.0,
Torch 2.13.0, Triton 3.8.0, and AITER 0.1.26.

## Candidate evidence

The image for Radiance commit `c35559b9288a98875571301e7cab17c19203dd03`
built locally as
`sha256:69b398b8d64973888d1a153606b52ff38ed3097bee196cff465e5909bf61de41`.
The native vLLM extension imported successfully.

- Native LFM3 FP8 and BF16 on one R9700 passed model ownership, streaming
  token parity, repeated long prompts, post-batch state isolation, nested JSON
  schema, and named tool round trips. Both failed strict batch token equivalence.
  The FP8 control on 0.33 produced the same two batch continuations in reversed
  order. The BF16 control on 0.33 also failed only the strict batch token
  equivalence check. Both candidate checkpoint configurations passed the
  installed registry, width-7 convolution configuration, and RoPE checks.
- The upstream LFM3 native suites first returned 327 passes and two failures
  on long BF16 width-7 convolution. The failures reproduced with the stock
  channel tile. Restoring the existing width-7 FP32 products correction in
  the convolution overlay brought the suites to 329 passes on the candidate
  with the corrected source mounted. The image rebuild is pending.
- Qwen 27B Quark MXFP4 with FP8 DFlash-7 and R4D attention on dual R9700s
  passed FP8 drafter ownership, streaming token parity, and a named tool call.
  The drafter proposed 149 tokens and the target accepted 80. This smoke test
  covers the tested configuration and carries no throughput claim.
- All eight draft-confidence kernel checks passed on each R9700, including
  HIP graph replay and fully masked rows.
- The V1 MTP loop installer was retired from image assembly. The V2 port of
  Radiance's confidence and n-gram controller remains open.

The upgrade remains in draft while qualification is open.
