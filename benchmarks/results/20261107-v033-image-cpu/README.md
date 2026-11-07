# vLLM 0.33 candidate-image CPU evidence

The first full image built from `2c8b2336f667fc577650ca73c08faafce78a0e08`
passed its final native imports, component versions, dependency check and
HIP/pybind11 compile-link-import gate. Its immutable image ID and installed
versions are recorded in `manifest.json` and `versions.json`.

The original CPU run passed **15 of 19** registered PR/candidate-image checks.
Every original log, including the four failures, is retained under `initial/`.
No GPU device was exposed. This does not establish native GPU or model behavior.

The failures found:

- KV restore's installed test still imported a retired annotation helper. Its
  replacement now exercises actual global planning, spec role metadata and
  native EAGLE/MTP versus DFlash/DSpark policy through the builder seam.
- The optional Qwen grammar mode preserved function envelopes but lost the
  omitted-strict parameter-schema default. The runtime correction restores
  the legacy opt-in without overriding explicit `strict=false`.
- Speculative grammar tests needed the new constraint-bounds and four-field
  bitmask interfaces; termination/padding/rollback assertions remain.
- XGrammar removed a private testing converter. The open-object check now uses
  public compiler/matcher APIs and accepts nested JSON while rejecting nested
  XML, wrong types and incomplete objects.

`focused-repairs/` retains the successful grammar/termination and open-object
reruns plus nine PR CPU checks. The grammar runtime correction was applied
only inside a disposable test container. Final corrected-image CPU/native/model
qualification remains pending. No production deployment or TP2/performance
claim follows from this evidence.
