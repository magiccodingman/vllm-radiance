# vLLM 0.33 installed native checks and LFM3 corrections

The base image/revision is recorded in `manifest.json`. Every GPU container
exposed only `/dev/kfd` and `/dev/dri/renderD128`, with HIP/ROCR device0 masks.
The preflight asserted one gfx1201 R9700. No production container was changed.

The installed image passed 12 paged-attention cases, both 12-case GDN head
geometries, independent DFlash noise and 138 verify-head tests. These are
synthetic single-GPU contracts, not distributed TP2 or speculative equivalence.

The first unchanged upstream LFM3 suite reported **78 failed / 248 passed**.
The importer incorrectly treated a matching forward fragment as proof that a
decode hunk was already present. Function-scoped context and independent whole
kernel fingerprints repair the missing width7 arithmetic/state branches.
The original upstream patch and tests remain byte-identical. The resulting
unadapted convolution source matches independent `git apply --no-index`.

That repair passed 324/326 cases. The remaining two width7 prefill comparisons
also fail with stock block256, so the wider Radiance tile is not their cause.
Independent FP64 calculations show native relative L2 error about0.00357.
Explicit FP32 width7 products reduce it to about0.00166 and satisfy the same
upstream assertions. Established width4 arithmetic remains unchanged. With the
complete correction and repeated installation, all **326 upstream tests pass**.
Original failures, stock-tiling control, mathematical controls and final focused
pass are preserved, including raw unmodified logs.

The initial NVFP4/upstream-native invocations failed argument parsing because
the harness supplied `cuda:0` to an integer device option. A subsequent NVFP4
probe omitted the required optional W4A8 flag and was stopped after its explicit
setup exception hung distributed teardown. Those checks are still pending with
corrected harness setup; no numerical pass is inferred.

The LFM repairs were applied inside disposable copies of the base image. Final
rebuilt-image checks, real Quark/LFM FP8 serving, tools and streaming remain
pending. No model-quality, capacity, performance, TP2 or release claim follows.
