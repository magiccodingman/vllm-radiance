# vllm-radiance: from-scratch vLLM + ROCm image for dual Radeon AI PRO R9700 (gfx1201 / RDNA4).
# Installs prebuilt vLLM/torch/ROCm wheels, builds aiter from source for gfx1201, applies
# RDNA4 (gfx1201) fix patches, and bakes the radiance kernels.
FROM ubuntu:24.04

ARG GFX_ARCH=gfx1201
# NOTE: VLLM_VERSION and ROCM_SDK_VERSION resolve from wheel indexes that keep only their newest
# builds; an exact pin stops resolving within days. Both MUST still be listed at build time. Bump
# to a current version first (see README "Building"):
#   vLLM:     curl -sL https://wheels.vllm.ai/rocm/vllm/ | grep -oiE 'vllm-[0-9][^"<> ]*'
#   ROCm SDK: curl -sL https://rocm.nightlies.amd.com/whl-multi-arch/rocm-sdk-libraries/ | grep -oE '7\.[0-9.]+a[0-9]+'
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/rocm
ARG VLLM_VERSION=0.25.1+rocm723
ARG FLASH_ATTN_VERSION=2.8.3
ARG ROCM_WHEEL_INDEX_URL=https://rocm.nightlies.amd.com/whl-multi-arch/
ARG ROCM_SDK_VERSION=7.15.0a20260717
ARG ROCM_SDK_LIBRARIES_PACKAGE=rocm-sdk-libraries
ARG ROCM_SDK_DEVICE_PACKAGE=rocm-sdk-device-gfx1201
ARG ROCM_SDK_LIBRARIES_DIR=_rocm_sdk_libraries

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/vllm
ENV PATH=/opt/vllm/bin:$PATH

# Runtime libs the ROCm userspace + torch wheels dlopen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    libatomic1 libdrm2 libdrm-amdgpu1 libelf1 libgfortran5 libgomp1 \
    libjpeg-turbo8 libnuma1 numactl libopenmpi3t64 \
    git curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/vllm \
    && python -m pip install --upgrade pip setuptools wheel uv

# vLLM ROCm wheel + the exact torch/triton/aiter/flash-attn stack it was built
# against, all from the single vLLM rocm index via --find-links.
RUN uv pip install \
    --find-links ${VLLM_WHEEL_URL}/vllm/ \
    --find-links ${VLLM_WHEEL_URL}/torch/ \
    --find-links ${VLLM_WHEEL_URL}/torchvision/ \
    --find-links ${VLLM_WHEEL_URL}/torchaudio/ \
    --find-links ${VLLM_WHEEL_URL}/triton/ \
    --find-links ${VLLM_WHEEL_URL}/triton-kernels/ \
    --find-links ${VLLM_WHEEL_URL}/amdsmi/ \
    --find-links ${VLLM_WHEEL_URL}/amd-aiter/ \
    --find-links ${VLLM_WHEEL_URL}/flash-attn/ \
    vllm==${VLLM_VERSION} flash-attn==${FLASH_ATTN_VERSION} \
    "fastapi[standard]<0.137" \
    && uv pip install --no-deps torch-c-dlpack-ext

# ROCm userspace libraries (TheRock, packaged as wheels). --no-deps so they don't
# replace torch/vLLM. rocm-sdk-devel ships a _devel.tar we unpack in place.
RUN uv pip install --no-deps \
    --index-url ${ROCM_WHEEL_INDEX_URL} \
    rocm-sdk-core==${ROCM_SDK_VERSION} \
    ${ROCM_SDK_LIBRARIES_PACKAGE}==${ROCM_SDK_VERSION} \
    ${ROCM_SDK_DEVICE_PACKAGE}==${ROCM_SDK_VERSION} \
    rocm-sdk-devel==${ROCM_SDK_VERSION} \
    && python - <<'PY'
import os, site, tarfile
sp = next((p for p in site.getsitepackages() if p.endswith("site-packages")), site.getsitepackages()[0])
tar_path = os.path.join(sp, "rocm_sdk_devel", "_devel.tar")
with tarfile.open(tar_path) as ar:
    root = os.path.abspath(sp)
    for m in ar.getmembers():
        t = os.path.abspath(os.path.join(root, m.name))
        if not t.startswith(root + os.sep):
            raise RuntimeError(f"unsafe ROCm SDK member path: {m.name}")
    ar.extractall(root)
os.remove(tar_path)
PY

# Point ROCm/HIP + the dynamic loader at the in-venv SDK dirs.
ENV SP=/opt/vllm/lib/python3.12/site-packages
# _rocm_sdk_core is the real ROCm root (has bin/, .info/version, bitcode);
# _rocm_sdk_devel's bin/ is empty in this SDK version, so point HIP/ROCm at core.
ENV ROCM_PATH=${SP}/_rocm_sdk_core \
    ROCM_HOME=${SP}/_rocm_sdk_core \
    HIP_PATH=${SP}/_rocm_sdk_core \
    HIP_HOME=${SP}/_rocm_sdk_core \
    HIP_DEVICE_LIB_PATH=${SP}/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
    DEVICE_LIB_PATH=${SP}/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
    CPATH=${SP}/_rocm_sdk_devel/include \
    LIBRARY_PATH=${SP}/_rocm_sdk_devel/lib \
    PYTHONPATH=${SP}/_rocm_sdk_core/share/amd_smi \
    HIPBLASLT_TENSILE_LIBPATH=${SP}/${ROCM_SDK_LIBRARIES_DIR}/lib/hipblaslt/library/${GFX_ARCH} \
    ROCBLAS_TENSILE_LIBPATH=${SP}/${ROCM_SDK_LIBRARIES_DIR}/lib/rocblas/library
ENV PATH=${SP}/_rocm_sdk_core/bin:${PATH}
ENV LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/llvm/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_libraries/lib

ENV HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_ROCM_GCN_ARCH=${GFX_ARCH} \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} \
    AMDGPU_TARGETS=${GFX_ARCH} \
    GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 \
    TOKENIZERS_PARALLELISM=false \
    TRITON_CACHE_AUTOTUNING=1
# ^ persist Triton autotune config selections (*.autotune.json). Only has effect
#   when TRITON_CACHE_DIR is a persistent mount; without it the FLA/GDN kernels
#   re-run do_bench (~20 min, looks like a hang) on every start.

# --- ROCm dev symlinks + aiter 0.1.16 (source build for gfx1201) ---
# TheRock ROCm-SDK wheels ship runtime libX.so.N but NOT the unversioned linker
# symlinks (libX.so) that aiter's runtime JIT needs (`ld -lamdhip64` else fails).
RUN cd ${SP}/_rocm_sdk_core/lib \
    && for f in *.so.*; do b="${f%%.so.*}.so"; [ -e "$b" ] || ln -s "$f" "$b"; done

# Replace the transitively-pulled amd-aiter 0.1.13 wheel (CDNA-only prebuilt .so,
# no gfx1201) with aiter 0.1.16.post3 built FROM SOURCE for gfx1201 so its Triton
# kernels (incl. the split-K block-FP8 GEMM) JIT for gfx1201. triton PINNED to
# 3.6.0 so aiter can't bump it (vLLM 0.25 hard-pins ==3.6.0). PREBUILD_KERNELS=0:
# HIP modules JIT at first use (cached in /cache/aiter); pip install stays GPU-free.
# aiter's asm quant kernels (module_quant) are CDNA-only (no gfx1201 code objects at
# any version), so VLLM_ROCM_USE_AITER_LINEAR stays 0 permanently (LINEAR=1 ->
# module_quant -> invalid device function). Block-FP8 GEMM routes on the LINEAR=0
# path via the radiance dispatcher (patch_radiance_dispatch.py / radiance_kernels.py).
ARG AITER_VERSION=v0.1.16.post3
RUN pip install --no-cache-dir cmake \
    && git clone --depth 1 --recurse-submodules --shallow-submodules \
        -b ${AITER_VERSION} https://github.com/ROCm/aiter.git /opt/aiter \
    && pip uninstall -y amd-aiter aiter \
    && printf 'triton==3.6.0\n' > /opt/aiter-constraints.txt \
    && cd /opt/aiter \
    && PIP_CONSTRAINT=/opt/aiter-constraints.txt GPU_ARCHS=${GFX_ARCH} \
       PREBUILD_KERNELS=0 MAX_JOBS=6 pip install --no-build-isolation . \
    # aiter's deps pull a triton_kernels build (1.0.0+amd...) that LACKS the
    # matmul_ogs submodule vLLM imports (non-fatal ERROR + MoE-OGS fallback).
    # Restore vLLM's triton_kernels (has matmul_ogs) from the vLLM index.
    && pip install --no-deps --force-reinstall --no-cache-dir \
       --find-links ${VLLM_WHEEL_URL}/triton-kernels/ triton_kernels \
    && python -c "import triton, triton_kernels.matmul_ogs, importlib.metadata as m; \
print('aiter', m.version('amd-aiter'), '| triton', triton.__version__, '| triton_kernels.matmul_ogs OK')"

# --- gfx1201 fix overlay ---
# (1) amdsmi init-order fix: a .pth executed at site-init (before HIP, in every
#     process) so amdsmi enumerates the GPUs and stays alive. Fixes platform
#     detection, device_count, get_device_name.
# (2) curated idempotent patches: gcn-arch env, AITER-enable-on-gfx12x, Triton
#     is_active, AITER-sampler-gate. See patch_gfx1201.py.
COPY radiance_amdsmi.py radiance_amdsmi.pth ${SP}/
COPY patch_gfx1201.py /opt/patch_gfx1201.py
RUN python /opt/patch_gfx1201.py

# --- gfx1201 tuned block-FP8 GEMM configs (GATED) + custom kernel dispatcher ---
# GATED set: default config @ M<=32 (no decode regression) + num_stages=1 @ M>=64
# (faster prefill), end-to-end A/B validated. The generic kernel auto-loads these
# device-named JSONs.
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
# Radiance dispatcher (baked): hooks the block-FP8 GEMM chokepoint (apply_block_scaled_mm)
# -> radiance_kernels.block_scaled_mm, routed per-M: M<=8 -> AITER split-K (with the K=8704
# NUM_KSPLIT scale-align crash-fix); M>=16 -> generic (reads the tuned JSONs above).
# Dynamo-safe (plain shape branch -> registered torch.ops.*).
COPY radiance_kernels.py patch_radiance_dispatch.py /opt/
RUN python /opt/patch_radiance_dispatch.py

# bf16 / 2-byte (--kv-cache-dtype auto) KV attention config for the aiter unified-attention 3D
# decode kernel. LDS-critical: a 2-byte KV cache overflows the R9700's 64 KiB shared memory at
# head_size 256 with aiter's default (TILE 64, stages 2) -> Triton OutOfResources at cudagraph
# capture. SOURCE patch of select_3d_config (not the RADIANCE_ATTN_TUNE _s3 wrapper, bypassed for
# the bf16 3D path); do_bench-tuned for gfx1201 head-256. See the patch header.
COPY patch_unified_attention_bf16.py /opt/patch_unified_attention_bf16.py
RUN python /opt/patch_unified_attention_bf16.py

# Preshuffle (AITER gemm_a8w8_blockscale_preshuffle) + tuned attention config, both e2e-validated.
#  - patch_preshuffle.py: BlockScaledMMLinearKernel.apply_weights output_shape fix (the shuffle
#    rewrites weight.shape[0] to N//16). radiance_kernels.py (above) carries the preshuffle
#    torch.op + dispatcher route.
#  - install_radiance_hooks.py: appends radiance_kernels.install_all() to vllm's per-process
#    load_general_plugins(): installs the weight-shuffle-at-load hook + select_2d/3d_config
#    attention overrides BEFORE model load, in every worker. Both env-gated by the ENV below.
COPY patch_preshuffle.py install_radiance_hooks.py /opt/
RUN python /opt/patch_preshuffle.py && python /opt/install_radiance_hooks.py
ENV RADIANCE_PRESHUFFLE=1 \
    RADIANCE_ATTN_TUNE=1 \
    RADIANCE_FUSE_RMS_QUANT=1

# --- radiance rms_norm(+fused_add) + group-fp8-quant fusion coverage fix ---
# Stock RocmAiterRMSNormQuantFusionPass registers the group rms+quant patterns with
# only the aiter-quant matcher; on gfx1201 our graph emits the NATIVE
# _C.per_token_group_fp8_quant, so it matched 0 and the standalone bf16->fp8
# group-quant kernels never folded into the rms epilogue. This registers the native
# variant too, reusing vLLM's own is_quant_fp8_enabled duplicate-pattern guard.
# Numerically exact (same fused replacement op). Gated by RADIANCE_FUSE_RMS_QUANT
# (default 1 above; set 0 for stock behaviour).
COPY patch_radiance_fusion.py /opt/patch_radiance_fusion.py
RUN python /opt/patch_radiance_fusion.py \
    && python -c "import ast; ast.parse(open('${SP}/vllm/compilation/passes/fusion/rocm_aiter_fusion.py').read()); print('radiance fusion patch parses OK')"

# --- radiance fast-reduce (multi-block one-shot P2P-BAR all-reduce, gfx1201 / TP=2 / PCIe) ---
# Prebuilt pybind11+HIP extensions: radiance_ar_ext.so (bf16 payload, byte-identical to RCCL) and
# radiance_ar_quant_ext.so (optional fp8 payload). The .hip sources are kept at the repo root and
# build with `make` for reproducibility. RadianceAllreduce wraps CudaCommunicator.all_reduce
# (size-gated; auto RCCL fallback above the gate). Installed per-process by install_custom_ar().
# ON by default (RADIANCE_FAST_REDUCE=1); byte-identical to RCCL (bit-checked). Set 0 for RCCL.
# Gate RADIANCE_AR_MAX_KB=32768 (32MB): the kernel wins at every size to 48MB (buffer 2*gate/rank).
# RADIANCE_AR_QUANT=1 quantizes the payload to block-scaled fp8 for messages >= RADIANCE_AR_QUANT_MIN_KB
# (halves the PCIe bytes; faster for large/prefill messages; not bit-identical to RCCL). On by default;
# set 0 for the exact bf16 all-reduce.
COPY radiance_ar_ext.so radiance_ar_quant_ext.so radiance_allreduce.py ${SP}/
ENV RADIANCE_FAST_REDUCE=1 \
    RADIANCE_AR_MAX_KB=32768 \
    RADIANCE_AR_QUANT=1 \
    RADIANCE_AR_QUANT_MIN_KB=128

# --- radiance dynamic drafting (RADIANCE_DYNAMIC_DRAFT) ---
# Runtime module (radiance_draft.py) installed per-process by install_all() -> install().
# LOSSLESS per-request MTP draft-depth control (changes only DRAFT LENGTH + whether tokens come
# from MTP or a verbatim copy of earlier text; greedy draft => any drafted token verifies
# identically through the unchanged rejection sampler -> outputs cannot change, throughput only).
# A per-request, per-slot confidence-threshold controller decides A=another MTP pass / B=take a verbatim
# n-gram / C=verify, all on-device (Triton confidence capture + n-gram matcher, radiance_draft_gpu.py).
# Baked ON below; set RADIANCE_DYNAMIC_DRAFT=0 for byte-identical stock MTP. Tune with the two knobs
# RADIANCE_DRAFT_SCHEDULE / RADIANCE_DRAFT_TAU. Details in the radiance_draft.py module docstring.
COPY radiance_draft.py radiance_draft_gpu.py ${SP}/
RUN python -c "import ast; ast.parse(open('${SP}/radiance_draft.py').read()); ast.parse(open('${SP}/radiance_draft_gpu.py').read()); print('radiance_draft modules parse OK')"
# The per-slot controller decides A=another MTP pass / B=take n-gram / C=verify, and short-circuits
# the draft loop; this patch injects the one `break` into SpecDecodeBaseProposer.propose that honours
# it (skips the remaining draft forwards). Without it the controller can only trim post-hoc.
COPY patch_mtp_loopbreak.py /opt/patch_mtp_loopbreak.py
RUN python /opt/patch_mtp_loopbreak.py \
    && python -c "import ast; ast.parse(open('${SP}/vllm/v1/spec_decode/llm_base_proposer.py').read()); print('radiance mtp loop-break patch parses OK')"
ENV RADIANCE_DYNAMIC_DRAFT=1 \
    RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 \
    RADIANCE_DRAFT_TAU=0.35

# --- radiance unpad fix (propagate seq_lens_cpu_upper_bound through CommonAttentionMetadata.unpadded) ---
# vLLM's unpadded() drops seq_lens_cpu_upper_bound; under MTP disable_padded_drafter_batch, when the
# real batch != padded cudagraph batch (chunked prefill / mixed), the drafter's prepare_inputs asserts
# on it -> EngineDeadError at >= 16 concurrent sequences. Restores the optimistic bound (sliced like the other per-req
# fields). Pure correctness fix, always on. Enables the MTP single-stream decode win
# (docker-compose.yml sets num_speculative_tokens=8 + disable_padded_drafter_batch:true).
COPY patch_unpad.py /opt/patch_unpad.py
RUN python /opt/patch_unpad.py \
    && python -c "import ast; ast.parse(open('${SP}/vllm/v1/attention/backend.py').read()); print('radiance unpad patch parses OK')"

# --- radiance tool-parser truncation fix (vLLM #47137: streaming vs non-streaming divergence) ---
# On a tool call truncated by max_tokens/stop INSIDE the <tool_call> opener, non-streaming leaked
# the raw markup as assistant content while streaming dropped it; truncated mid-parameter,
# non-streaming returned '{}' while streaming had accumulated the partial args. Two surgical edits
# (engine-based parsers only: qwen3_coder/qwen3_xml, gemma4, ...) make non-streaming match streaming:
# return the parser's stripped content when no call is promoted, and reuse the streamed args on a
# truncated call. Enable via --enable-auto-tool-choice --tool-call-parser qwen3_coder (see docker-compose.yml).
COPY patch_qwen3_toolparse.py /opt/patch_qwen3_toolparse.py
RUN python /opt/patch_qwen3_toolparse.py \
    && python -c "import ast; ast.parse(open('${SP}/vllm/parser/abstract_parser.py').read()); ast.parse(open('${SP}/vllm/parser/engine/parser_engine.py').read()); print('radiance tool-parser patch parses OK')"

# --- radiance from_json Jinja filter (unblocks the "fixed" Qwen3.6 chat template) ---
# The stock Qwen3.6 chat template crashes multi-turn tool loops: `tool_call.arguments | items`
# hits the JSON *string* vLLM replays for an assistant tool call -> "Can only get item pairs from
# a mapping", stalling any agent after its first tool call. This image bundles the community fix
# qwen3.6-enhanced.jinja (--chat-template /work/qwen3.6-enhanced.jinja) which parses string args
# via `arguments | from_json`, but transformers' sandboxed jinja env registers only tojson /
# raise_exception / strftime_now, so the template raises "No filter named 'from_json'". This adds
# the one missing filter (json.loads on a string, NOOP otherwise). Superset-safe: no existing
# template references from_json.
COPY patch_from_json_filter.py /opt/patch_from_json_filter.py
RUN python /opt/patch_from_json_filter.py \
    && python -c "import ast; ast.parse(open('${SP}/transformers/utils/chat_template_utils.py').read()); print('radiance from_json filter patch parses OK')"

# Fail the build early if the native stack doesn't import.
RUN python -c 'import importlib.metadata as md, torch, vllm, vllm._C, vllm._rocm_C, flash_attn; \
print("torch", torch.__version__, "| hip", torch.version.hip); \
print("vllm", vllm.__version__, "| aiter", md.version("amd-aiter")); \
print("native ext OK")'

# --- pyc hygiene: never let a stale/baked .pyc shadow a patched or mounted .py ---
# Every radiance patch edits a vllm .py in place; the image's .pyc are timestamp-based
# and can silently shadow an edit (or a runtime -v mount). Wipe all baked vllm .pyc and
# disable runtime bytecode writes so the patched source is always what runs. Cost: a few
# seconds of in-memory recompile at startup.
ENV PYTHONDONTWRITEBYTECODE=1
RUN find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true

# --- P2P measurement tooling (PREBUILT binaries, no build deps baked into the image) ---
# rocm-bandwidth-test (rocm-6.4.4 classic tag, built ONCE against this image's exact libhsa)
# + radiance_p2p_probe (in-kernel direct-BAR P2P latency/bandwidth probe). Both link only in-image
# libs (ldd fully resolves) and are used to measure/validate the PCIe P2P path the all-reduce uses.
# Both land on PATH (/opt/vllm/bin): run as `rocm-bandwidth-test -t` or `radiance_p2p_probe`
# (needs both GPUs).
COPY rocm-bandwidth-test radiance_p2p_probe /opt/vllm/bin/
RUN chmod +x /opt/vllm/bin/rocm-bandwidth-test /opt/vllm/bin/radiance_p2p_probe

# --- RADIANCE startup banner + environment preamble ---
# A thin entrypoint (radiance_entrypoint.sh) that prints the RADIANCE ASCII banner and runs
# best-effort prechecks (GPU count, gfx1201-only verification, P2P access, the enabled/baked
# radiance optimizations, component versions), then exec's `vllm serve "$@"` so vLLM becomes
# PID 1 and the container `command:` args pass through verbatim. Also kicks off rocm-bandwidth-test
# in the BACKGROUND. Every check is non-fatal. The banner runs once in the launcher process, not
# per TP worker. RADIANCE_VERSION is the source of truth for the version string the banner prints
# (bump via --build-arg RADIANCE_VERSION=...).
ARG RADIANCE_VERSION=0.2.3
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh

ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
