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

# vLLM ROCm wheel + the exact torch/triton/aiter/flash-attn stack it was built against, all from
# the single vLLM rocm index via --find-links. --no-cache keeps uv from leaving its (~15 GB,
# hardlinked) download cache in the layer. The trailing RM strips build/CUDA-only bloat IN THIS
# LAYER (so the image actually shrinks; a later RM only whiteouts): the Triton NVIDIA backend
# (CUDA-only, dead on ROCm; also ~74 MB of cupti static libs) and xgrammar's link-time static
# archive (the runtime uses xgrammar's .so).
RUN uv pip install --no-cache \
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
    && uv pip install --no-cache --no-deps torch-c-dlpack-ext \
    && rm -rf /opt/vllm/lib/python3.12/site-packages/triton/backends/nvidia \
              /opt/vllm/lib/python3.12/site-packages/xgrammar/lib/libxgrammar.a

# ROCm userspace libraries (TheRock, packaged as wheels). --no-deps so they don't replace
# torch/vLLM. We deliberately DO NOT install rocm-sdk-devel: its only payload is a ~2.7 GB
# _devel.tar of headers/static libs that aiter's runtime JIT never uses (JIT compiles against
# _rocm_sdk_core's hip headers), so it is pure image bloat.
RUN uv pip install --no-cache --no-deps \
    --index-url ${ROCM_WHEEL_INDEX_URL} \
    rocm-sdk-core==${ROCM_SDK_VERSION} \
    ${ROCM_SDK_LIBRARIES_PACKAGE}==${ROCM_SDK_VERSION} \
    ${ROCM_SDK_DEVICE_PACKAGE}==${ROCM_SDK_VERSION}

# Point ROCm/HIP + the dynamic loader at the in-venv SDK dirs.
ENV SP=/opt/vllm/lib/python3.12/site-packages
# _rocm_sdk_core is the real ROCm root (has bin/, .info/version, bitcode, hip headers); HIP/ROCm,
# the JIT include path (CPATH) and the linker path (LIBRARY_PATH) all point at it.
ENV ROCM_PATH=${SP}/_rocm_sdk_core \
    ROCM_HOME=${SP}/_rocm_sdk_core \
    HIP_PATH=${SP}/_rocm_sdk_core \
    HIP_HOME=${SP}/_rocm_sdk_core \
    HIP_DEVICE_LIB_PATH=${SP}/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
    DEVICE_LIB_PATH=${SP}/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
    CPATH=${SP}/_rocm_sdk_core/include \
    LIBRARY_PATH=${SP}/_rocm_sdk_core/lib \
    PYTHONPATH=${SP}/_rocm_sdk_core/share/amd_smi \
    HIPBLASLT_TENSILE_LIBPATH=${SP}/${ROCM_SDK_LIBRARIES_DIR}/lib/hipblaslt/library/${GFX_ARCH} \
    ROCBLAS_TENSILE_LIBPATH=${SP}/${ROCM_SDK_LIBRARIES_DIR}/lib/rocblas/library
ENV PATH=${SP}/_rocm_sdk_core/bin:${PATH}
ENV LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/llvm/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_libraries/lib

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
# The final `&& rm/uninstall` strip build-only bloat IN THIS LAYER (so it shrinks the
# image): the aiter source clone (aiter installs into site-packages; /opt/aiter is
# unused at runtime), cmake (aiter JIT uses ninja, not cmake), the CDNA (MI3xx) code
# objects gfx1201 never loads, and the pip/uv caches.
ARG AITER_VERSION=v0.1.16.post3
RUN pip install --no-cache-dir cmake \
    && git clone --depth 1 --recurse-submodules --shallow-submodules \
        -b ${AITER_VERSION} https://github.com/ROCm/aiter.git /opt/aiter \
    && pip uninstall -y amd-aiter aiter \
    && printf 'triton==3.6.0\n' > /opt/aiter-constraints.txt \
    && cd /opt/aiter \
    && PIP_CONSTRAINT=/opt/aiter-constraints.txt GPU_ARCHS=${GFX_ARCH} \
       PREBUILD_KERNELS=0 MAX_JOBS=6 pip install --no-cache-dir --no-build-isolation . \
    # aiter's deps pull a triton_kernels build (1.0.0+amd...) that LACKS the
    # matmul_ogs submodule vLLM imports (non-fatal ERROR + MoE-OGS fallback).
    # Restore vLLM's triton_kernels (has matmul_ogs) from the vLLM index.
    && pip install --no-deps --force-reinstall --no-cache-dir \
       --find-links ${VLLM_WHEEL_URL}/triton-kernels/ triton_kernels \
    && python -c "import triton, triton_kernels.matmul_ogs, importlib.metadata as m; \
print('aiter', m.version('amd-aiter'), '| triton', triton.__version__, '| triton_kernels.matmul_ogs OK')" \
    && pip uninstall -y cmake \
    && rm -rf /opt/aiter /opt/aiter-constraints.txt \
       ${SP}/aiter_meta/hsa/gfx942 ${SP}/aiter_meta/hsa/gfx950 \
       /root/.cache/pip /root/.cache/uv

# =====================================================================================
# gfx1201 fix + tuned-kernel overlay (single consolidated block)
# =====================================================================================
# The full rationale for every fix lives in that patch's own module docstring; each patch
# ast.parse()s the result before writing and exits nonzero on upstream source drift, so the
# loop below needs no separate parse step. Patches are idempotent and order-independent (they
# edit distinct files). Ordered/grouped by purpose: gfx1201-enablement, kernel-tuning,
# correctness fixes, agentic serving.
#
# What each COPY'd runtime module / patch does:
#   radiance_amdsmi.py + .pth  amdsmi init-order fix; the .pth execs `import radiance_amdsmi` at
#                              site-init (before HIP, in every process) so amdsmi enumerates the
#                              GPUs and stays alive -> platform detect / device_count / names.
#   radiance_kernels.py        the block-FP8 GEMM dispatch table + install_all() runtime-hook
#                              fan-out (called from vllm's plugin loader via install_radiance_hooks).
#   fp8-configs/*.json         gfx1201 tuned block-FP8 GEMM configs (default @ M<=32, num_stages=1
#                              @ M>=64), auto-loaded by the generic kernel.
#   radiance_vit_attn.py       native head_dim-72 ViT flash attention (multimodal).
#   radiance_ar_ext.so /       prebuilt P2P one-shot all-reduce extensions (bf16 exact + fp8
#     radiance_ar_quant_ext.so payload); radiance_allreduce.py wraps CudaCommunicator.all_reduce.
#   radiance_draft*.py         lossless per-request MTP draft-depth controller.
#   patch_gfx1201              gcn-arch env, AITER-enable-on-gfx12x, Triton is_active, sampler gate.
#   patch_radiance_dispatch    hooks apply_block_scaled_mm -> dispatcher (+ K=8704 splitk fix).
#   patch_unified_attention_bf16  bf16/auto KV 3D-decode config (fits the R9700 64 KiB LDS @head256).
#   patch_gdn_wmma             gated-delta-net gram + triangular solve on the fp16 matrix cores.
#   patch_preshuffle           BlockScaledMM output_shape fix for the preshuffled weight layout.
#   patch_radiance_fusion      folds native group-FP8 quant into the RMSNorm epilogue on gfx1201.
#   install_radiance_hooks     appends radiance_kernels.install_all() to vllm's plugin loader.
#   patch_unpad                propagate seq_lens_cpu_upper_bound through unpadded() (MTP drafter).
#   patch_mtp_mm_mask          align the image-placeholder mask with the compacted drafter batch.
#   patch_mtp_loopbreak        injects the `break` the draft-depth controller uses to stop early.
#   patch_qwen3_toolparse      tool-parser streaming vs non-streaming consistency (vLLM #47137).
#   patch_from_json_filter     adds the `from_json` Jinja filter the fixed chat template needs.
COPY radiance_amdsmi.py radiance_amdsmi.pth \
     radiance_kernels.py radiance_vit_attn.py \
     radiance_ar_ext.so radiance_ar_quant_ext.so radiance_allreduce.py \
     radiance_draft.py radiance_draft_gpu.py ${SP}/
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
COPY patch_*.py install_radiance_hooks.py _patchlib.py /opt/patches/
RUN set -eu; cd /opt/patches; \
    for p in \
      patch_gfx1201 \
      patch_radiance_dispatch patch_unified_attention_bf16 patch_gdn_wmma \
      patch_preshuffle patch_radiance_fusion install_radiance_hooks \
      patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak \
      patch_qwen3_toolparse patch_from_json_filter; do \
        echo "== applying $p =="; python "$p.py"; \
    done; \
    python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; print('radiance runtime modules parse OK')"

# Radiance feature flags — all baked ON (each set 0 to fall back to stock). Rationale for each
# is in the corresponding patch/module docstring; the startup banner prints their live state.
#   PRESHUFFLE/ATTN_TUNE/FUSE_RMS_QUANT  GEMM + attention + rmsnorm kernel paths
#   GDN_WMMA/VIT_FLASH                    fp16 matrix-core GDN gram; native-72 ViT flash
#   FAST_REDUCE/AR_*                      P2P all-reduce + fp8-payload size gates (TP=2)
#   DYNAMIC_DRAFT/DRAFT_*                 lossless per-request MTP draft-depth controller
ENV RADIANCE_PRESHUFFLE=1 \
    RADIANCE_ATTN_TUNE=1 \
    RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_GDN_WMMA=1 \
    RADIANCE_VIT_FLASH=1 \
    RADIANCE_FAST_REDUCE=1 \
    RADIANCE_AR_MAX_KB=32768 \
    RADIANCE_AR_QUANT=1 \
    RADIANCE_AR_QUANT_MIN_KB=128 \
    RADIANCE_DYNAMIC_DRAFT=1 \
    RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 \
    RADIANCE_DRAFT_TAU=0.35

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

# --- P2P bandwidth measurement (rocm-bandwidth-test, PREBUILT; no build deps in the image) ---
# AMD's upstream tool (rocm-6.4.4 classic tag, built once against this image's exact libhsa; ldd
# fully resolves). Lands on PATH (/opt/vllm/bin): run as `rocm-bandwidth-test -t`. The entrypoint
# can run it as an OPTIONAL startup topology/bandwidth sweep (off by default; RADIANCE_RUN_BWTEST=1).
COPY rocm-bandwidth-test /opt/vllm/bin/
RUN chmod +x /opt/vllm/bin/rocm-bandwidth-test

# --- RADIANCE startup banner + environment preamble ---
# A thin entrypoint (radiance_entrypoint.sh) that prints the RADIANCE ASCII banner and runs
# best-effort prechecks (GPU count, gfx1201-only verification, P2P access, the enabled/baked
# radiance optimizations, component versions), then exec's `vllm serve "$@"` so vLLM becomes
# PID 1 and the container `command:` args pass through verbatim. Every check is non-fatal. The
# banner runs once in the launcher process, not per TP worker. RADIANCE_VERSION is the source of
# truth for the version string the banner prints (bump via --build-arg RADIANCE_VERSION=... or the
# VERSION file that the Makefile / build command reads).
ARG RADIANCE_VERSION=0.2.8
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh

ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
