# vllm-radiance: vLLM/torch/triton/aiter stack for RDNA4 (gfx1201 / R9700), plus the radiance
# patches and kernels. Single multistage build on the official AMD ROCm image. Stage 1 compiles
# the stack from source into wheels; stage 2 installs them and applies the patches and kernels.
# No prebuilt component wheels and no checked-in binaries.
#
# stack: torch 2.13.0, triton 3.7.1, torchvision 0.28.0, aiter v0.1.17, vLLM v0.26.0,
# all compiled for PYTORCH_ROCM_ARCH=gfx1201 against the base image's ROCm 7.14.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7
ARG GFX_ARCH=gfx1201

# =====================================================================================
# STAGE 1 builder: compile the stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    USE_ROCM=1 USE_CUDA=0 MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32

# Build tooling the base dev image lacks (git/venv/pkg-config + the -dev packages torch's cmake
# probes: libdrm for rocm_smi, libnuma, libelf).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3.12-venv build-essential ccache pkg-config \
      libdrm-dev libnuma-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py
ENV PATH=/opt/py/bin:$PATH
RUN pip install -U pip wheel setuptools "cmake<4" ninja pybind11 numpy pyyaml typing_extensions cffi requests
RUN mkdir -p /wheels

# --- torch 2.13.0 (AOTriton off: its gfx1201 source-configure fails and vLLM never uses torch
#     SDPA-flash; USE_MAGMA=0: base has no magma) ---
RUN git clone --depth 1 -b v2.13.0 --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/pytorch.git /src/pytorch \
    && cd /src/pytorch \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       PYTORCH_BUILD_VERSION=2.13.0+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       python setup.py bdist_wheel \
    && cp dist/*.whl /wheels/ && pip install dist/*.whl && rm -rf /src/pytorch

# --- triton 3.7.1 ---
RUN git clone --depth 1 -b v3.7.1 https://github.com/triton-lang/triton.git /src/triton \
    && cd /src/triton && pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- torchvision 0.28.0 ---
# FORCE_CUDA=1 is REQUIRED: torchvision 0.28's BUILD_CUDA_SOURCES gates on torch.cuda.is_available(),
# which is false in `docker build` (no GPU) -> it picks CppExtension, where torch's build-time hipify
# double-compiles vision.cpp + vision_hip.cpp -> "multiple definition of vision::cuda_version()".
# FORCE_CUDA=1 forces CUDAExtension (correct hipify source replacement); hipcc needs no GPU to compile.
RUN git clone --depth 1 -b v0.28.0 https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- aiter v0.1.17 (gfx1201; kernels JIT at runtime, PREBUILD_KERNELS=0) ---
RUN git clone --recursive --shallow-submodules -b v0.1.17 https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/*aiter-*.whl && rm -rf /src/aiter

# --- vLLM v0.26.0, built against the torch above. use_existing_torch drops the torch==2.11 pin;
#     setuptools-scm and setuptools-rust are pyproject build requirements that --no-build-isolation
#     does not install. ---
RUN git clone --depth 1 -b v0.26.0 https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && python use_existing_torch.py \
    && pip install "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" \
    && VLLM_TARGET_DEVICE=rocm pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vllm
RUN ls -la /wheels

# =====================================================================================
# STAGE 2 final: install the wheels and apply the patches and kernels
# =====================================================================================
FROM ${ROCM_BASE} AS final
ARG GFX_ARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv libnuma1 numactl git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/vllm
RUN python3 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:$PATH
ENV SP=/opt/vllm/lib/python3.12/site-packages

# --- install the wheels ---
# torch/triton/vision/aiter with --no-deps so pip does not replace them; vLLM with its pure-python
# dependencies. amdsmi (the ROCm python bindings the base image ships) is required for vLLM's ROCm
# platform detection.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir -U pip wheel setuptools \
 && pip install --no-cache-dir --no-deps \
      /wheels/torch-*.whl /wheels/triton-*.whl /wheels/torchvision-*.whl /wheels/*aiter-*.whl \
 && pip install --no-cache-dir /wheels/vllm-*.whl \
 && pip install --no-cache-dir /opt/rocm/share/amd_smi pillow pybind11 \
 && rm -rf /wheels /root/.cache

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} VLLM_ROCM_GCN_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- runtime modules and configs ---
# radiance_amdsmi.py and .pth: amdsmi init-order fix. amdsmi must init before HIP at site-init in
# every process, otherwise it enumerates 0 devices and platform detection fails.
COPY radiance_amdsmi.py radiance_amdsmi.pth \
     radiance_kernels.py radiance_vit_attn.py radiance_allreduce.py \
     radiance_draft.py radiance_draft_gpu.py radiance_router.py ${SP}/
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
COPY moe-configs/ ${SP}/vllm/model_executor/layers/fused_moe/configs/

# --- gfx1201 fixes and tuned-kernel patches ---
# Each patch edits a vLLM (or aiter/triton) source file in place and checks for source drift before
# writing. patch_gdn_wmma covers the solve_tril triangular block-inverse only; the gated-delta-net
# gram cast is handled in 0.26.0 upstream.
COPY patch_*.py install_radiance_hooks.py _patchlib.py /opt/patches/
RUN set -eu; cd /opt/patches; \
    for p in patch_gfx1201 patch_radiance_dispatch patch_router_gemm patch_unified_attention_lds \
             patch_gdn_wmma patch_preshuffle patch_radiance_fusion install_radiance_hooks \
             patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter; do \
      echo "== applying $p =="; python "$p.py"; \
    done; \
    python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; print('radiance modules parse OK')"

# --- gfx1201 HIP kernels, compiled from source ---
# router_gemm: bf16 MoE-gate GEMM. radiance_ar_ext: bf16 P2P all-reduce. radiance_ar_quant_ext:
# fp8-payload all-reduce, built with -ffp-contract=off (otherwise the two TP ranks diverge by ~1 ULP).
COPY router_gemm.hip radiance_ar_ext.hip radiance_ar_quant_ext.hip /opt/patches/
RUN INC=$(python -m pybind11 --includes); B="-O3 -std=c++17 -fPIC -shared --offload-arch=${GFX_ARCH} -Wno-unused-result"; \
    hipcc $B -DTEMPORAL $INC /opt/patches/router_gemm.hip        -o ${SP}/router_gemm.so && \
    hipcc $B              $INC /opt/patches/radiance_ar_ext.hip   -o ${SP}/radiance_ar_ext.so && \
    hipcc $B -ffp-contract=off $INC /opt/patches/radiance_ar_quant_ext.hip -o ${SP}/radiance_ar_quant_ext.so && \
    test -f ${SP}/router_gemm.so && test -f ${SP}/radiance_ar_ext.so && test -f ${SP}/radiance_ar_quant_ext.so && \
    echo "radiance HIP kernels built"

# --- radiance feature flags (set any to 0 to fall back to stock). RADIANCE_GDN_WMMA gates the
#     solve_tril fp16 path. ---
ENV RADIANCE_PRESHUFFLE=1 RADIANCE_ATTN_TUNE=1 RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_GDN_WMMA=1 RADIANCE_VIT_FLASH=1 \
    RADIANCE_FAST_REDUCE=1 RADIANCE_AR_MAX_KB=32768 RADIANCE_AR_QUANT=1 RADIANCE_AR_QUANT_MIN_KB=128 \
    RADIANCE_DYNAMIC_DRAFT=1 RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 RADIANCE_DRAFT_TAU=0.35 \
    RADIANCE_MOE_ROUTER=1

# Fail the build if the native stack does not import. Kept GPU-free: no `import aiter` (it runs
# rocminfo) and no full `import vllm`; versions are read from package metadata.
RUN python -c 'import torch, vllm._C, amdsmi, importlib.metadata as m; \
print("stack OK | vllm", m.version("vllm"), "| torch", torch.__version__, "| aiter", m.version("amd-aiter"))'
RUN find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true

ARG RADIANCE_VERSION=0.5.0
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
