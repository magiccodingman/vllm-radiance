# vllm-radiance: vLLM/torch/triton/aiter stack for RDNA4 (gfx1201 / R9700), plus the radiance
# patches and kernels. Single multistage build on the official AMD ROCm image. Stage 1 compiles
# the stack from source into wheels; stage 2 installs them and applies the patches and kernels.
# No prebuilt component wheels and no checked-in binaries.
#
# stack: torch 2.11.0, triton 3.6.0, torchvision 0.24.1, aiter v0.1.17, vLLM v0.26.0,
# all compiled for PYTORCH_ROCM_ARCH=gfx1201 against the base image's ROCm 7.14.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7
ARG GFX_ARCH=gfx1201

# Component pins, in one place. Each is both the git tag that gets compiled and the version the
# resulting wheel reports, so `pip show`, `importlib.metadata`, and the startup banner all agree
# with what was actually built.
# torch/triton/torchvision are NOT free choices: vLLM 0.26.0's pyproject pins `torch == 2.11.0`,
# torch 2.11.0 pins triton 3.6.0, and torchvision 0.24.1 is its matching release. Building against
# newer ones means running a combination upstream never tests. 0.5.0-0.5.4 did exactly that (torch
# 2.13 / triton 3.7.1 / torchvision 0.28) because `use_existing_torch.py` strips the pin, and those
# builds hang the GPU under load where 0.4.0 -- which used this sanctioned trio -- does not.
ARG TORCH_VERSION=2.11.0
ARG TRITON_VERSION=3.6.0
ARG TORCHVISION_VERSION=0.24.1
ARG AITER_VERSION=0.1.17
ARG VLLM_VERSION=0.26.0
# rocm-bandwidth-test for the startup topology/bandwidth sweep. Pinned to the NEWEST tag that still
# has a plain CMakeLists: the rocm-7.x tags moved to a cmake framework that demands clang>=19 on PATH
# plus vendored boost/fmt/curl submodules, none of which this tool needs.
ARG RBT_VERSION=rocm-6.4.4

# =====================================================================================
# STAGE 1 builder: compile the stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG TORCH_VERSION
ARG TRITON_VERSION
ARG TORCHVISION_VERSION
ARG AITER_VERSION
ARG VLLM_VERSION
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
# setuptools-scm is a build requirement of aiter and vLLM (both take their version from the git
# tag). --no-build-isolation means it is NOT auto-installed: without it here setuptools silently
# ignores `use_scm_version` and the wheel is stamped 0.0.0.
RUN pip install -U pip wheel setuptools "setuptools-scm>=8.0" "cmake<4" ninja pybind11 numpy \
      pyyaml typing_extensions cffi requests
RUN mkdir -p /wheels

# --- torch (AOTriton off: its gfx1201 source-configure fails and vLLM never uses torch
#     SDPA-flash; USE_MAGMA=0: base has no magma) ---
RUN git clone --depth 1 -b v${TORCH_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/pytorch.git /src/pytorch \
    && cd /src/pytorch \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       PYTORCH_BUILD_VERSION=${TORCH_VERSION}+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       python setup.py bdist_wheel \
    && cp dist/*.whl /wheels/ && pip install dist/*.whl && rm -rf /src/pytorch

# --- triton ---
RUN git clone --depth 1 -b v${TRITON_VERSION} https://github.com/triton-lang/triton.git /src/triton \
    && cd /src/triton && pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- torchvision ---
# FORCE_CUDA=1 is REQUIRED: torchvision's BUILD_CUDA_SOURCES gates on torch.cuda.is_available(),
# which is false in `docker build` (no GPU) -> it picks CppExtension, where torch's build-time hipify
# double-compiles vision.cpp + vision_hip.cpp -> "multiple definition of vision::cuda_version()".
# FORCE_CUDA=1 forces CUDAExtension (correct hipify source replacement); hipcc needs no GPU to compile.
RUN git clone --depth 1 -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- aiter (gfx1201; kernels JIT at runtime, PREBUILD_KERNELS=0) ---
# PRETEND_VERSION: the checkout is shallow, so setuptools-scm cannot describe the tag and would fall
# back to a placeholder version; pin it to the tag being built.
RUN git clone --recursive --shallow-submodules -b v${AITER_VERSION} https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 \
       SETUPTOOLS_SCM_PRETEND_VERSION=${AITER_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/*aiter-*.whl && rm -rf /src/aiter

# --- vLLM, built against the torch above. use_existing_torch strips the torch/torchvision pins so
#     pip does not try to fetch them; the versions built above ARE the pinned ones, so this is now
#     just "use what we compiled", not an override.
#     setuptools-rust is a pyproject build requirement that --no-build-isolation does not install.
#     VLLM_VERSION_OVERRIDE pins the reported version to the tag: the tree is dirty (use_existing_torch
#     rewrites the requirements files) and shallow, so setuptools-scm would otherwise stamp the wheel
#     with a guessed next-release dev version plus the build date. ---
RUN git clone --depth 1 -b v${VLLM_VERSION} https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && python use_existing_torch.py \
    && pip install "setuptools-rust>=1.9.0" \
    && VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE=${VLLM_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vllm
RUN ls -la /wheels

# --- rocm-bandwidth-test (the startup topology + bandwidth sweep) ---
# ARG is declared HERE, not in the stage's opening block: an ARG line is a cache-key instruction, so
# putting it up there would invalidate every layer below it -- including the PyTorch compile.
ARG RBT_VERSION
# Plain cmake against the HSA headers/libs the base image already ships; no extra build deps (the
# builder venv's pip cmake and the apt build-essential above cover it). The RUNPATH is REQUIRED:
# ROCm 7.14 keeps libhsa-runtime64.so under the versioned component dir (/opt/rocm/core-<ver>/lib,
# reachable via the `core` alternatives symlink), which is not on the loader's default search path,
# so an un-rpathed binary dies with "libhsa-runtime64.so.1: cannot open shared object file".
RUN git clone --depth 1 -b ${RBT_VERSION} https://github.com/ROCm/rocm_bandwidth_test.git /src/rbt \
    && cmake -S /src/rbt -B /src/rbt/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/opt/rocm/core/lib:/opt/rocm/lib" \
    && cmake --build /src/rbt/build -j 16 \
    && mkdir -p /artifacts && cp /src/rbt/build/rocm-bandwidth-test /artifacts/ \
    && rm -rf /src/rbt

# =====================================================================================
# STAGE 2 final: install the wheels and apply the patches and kernels
# =====================================================================================
FROM ${ROCM_BASE} AS final
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ENV DEBIAN_FRONTEND=noninteractive
# libnuma-dev, not just libnuma1: the rocSHMEM transport linked into torch dlopen()s the unversioned
# `libnuma.so`, which only the -dev package ships. Without it every process that imports torch prints
# "E-001h rocSHMEM Could not open libnuma".
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv libnuma-dev numactl git curl ca-certificates \
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

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
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
             patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter \
             patch_dynamo_metrics; do \
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

# --- the startup topology + bandwidth sweep (RADIANCE_RUN_BWTEST, on by default below) ---
COPY --from=builder /artifacts/rocm-bandwidth-test /usr/local/bin/rocm-bandwidth-test

# --- radiance feature flags (set any to 0 to fall back to stock). RADIANCE_GDN_WMMA gates the
#     solve_tril fp16 path. RADIANCE_RUN_BWTEST runs the bandwidth sweep at startup; it is
#     backgrounded and takes about a second, so it never delays the serve. ---
ENV RADIANCE_PRESHUFFLE=1 RADIANCE_ATTN_TUNE=1 RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_GDN_WMMA=1 RADIANCE_VIT_FLASH=1 \
    RADIANCE_FAST_REDUCE=1 RADIANCE_AR_MAX_KB=32768 RADIANCE_AR_QUANT=1 RADIANCE_AR_QUANT_MIN_KB=128 \
    RADIANCE_DYNAMIC_DRAFT=1 RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 RADIANCE_DRAFT_TAU=0.35 \
    RADIANCE_MOE_ROUTER=1 RADIANCE_RUN_BWTEST=1

# Fail the build if the native stack does not import, or if a wheel reports a version that does not
# match the source it was built from (a silently mis-stamped wheel is how "aiter 0.0.0" shipped).
# Kept GPU-free: no `import aiter` (it runs rocminfo) and no full `import vllm`; versions come from
# package metadata.
RUN WANT_VLLM=${VLLM_VERSION} WANT_AITER=${AITER_VERSION} \
    python -c 'import os, torch, vllm._C, amdsmi, importlib.metadata as m; \
v, a = m.version("vllm"), m.version("amd-aiter"); \
assert v.startswith(os.environ["WANT_VLLM"]), "vllm wheel reports " + v + ", built tag is " + os.environ["WANT_VLLM"]; \
assert a.startswith(os.environ["WANT_AITER"]), "aiter wheel reports " + a + ", built tag is " + os.environ["WANT_AITER"]; \
print("stack OK | vllm", v, "| torch", torch.__version__, "| aiter", a, \
      "| torchvision", m.version("torchvision"), "| triton", m.version("triton"))'
RUN find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true

ARG RADIANCE_VERSION=0.5.7
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
