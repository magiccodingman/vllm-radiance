# syntax=docker/dockerfile:1.7
# vllm-radiance: vLLM/torch/triton/aiter stack for RDNA4 (gfx1201 / R9700), plus the radiance
# patches and kernels. Single multistage build on the official AMD ROCm image, in four stages:
#   1. builder    compile torch/triton/torchvision/aiter/vLLM from source into /wheels
#   2. rocmprune  cut the 19 GB ROCm tree down to this one GPU architecture
#   3. assemble   install the wheels, apply the patches, compile the HIP kernels
#   4. final      the release image: a clean Ubuntu with only the pruned ROCm and the venv
# No prebuilt component wheels and no checked-in binaries. The release image carries neither the
# build toolchain nor the wheels, which is most of the reason it is far smaller than the base.
#
# stack: AMD torch 2.12, AMD Triton 3.7.1, torchvision 0.27.1, AITER 0.1.20,
# pinned vLLM main,
# all compiled for PYTORCH_ROCM_ARCH=gfx1201 against the base image's ROCm 7.14.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7
ARG GFX_ARCH=gfx1201
# The release stage starts from a clean distro image rather than the ROCm base, and COPYs in only
# the pruned ROCm tree plus the venv. Same Ubuntu release as the ROCm base (24.04), so the venv's
# interpreter (python 3.12.3) matches.
ARG RELEASE_BASE=ubuntu:24.04@sha256:a08e551cb33850e4740772b38217fc1796a66da2506d312abe51acda354ff061

# Component pins, in one place. Each is both the git tag that gets compiled and the version the
# resulting wheel reports, so `pip show`, `importlib.metadata`, and the startup banner all agree
# with what was actually built.
# Treat the compiler stack as one atomic upstream-tested ROCm unit. These are
# the exact AMD commits used by pinned vLLM main's Dockerfile.rocm_base, not the
# generic PyTorch 2.13 / upstream Triton 3.7.1 combination that hung TP2 Radiance.
ARG TORCH_REPO=https://github.com/ROCm/pytorch.git
ARG TORCH_COMMIT=6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5
ARG TORCH_VERSION=2.12.0
ARG TRITON_REPO=https://github.com/ROCm/triton.git
ARG TRITON_COMMIT=f0b55c07da61c71775bef6d1a15ebf846430ac75
ARG TRITON_VERSION=3.7.1
ARG TORCHVISION_VERSION=0.27.1
ARG AITER_COMMIT=fc2e5d57fb5b8ad8e7e23f7103071dde798ea618
ARG AITER_VERSION=0.1.20
# Never float main: this is the exact 2026-08-22 tree containing the RDNA4
# routing work and DFlash2. VLLM_VERSION is only the deterministic wheel stamp.
ARG VLLM_COMMIT=a014e35f38c80fb0652387740193ad2147fed6a3
ARG VLLM_VERSION=0.28.0.dev0+a014e35
# rocm-bandwidth-test for the startup topology/bandwidth sweep. Pinned to the NEWEST tag that still
# has a plain CMakeLists: the rocm-7.x tags moved to a cmake framework that demands clang>=19 on PATH
# plus vendored boost/fmt/curl submodules, none of which this tool needs.
ARG RBT_VERSION=rocm-6.4.4

# =====================================================================================
# STAGE 1 builder: compile the stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    USE_ROCM=1 USE_CUDA=0 \
    MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAKEFLAGS=-j${BUILD_JOBS} CARGO_BUILD_JOBS=${BUILD_JOBS} \
    CCACHE_MAXSIZE=4G CCACHE_COMPRESS=1

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
ARG TORCH_REPO
ARG TORCH_COMMIT
ARG TORCH_VERSION
RUN --mount=type=cache,id=radiance-ccache-gfx1201,target=/root/.cache/ccache \
    git clone --filter=blob:none --no-checkout ${TORCH_REPO} /src/pytorch \
    && cd /src/pytorch \
    && git checkout --detach ${TORCH_COMMIT} \
    && test "$(git rev-parse HEAD)" = "${TORCH_COMMIT}" \
    && git submodule update --init --recursive --depth 1 \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       CC=gcc-13 CXX=g++-13 \
       PYTORCH_BUILD_VERSION=${TORCH_VERSION}+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       python setup.py bdist_wheel \
    && cp dist/*.whl /wheels/ && pip install dist/*.whl && rm -rf /src/pytorch

# --- triton ---
ARG TRITON_REPO
ARG TRITON_COMMIT
RUN --mount=type=cache,id=radiance-ccache-gfx1201,target=/root/.cache/ccache \
    git clone --filter=blob:none --no-checkout ${TRITON_REPO} /src/triton \
    && cd /src/triton \
    && git checkout --detach ${TRITON_COMMIT} \
    && test "$(git rev-parse HEAD)" = "${TRITON_COMMIT}" \
    && TRITON_BUILD_PROTON=OFF pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- torchvision ---
# FORCE_CUDA=1 is REQUIRED: torchvision's BUILD_CUDA_SOURCES gates on torch.cuda.is_available(),
# which is false in `docker build` (no GPU) -> it picks CppExtension, where torch's build-time hipify
# double-compiles vision.cpp + vision_hip.cpp -> "multiple definition of vision::cuda_version()".
# FORCE_CUDA=1 forces CUDAExtension (correct hipify source replacement); hipcc needs no GPU to compile.
ARG TORCHVISION_VERSION
RUN --mount=type=cache,id=radiance-ccache-gfx1201,target=/root/.cache/ccache \
    git clone --depth 1 -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- aiter (gfx1201; kernels JIT at runtime, PREBUILD_KERNELS=0) ---
# PRETEND_VERSION gives the exact release label to the detached commit.
ARG AITER_COMMIT
ARG AITER_VERSION
RUN git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter \
    && git checkout --detach ${AITER_COMMIT} \
    && test "$(git rev-parse HEAD)" = "${AITER_COMMIT}" \
    && git submodule update --init --recursive --depth 1 \
    && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1 \
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
ARG VLLM_COMMIT
ARG VLLM_VERSION
RUN --mount=type=cache,id=radiance-ccache-gfx1201,target=/root/.cache/ccache \
    git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && git checkout --detach ${VLLM_COMMIT} \
    && test "$(git rev-parse HEAD)" = "${VLLM_COMMIT}" \
    && python use_existing_torch.py \
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
    && cmake --build /src/rbt/build -j ${BUILD_JOBS} \
    && mkdir -p /artifacts && cp /src/rbt/build/rocm-bandwidth-test /artifacts/ \
    && rm -rf /src/rbt

# =====================================================================================
# STAGE 2 rocmprune: cut the ROCm tree down to this image's single GPU architecture
# =====================================================================================
# ~19 GB of the base is device code for GPUs this image cannot run on, plus link-time-only
# archives. Pruning has to happen in a stage that the release stage COPYs FROM: deleting files
# in a layer stacked on the base reclaims nothing, it only writes whiteouts. See prune_rocm.sh
# for what is kept and why (the runtime still has to compile: AITER JITs kernels on first use).
FROM ${ROCM_BASE} AS rocmprune
ARG GFX_ARCH
COPY prune_rocm.sh /tmp/prune_rocm.sh
RUN bash /tmp/prune_rocm.sh ${GFX_ARCH} && rm -f /tmp/prune_rocm.sh

# =====================================================================================
# STAGE 3 assemble: install the wheels and apply the patches and kernels
# =====================================================================================
# Runs on the FULL base because it needs the toolchain (hipcc, headers, static archives) to
# compile the HIP kernels. Only the resulting /opt/vllm venv is carried into the release image.
FROM ${ROCM_BASE} AS assemble
ARG GFX_ARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv \
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
             patch_gdn_wmma patch_preshuffle install_radiance_hooks \
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

# --- strip debug symbols from the installed extensions (worth ~1 GB) ---
# These are release builds, but they still carry .debug_* sections that nothing reads at runtime.
# Our three HIP kernels are excluded: they are tiny and carry device fatbins.
RUN find /opt/vllm -type f -name '*.so*' ! -name 'radiance_ar*' ! -name 'router_gemm*' \
      -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true; \
    echo "extensions stripped"

# =====================================================================================
# STAGE 4 final: the release image -- a clean Ubuntu with only what is needed to serve
# =====================================================================================
# Built by COPYing an allowlist rather than by inheriting the ROCm base, which is what makes the
# prune above pay: the release image never contains the 19 GB tree, the build toolchain, the
# wheels, or the patch sources. Only the pruned ROCm, the venv, and the entrypoint come across.
FROM ${RELEASE_BASE} AS final
ARG GFX_ARCH
ENV DEBIAN_FRONTEND=noninteractive
# ROCm 7.14 vendors its own libdrm / numa / elf / sqlite / zlib / zstd (the librocm_sysdeps_* set),
# so the release image needs very little from the distro:
#   python3.12     the interpreter the /opt/vllm venv was built against (Ubuntu 24.04 ships 3.12.3)
#   libnuma-dev    rocSHMEM dlopen()s the UNVERSIONED libnuma.so, which only the -dev package ships
#   numactl        optional --numa-bind;  curl  the compose healthcheck runs it inside the container
#   g++            NOT optional: AITER JIT-compiles its kernels on FIRST USE, inside this image, and
#                  hipcc needs the C++ standard headers (and the same g++ major torch was built
#                  with). Without it every JIT build dies with "Could not find standard C++ header
#                  'cmath'", aiter's flag probes all fail (including --offload-arch), and the engine
#                  crashes. The build-time probe below is what keeps this honest.
#   python3.12-dev Python.h, for the pybind11 modules aiter JIT-builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3.12 python3.12-dev g++ libnuma-dev numactl curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /opt/rocm is a symlink farm pointing through /etc/alternatives into core-<ver>, so both have to
# come across or nothing resolves. The pruned tree is stable across radiance releases, which keeps
# it a cached layer users do not re-download for every version bump.
COPY --from=rocmprune /opt/rocm /opt/rocm
COPY --from=rocmprune /etc/alternatives /etc/alternatives
COPY --from=assemble /opt/vllm /opt/vllm
COPY --from=builder /artifacts/rocm-bandwidth-test /usr/local/bin/rocm-bandwidth-test

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV VIRTUAL_ENV=/opt/vllm \
    PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

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
# Running this in the RELEASE stage also proves the allowlist above is complete: a library left
# behind by the prune or by the slim base shows up here as an ImportError, not in production.
# Kept GPU-free: no `import aiter` (it runs rocminfo) and no full `import vllm`; versions come from
# package metadata. The radiance kernels are imported after torch, which is what loads libamdhip64.
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TORCH_VERSION
ARG TRITON_VERSION
ARG TORCHVISION_VERSION
RUN WANT_VLLM=${VLLM_VERSION} WANT_AITER=${AITER_VERSION} WANT_TORCH=${TORCH_VERSION} \
    WANT_TRITON=${TRITON_VERSION} WANT_VISION=${TORCHVISION_VERSION} \
    python -c 'import os, torch, vllm._C, amdsmi, importlib.metadata as m; \
import radiance_ar_ext, radiance_ar_quant_ext, router_gemm; \
v, a, t, r, tv = m.version("vllm"), m.version("amd-aiter"), m.version("torch"), m.version("triton"), m.version("torchvision"); \
assert v.startswith(os.environ["WANT_VLLM"]), "vllm wheel reports " + v + ", built tag is " + os.environ["WANT_VLLM"]; \
assert a.startswith(os.environ["WANT_AITER"]), "aiter wheel reports " + a + ", built tag is " + os.environ["WANT_AITER"]; \
assert t.startswith(os.environ["WANT_TORCH"]), "torch wheel reports " + t; \
assert r.startswith(os.environ["WANT_TRITON"]), "triton wheel reports " + r; \
assert tv.startswith(os.environ["WANT_VISION"]), "torchvision wheel reports " + tv; \
print("stack OK | vllm", v, "| torch", torch.__version__, "| aiter", a, \
      "| torchvision", m.version("torchvision"), "| triton", m.version("triton"))'

# The release image must still be able to COMPILE. AITER JIT-builds its kernels on first use, as a
# pybind11 HIP extension, so the shipped image needs hipcc AND the C++ standard headers AND Python.h.
# `hipcc --version` does not prove any of that -- it passes on an image whose JIT is broken, which is
# exactly how a slim release stage shipped with no libstdc++ headers. This mirrors aiter's real
# compile: build a pybind11 HIP module that includes <cmath>, then import it and call into it.
# torch is imported first because that is what pulls libamdhip64 into the process -- a bare HIP
# extension cannot resolve it on its own (no ROCm entry in ld.so.conf), here or in any prior release.
RUN printf '%s\n' \
      '#include <hip/hip_runtime.h>' \
      '#include <cmath>' \
      '#include <pybind11/pybind11.h>' \
      '__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }' \
      'PYBIND11_MODULE(_jit_probe, m) { m.def("f", [](double x) { return std::sqrt(x); }); }' \
      > /tmp/_jit_probe.hip \
 && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
 && python -c "import torch, sys; sys.path.insert(0, '/tmp'); import _jit_probe; assert _jit_probe.f(4.0) == 2.0" \
 && rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so \
 && echo "runtime JIT toolchain OK (hipcc + libstdc++ headers + Python.h + pybind11)"

ARG RADIANCE_VERSION=0.6.0-dev.a014e35
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
