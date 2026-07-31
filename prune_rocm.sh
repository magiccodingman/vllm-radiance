#!/usr/bin/env bash
# Strip the ROCm install down to what this image actually runs on: one GPU architecture.
#
# The stock rocm/dev-ubuntu-24.04:*-full tree is ~19 GB, and most of it is device code for GPUs
# this image cannot run on (it is compiled for gfx1201 only) or link-time-only archives. Pruning
# it is worth ~13 GB uncompressed, which is most of the image's download size.
#
# This MUST run in a stage whose /opt/rocm is then COPYed into the final image: deleting files in
# a layer on top of the base reclaims nothing, it only records whiteouts.
#
# What is deliberately KEPT:
#   * clang's compiler-rt archives (llvm/lib/clang/**). hipcc links every HIP object against
#     libclang_rt.builtins; AITER JIT-compiles kernels at RUNTIME, so the compiler has to keep
#     working inside the shipped image. Removing these breaks the JIT, not the build.
#   * the device bitcode libraries (amdgcn/**.bc) hipcc needs for the same reason.
#   * every arch-neutral file in the rocBLAS/hipBLASLt library dirs (logic/manifest metadata).
set -eu

GFX="${1:?usage: prune_rocm.sh <gfx-arch>}"
R=/opt/rocm

mb() { local s=0; [ "$#" -gt 0 ] && s=$(du -xc --apparent-size "$@" 2>/dev/null | tail -1 | cut -f1); echo $((s/1024)); }
before=$(du -xs "$R/" 2>/dev/null | cut -f1)

# 1. static archives: link-time only, never loaded at runtime -- except clang's own runtime libs.
mapfile -t archives < <(find "$R/" -name '*.a' -type f ! -path '*/llvm/lib/clang/*')
echo "  -$(mb "${archives[@]}") MB  static archives ($(( ${#archives[@]} )) files, clang runtime kept)"
if [ "${#archives[@]}" -gt 0 ]; then printf '%s\0' "${archives[@]}" | xargs -0 rm -f; fi

# 2. MIOpen convolution kernel libraries for other architectures (~370 MB each).
mapfile -t miopen < <(find "$R/" -name 'libMIOpenCKGroupedConv_gfx*.so' -type f ! -name "*${GFX}*")
echo "  -$(mb "${miopen[@]}") MB  MIOpen conv kernels for other archs ($(( ${#miopen[@]} )) libs)"
if [ "${#miopen[@]}" -gt 0 ]; then printf '%s\0' "${miopen[@]}" | xargs -0 rm -f; fi

# 3. MIOpen perf databases for other architectures.
mapfile -t miodb < <(find "$R/" \( -name '*.kdb*' -o -name '*.db' -o -name '*.db.txt' \) -type f ! -name "*${GFX}*")
echo "  -$(mb "${miodb[@]}") MB  MIOpen perf databases for other archs"
if [ "${#miodb[@]}" -gt 0 ]; then printf '%s\0' "${miodb[@]}" | xargs -0 rm -f; fi

# 4. hipBLASLt kernels: one directory per arch (gfx942 alone is 1.2 GB).
mapfile -t hbl < <(find "$R/lib/hipblaslt/library" -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name "$GFX" 2>/dev/null)
echo "  -$(mb "${hbl[@]}") MB  hipBLASLt kernel dirs for other archs ($(( ${#hbl[@]} )) archs)"
if [ "${#hbl[@]}" -gt 0 ]; then printf '%s\0' "${hbl[@]}" | xargs -0 rm -rf; fi

# 5. rocBLAS Tensile blobs: flat files tagged with the arch in the filename.
mapfile -t rb < <(find "$R/lib/rocblas/library" -type f -name '*gfx*' ! -name "*${GFX}*" 2>/dev/null)
echo "  -$(mb "${rb[@]}") MB  rocBLAS Tensile blobs for other archs ($(( ${#rb[@]} )) files)"
if [ "${#rb[@]}" -gt 0 ]; then printf '%s\0' "${rb[@]}" | xargs -0 rm -f; fi

after=$(du -xs "$R/" 2>/dev/null | cut -f1)
echo "  ROCm: $((before/1024)) MB -> $((after/1024)) MB (saved $(( (before-after)/1024 )) MB)"

# Fail loudly if the prune ate something this image needs: the arch's own kernels must survive,
# and hipcc must still be able to compile AND LINK a HIP shared object (the AITER JIT path).
test -f "$R/lib/libMIOpenCKGroupedConv_${GFX}.so" || { echo "FATAL: pruned this arch's MIOpen lib"; exit 1; }
test -d "$R/lib/hipblaslt/library/${GFX}"         || { echo "FATAL: pruned this arch's hipBLASLt kernels"; exit 1; }
ls "$R/lib/rocblas/library/" | grep -q "$GFX"     || { echo "FATAL: pruned this arch's rocBLAS kernels"; exit 1; }

cat >/tmp/_jit_probe.hip <<'EOF'
#include <hip/hip_runtime.h>
__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }
extern "C" void launch(float* o) { hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, o); }
EOF
"$R/bin/hipcc" -O3 -fPIC -shared --offload-arch="${GFX}" /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
  || { echo "FATAL: hipcc can no longer build a HIP shared object -- the AITER runtime JIT would break"; exit 1; }
rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so
echo "  hipcc still compiles+links HIP shared objects (AITER JIT path intact)"
