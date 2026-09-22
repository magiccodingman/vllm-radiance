// SPDX-License-Identifier: Apache-2.0
// Shape-specialized gfx1x BF16 wvSplitK arithmetic from vLLM skinny_gemms.cu.
// Preserve the M4 dot-product and reduction order. Interleave the two M4
// groups in one launch so adjacent workgroups visit the same vocabulary rows.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "This experiment admits gfx1201 only"
#endif

namespace {
constexpr int K = 5120;
constexpr int M = 248320;
constexpr int N = 4;
constexpr int Threads = 32;
constexpr int Waves = 16;
constexpr int Tile = 4;
constexpr int Chunk = 8;
constexpr int CUs = 32;
using Vector = __attribute__((vector_size(16))) float;
union Packed {
  __hip_bfloat16 h[Chunk];
  float f[Chunk / 2];
  Vector vector;
};

__global__ void __launch_bounds__(Threads * Waves)
stock_m1_head_pair(const __hip_bfloat16* __restrict__ weights,
                   const __hip_bfloat16* __restrict__ hidden,
                   __hip_bfloat16* __restrict__ output) {
  const int group = blockIdx.x % 2;
  const int tile = blockIdx.x / 2;
  hidden += group * N * K;
  output += group * N * M;
  // Same LDS occupancy as the qualified M4 operator.
  __shared__ __hip_bfloat16 s[32768];
  for (uint32_t k = (threadIdx.y * Threads + threadIdx.x) * Chunk;
       k < K * N; k += Threads * Waves * Chunk) {
    *reinterpret_cast<Packed*>(&s[k]) = *reinterpret_cast<const Packed*>(&hidden[k]);
  }
  __syncthreads();
  for (uint32_t m = (tile * Waves + threadIdx.y) * Tile; m < M;
       m += CUs * Waves * Tile) {
    float sum[N][Tile] = {};
    for (uint32_t k1 = 0; k1 < K; k1 += Threads * Chunk) {
      const uint32_t k = k1 + threadIdx.x * Chunk;
      Packed a[N];
      Packed b[Tile];
      for (int y = 0; y < Tile; ++y)
        b[y].vector = __builtin_nontemporal_load(
            reinterpret_cast<const Vector*>(&weights[(m + y) * K + k]));
      for (int n = 0; n < N; ++n)
        a[n] = *reinterpret_cast<const Packed*>(&s[k + n * K]);
      for (int n = 0; n < N; ++n) {
        for (int y = 0; y < Tile; ++y) {
          for (int pair = 0; pair < Chunk / 2; ++pair) {
            float2 product =
                __bfloat1622float2(*reinterpret_cast<__hip_bfloat162*>(&a[n].f[pair])) *
                __bfloat1622float2(*reinterpret_cast<__hip_bfloat162*>(&b[y].f[pair]));
            sum[n][y] += product.x + product.y;
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int n = 0; n < N; ++n) {
      for (int y = 0; y < Tile; ++y) {
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
    if (threadIdx.x == Threads - 1) {
      for (int n = 0; n < N; ++n)
        for (int y = 0; y < Tile; ++y)
          output[m + y + n * M] = __float2bfloat16(sum[n][y] + 0.0f);
    }
  }
}
}  // namespace

extern "C" int qwen_stock_m1_head_pair(const void* weights, const void* hidden,
                                       void* output, void* stream) {
  if (!weights || !hidden || !output) return int(hipErrorInvalidValue);
  stock_m1_head_pair<<<dim3(2 * CUs), dim3(Threads, Waves), 0,
                      reinterpret_cast<hipStream_t>(stream)>>>(
      static_cast<const __hip_bfloat16*>(weights),
      static_cast<const __hip_bfloat16*>(hidden),
      static_cast<__hip_bfloat16*>(output));
  return int(hipGetLastError());
}
