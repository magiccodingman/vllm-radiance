// SPDX-License-Identifier: BSD-3-Clause
// Reduction ordering follows PyTorch ATen/native/cuda/Reduce.cuh and
// SharedReduceOps.h (Copyright PyTorch contributors). This is a pinned gfx1201
// candidate for the 5120-wide, FP32-accumulating stock M1 Gemma RMSNorm.
// Compile without FP contraction: stock's square, sums and multiplies are
// separate FP32 operations. No claim of qualification follows from this source.
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float widen(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

static __device__ __forceinline__ uint16_t narrow(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return static_cast<uint16_t>((bits >> 16) | 0x40U);
  return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

template <bool RESIDUAL, int WIDTH, int THREADS>
__global__ __launch_bounds__(THREADS) void stock_m1_gemma_norm(
    const uint16_t *__restrict__ x, const uint16_t *__restrict__ residual,
    const uint16_t *__restrict__ weight, uint16_t *__restrict__ output,
    uint16_t *__restrict__ residual_output, long stride_x, long stride_residual,
    float epsilon, float *moments) {
  const int tid = threadIdx.x, row = blockIdx.x;
  __shared__ float partial[THREADS];
  __shared__ float inverse;
  float accum[4] = {0, 0, 0, 0};
  // Native contiguous FP32 mean: four vector accumulators, block width 512.
  for (int vector = tid; vector < WIDTH / 4; vector += THREADS) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const int column = vector * 4 + component;
      float z = widen(x[row * stride_x + column]);
      if (RESIDUAL)
        z += widen(residual[row * stride_residual + column]);
      const float square = z * z;
      accum[component] += square;
    }
  }
  float total = ((accum[0] + accum[1]) + accum[2]) + accum[3];
  partial[tid] = total;
  // Match native's decreasing inter-wave order and ROCm's increasing
  // intra-wave shuffle order. Every logical row always gets this same layout.
  for (int offset = THREADS / 2; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (tid < offset) {
      total += partial[tid + offset];
      partial[tid] = total;
    }
  }
  __syncthreads();
  if (tid < 32) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
      total += __shfl_down(total, offset, 32);
    if (tid == 0) {
      const float variance = total * (1.0f / static_cast<float>(WIDTH));
      // The pinned PyTorch scalar call is ::rsqrt, not rsqrtf. Preserve its
      // double intermediate explicitly instead of selecting OCML's f32 path.
      inverse = static_cast<float>(::rsqrt(static_cast<double>(variance + epsilon)));
      if (moments) {
        moments[row * 2] = variance;
        moments[row * 2 + 1] = inverse;
      }
    }
  }
  __syncthreads();
  for (int column = tid; column < WIDTH; column += THREADS) {
    float z = widen(x[row * stride_x + column]);
    if (RESIDUAL) {
      z += widen(residual[row * stride_residual + column]);
      residual_output[row * WIDTH + column] = narrow(z);
    }
    const float normalized = z * inverse;
    const float weighted = normalized * (widen(weight[column]) + 1.0f);
    output[row * WIDTH + column] = narrow(weighted);
  }
}

template <int WIDTH, int THREADS>
int launch_norm(
    const void *x, const void *residual, const void *weight, void *output,
    void *residual_output, int rows, long stride_x, long stride_residual,
    float epsilon, void *stream, void *moments) {
  if (rows < 0 || stride_x < WIDTH || (residual && stride_residual < WIDTH) ||
      !x || !weight || !output || (residual && !residual_output))
    return static_cast<int>(hipErrorInvalidValue);
  if (!rows) return 0;
  const dim3 grid(rows), block(THREADS);
  const hipStream_t selected = static_cast<hipStream_t>(stream);
  if (residual)
    hipLaunchKernelGGL((stock_m1_gemma_norm<true, WIDTH, THREADS>), grid, block, 0, selected,
      static_cast<const uint16_t *>(x), static_cast<const uint16_t *>(residual),
      static_cast<const uint16_t *>(weight), static_cast<uint16_t *>(output),
      static_cast<uint16_t *>(residual_output), stride_x, stride_residual, epsilon,
      static_cast<float *>(moments));
  else
    hipLaunchKernelGGL((stock_m1_gemma_norm<false, WIDTH, THREADS>), grid, block, 0, selected,
      static_cast<const uint16_t *>(x), nullptr, static_cast<const uint16_t *>(weight),
      static_cast<uint16_t *>(output), nullptr, stride_x, stride_residual, epsilon,
      static_cast<float *>(moments));
  return static_cast<int>(hipGetLastError());
}

extern "C" int qwen_stock_m1_gemma_norm(
    const void *x, const void *residual, const void *weight, void *output,
    void *residual_output, int rows, long stride_x, long stride_residual,
    float epsilon, void *stream, void *moments, int width, int threads) {
  if (width == 5120 && threads == 512)
    return launch_norm<5120, 512>(x, residual, weight, output, residual_output,
        rows, stride_x, stride_residual, epsilon, stream, moments);
  if (width == 256 && threads == 64)
    return launch_norm<256, 64>(x, residual, weight, output, residual_output,
        rows, stride_x, stride_residual, epsilon, stream, moments);
  if (width == 256 && threads == 32)
    return launch_norm<256, 32>(x, residual, weight, output, residual_output,
        rows, stride_x, stride_residual, epsilon, stream, moments);
  return static_cast<int>(hipErrorInvalidValue);
}
