// Qualification candidate: causal prefill using R4D's unfused M1 arithmetic.
// q/k/v are raw BF16 convolution outputs; g/beta are per-token FP32 gates.
// No cumulative gates, future-dependent rescaling, WY solve, or dispatch by
// whole-sequence decay span. Compile with -ffp-contract=off (as R4D M1 does).
// Initial and final state are separate, packed [sequence, head, V, K] arrays.
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float from_bf16(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

static __device__ __forceinline__ uint16_t to_native_bf16(float value) {
  // R4D M1's declared f2bf conversion rounds halfway away from zero.
  // Preserve that existing arithmetic here; it is NOT an RTNE certificate.
  return static_cast<uint16_t>((__float_as_uint(value) + 0x8000U) >> 16);
}

template <int SPLITS, bool PREPARED>
__global__ __launch_bounds__(256 / SPLITS) void causal_prefill(
    const void *__restrict__ q, const void *__restrict__ k,
    const uint16_t *__restrict__ v, const float *__restrict__ gates,
    const float *__restrict__ beta, const float *__restrict__ initial,
    uint16_t *__restrict__ output, float *__restrict__ final,
    const int *__restrict__ cu, int heads, int query_heads, float scale) {
  const int sequence = blockIdx.x, head = blockIdx.y;
  const int begin = cu[sequence], end = cu[sequence + 1];
  const int row = (threadIdx.x >> 1) + blockIdx.z * (128 / SPLITS);
  const int half = threadIdx.x & 1, start_k = half * 64;
  const int query_head = head / (heads / query_heads);
  const size_t state_offset = ((size_t)sequence * heads + head) * 128 * 128 +
                              row * 128 + start_k;
  float state[64];
#pragma unroll
  for (int j = 0; j < 64; ++j)
    state[j] = initial[state_offset + j];
  for (int token = begin; token < end; ++token) {
    const size_t gate_offset = (size_t)token * heads + head;
    const size_t qk_offset = ((size_t)token * query_heads + query_head) * 128 + start_k;
    float queries[64], keys[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      queries[j] = PREPARED ? ((const float *)q)[qk_offset + j]
                            : from_bf16(((const uint16_t *)q)[qk_offset + j]);
      keys[j] = PREPARED ? ((const float *)k)[qk_offset + j]
                         : from_bf16(((const uint16_t *)k)[qk_offset + j]);
    }
    if (!PREPARED) {
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      sq += queries[j] * queries[j];
      sk += keys[j] * keys[j];
    }
    sq += __shfl_xor(sq, 1, 32);
    sk += __shfl_xor(sk, 1, 32);
    const float qs = __frsqrt_rn(sq + 1e-6f) * scale;
    const float ks = __frsqrt_rn(sk + 1e-6f);
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      queries[j] *= qs;
      keys[j] *= ks;
    }
    }
    const float decay = __expf(gates[gate_offset]);
    float prediction = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      state[j] *= decay;
      prediction += state[j] * keys[j];
    }
    prediction += __shfl_xor(prediction, 1, 32);
    const float residual = (from_bf16(v[gate_offset * 128 + row]) - prediction) *
                           beta[gate_offset];
    float value = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      state[j] += residual * keys[j];
      value += state[j] * queries[j];
    }
    value += __shfl_xor(value, 1, 32);
    if (half == 0)
      output[gate_offset * 128 + row] = to_native_bf16(value);
  }
  // Empty sequences are identity transitions too: never leave final uninitialized.
#pragma unroll
  for (int j = 0; j < 64; ++j)
    final[state_offset + j] = state[j];
}

template <bool PREPARED> static int launch_prefill(
    const void *q, const void *k, const void *v, const void *g, const void *beta,
    const void *initial, void *output, void *final, const void *cu,
    int sequences, int heads, int query_heads, float scale, void *stream) {
  if (sequences < 0 || heads <= 0 || query_heads <= 0 || heads % query_heads)
    return static_cast<int>(hipErrorInvalidValue);
  if (sequences == 0)
    return 0;
  if (!initial || !final || !cu || initial == final)
    return static_cast<int>(hipErrorInvalidValue);
  const int want = (64 + sequences * heads - 1) / (sequences * heads);
  const int splits = want >= 4 ? 4 : (want >= 2 ? 2 : 1);
#define LAUNCH(S) \
  hipLaunchKernelGGL((causal_prefill<S, PREPARED>), dim3(sequences, heads, S), 256 / S, 0, \
      (hipStream_t)stream, q, k, \
      (const uint16_t *)v, (const float *)g, (const float *)beta, \
      (const float *)initial, (uint16_t *)output, (float *)final, \
      (const int *)cu, heads, query_heads, scale)
  if (splits == 4) { LAUNCH(4); }
  else if (splits == 2) { LAUNCH(2); }
  else { LAUNCH(1); }
#undef LAUNCH
  return static_cast<int>(hipGetLastError());
}

// Precompute each normalization once instead of repeating it for all 128 value
// rows. The two lanes and their 64 ordered terms are identical to native M1.
__global__ __launch_bounds__(256) void normalize_qk(
    const uint16_t *__restrict__ q, const uint16_t *__restrict__ k,
    float *__restrict__ out_q, float *__restrict__ out_k, int rows, float scale) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 2;
  if (row >= rows) return;
  const size_t offset = (size_t)row * 128 + (threadIdx.x & 1) * 64;
  float qq[64], kk[64], sq = 0.0f, sk = 0.0f;
#pragma unroll
  for (int j = 0; j < 64; ++j) {
    qq[j] = from_bf16(q[offset + j]);
    kk[j] = from_bf16(k[offset + j]);
    sq += qq[j] * qq[j];
    sk += kk[j] * kk[j];
  }
  sq += __shfl_xor(sq, 1, 32);
  sk += __shfl_xor(sk, 1, 32);
  const float qs = __frsqrt_rn(sq + 1e-6f) * scale;
  const float ks = __frsqrt_rn(sk + 1e-6f);
#pragma unroll
  for (int j = 0; j < 64; ++j) {
    out_q[offset + j] = qq[j] * qs;
    out_k[offset + j] = kk[j] * ks;
  }
}

extern "C" int qwen_gdn_normalize_qk(const void *q, const void *k, void *out_q,
    void *out_k, int rows, float scale, void *stream) {
  if (rows < 0) return static_cast<int>(hipErrorInvalidValue);
  if (!rows) return 0;
  normalize_qk<<<(rows + 127) / 128, 256, 0, (hipStream_t)stream>>>(
      (const uint16_t *)q, (const uint16_t *)k, (float *)out_q, (float *)out_k, rows, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int qwen_gdn_causal_prefill(
    const void *q, const void *k, const void *v, const void *g, const void *beta,
    const void *initial, void *output, void *final, const void *cu,
    int sequences, int heads, int query_heads, float scale, void *stream) {
  return launch_prefill<false>(q, k, v, g, beta, initial, output, final, cu,
                               sequences, heads, query_heads, scale, stream);
}

extern "C" int qwen_gdn_causal_prefill_prepared(
    const void *q, const void *k, const void *v, const void *g, const void *beta,
    const void *initial, void *output, void *final, const void *cu,
    int sequences, int heads, int query_heads, float scale, void *stream) {
  return launch_prefill<true>(q, k, v, g, beta, initial, output, final, cu,
                              sequences, heads, query_heads, scale, stream);
}
