// radiance_ar_quant_ext.hip: fp8 (block-scaled e4m3) variant of the one-shot P2P all-reduce for
// dual R9700 (gfx1201/RDNA4, TP=2, PCIe). Companion to radiance_ar_ext.hip: it reuses the same
// IPC scratch, per-block flags and device-resident seq counters that radiance_ar_ext allocates,
// and is dispatched by the same wrapper for large (bandwidth-bound) messages when RADIANCE_AR_QUANT=1.
//
// Design: the PUSH quantizes the local input to block-scaled e4m3 (half the PCIe bytes) and writes
// it into the peer's scratch. After the per-block flag handshake the REDUCE dequantizes the peer
// half from my scratch, recomputes the self half from the local input (deterministic round-to-
// nearest -> reproduces the exact bytes I pushed, so both ranks reduce the identical
// dequant(Q(x0))+dequant(Q(x1)); fp32 add commutes -> bit-identical across ranks). fp32 accumulate,
// bf16/fp16 out.
//
// MUST be built with -ffp-contract=off: otherwise the compiler contracts the reduce's
// dequant(a)*sa + dequant(b)*sb into an FMA that absorbs one product with a single rounding, and
// because the self/peer products are swapped between the two ranks, each rank fuses a different
// multiply and the ranks disagree by ~1 ULP on a few elements.
//
// fp8 payload layout (per double-buffer slot, inside the shared 2*max_bytes scratch):
//   [ e4m3 bytes: n_elem ] ... [ scale_off_bytes ] ... [ fp32 scales: n_groups ]
//   slot stride = max_bytes; scale_off = max_bytes/2 (the fp8 payload is n_elem = nbytes/2 <=
//   max_bytes/2, since the size gate is on the bf16 size). GROUP elements share one scale.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace py = pybind11;

#define RADIANCE_SPIN_MAX 4000000000ULL
#define RADIANCE_MAX_BLOCKS 512
#define GROUP 128                 // elements per block-scaling group
#define FP8E4M3_MAX 448.0f        // OCP e4m3 max (gfx1201 uses the OCP fn variant, not fnuz)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void do_drain(int drain) {
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
}

__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// e4m3 conversion via the public hip_fp8 API (OCP e4m3 on gfx12).
__device__ __forceinline__ unsigned char f2e4m3(float v) { __hip_fp8_e4m3 q(v); return q.__x; }
__device__ __forceinline__ float e4m3_2f(unsigned char b) { __hip_fp8_e4m3 q; q.__x = b; return (float)q; }

// warp-wide absmax; a GROUP maps exactly onto one warp (wave32: GROUP/32=4 elems/lane).
__device__ __forceinline__ float warp_amax(float v) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

// One warp owns one GROUP; lane l owns EPL=GROUP/warpSize contiguous elems [l*EPL, l*EPL+EPL) so
// loads and fp8 stores are coalesced.
template <typename T>
__global__ void radiance_ar_mb_fp8(
    const T* __restrict__ in, char* peer_scratch, const char* my_scratch, T* __restrict__ out,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n_groups, long slot_stride_bytes, long scale_off_bytes, int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid % warpSize, wid = tid / warpSize, nwarp = nt / warpSize;
  const int epl = GROUP / warpSize;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;   // per-block, replay-safe (cudagraph)
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  char* pbase = peer_scratch + (size_t)slot * slot_stride_bytes;
  const char* mbase = my_scratch + (size_t)slot * slot_stride_bytes;
  unsigned char* p_fp8 = (unsigned char*)pbase;
  float* p_scale = (float*)(pbase + scale_off_bytes);
  const unsigned char* m_fp8 = (const unsigned char*)mbase;
  const float* m_scale = (const float*)(mbase + scale_off_bytes);

  const int gpb = (n_groups + nb - 1) / nb;                 // block b owns groups [g0,g1)
  const int g0 = b * gpb; int g1 = g0 + gpb; if (g1 > n_groups) g1 = n_groups;

  // PUSH: quantize local input -> peer scratch. Unrolled 4 groups/warp so 4 PCIe stores are in
  // flight at once (memory-level parallelism), which saturates the PCIe write path; the 4 per-group
  // warp_amax reductions are independent so their shuffle latencies overlap.
  for (int gg = g0 + wid * 4; gg < g1; gg += nwarp * 4) {
    const int ng = (g1 - gg) < 4 ? (g1 - gg) : 4;           // uniform across the warp (no divergence)
    float vv[4][4]; float amx[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; ++j) if (j < ng) {
      const size_t off = (size_t)(gg + j) * GROUP + (size_t)lane * epl;
#pragma unroll
      for (int i = 0; i < epl; ++i) { vv[j][i] = rd_to_f(in[off + i]); amx[j] = fmaxf(amx[j], fabsf(vv[j][i])); }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) amx[j] = warp_amax(amx[j]);
#pragma unroll
    for (int j = 0; j < 4; ++j) if (j < ng) {
      const int grp = gg + j;
      const size_t off = (size_t)grp * GROUP + (size_t)lane * epl;
      const float scale = (amx[j] > 0.f) ? amx[j] * (1.0f / FP8E4M3_MAX) : 1.0f;
      const float inv = 1.0f / scale;
      unsigned int packed = 0u;
#pragma unroll
      for (int i = 0; i < epl; ++i) packed |= (unsigned int)f2e4m3(vv[j][i] * inv) << (8 * i);
      if (epl == 4) *(unsigned int*)(p_fp8 + off) = packed;
      else for (int i = 0; i < epl; ++i) p_fp8[off + i] = (unsigned char)((packed >> (8 * i)) & 0xffu);
      if (lane == 0) p_scale[grp] = scale;
    }
  }

  // handshake (drain the pushes into the fabric, then a per-block release flag + spin)
  do_drain(drain);
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0; while (load_sys_acq(&my_flags[b]) < s) { if (++z > RADIANCE_SPIN_MAX) break; } }
  __syncthreads();
  if (acq) __threadfence_system();

  // REDUCE: dequant(peer) + dequant(recomputed self) -> out, fp32 accumulate.
  for (int g = g0 + wid; g < g1; g += nwarp) {
    const size_t off = (size_t)g * GROUP + (size_t)lane * epl;
    const float psc = m_scale[g];                            // peer half (peer pushed into my scratch)
    unsigned int pk;
    if (epl == 4) pk = *(const unsigned int*)(m_fp8 + off);
    else { pk = 0u; for (int i = 0; i < epl; ++i) pk |= (unsigned int)m_fp8[off + i] << (8 * i); }
    // self half: recompute the SAME quantization from the local input (deterministic RTN + no FP
    // contraction) so it reproduces the exact bytes pushed -> matches the peer's view bit-for-bit.
    float v[4]; float amax = 0.f;
#pragma unroll
    for (int i = 0; i < epl; ++i) { v[i] = rd_to_f(in[off + i]); amax = fmaxf(amax, fabsf(v[i])); }
    amax = warp_amax(amax);
    const float ssc = (amax > 0.f) ? amax * (1.0f / FP8E4M3_MAX) : 1.0f;
    const float sinv = 1.0f / ssc;
#pragma unroll
    for (int i = 0; i < epl; ++i) {
      const float peer_v = e4m3_2f((unsigned char)((pk >> (8 * i)) & 0xffu)) * psc;
      const float self_v = e4m3_2f(f2e4m3(v[i] * sinv)) * ssc;
      rd_from_f(self_v + peer_v, out[off + i]);
    }
  }
}

static int clamp_blocks(int nb, int upper) {
  if (nb < 1) nb = 1; if (nb > upper) nb = upper;
  if (nb > RADIANCE_MAX_BLOCKS) nb = RADIANCE_MAX_BLOCKS; return nb;
}
static int clamp_threads(int nt) { if (nt < 64) nt = 64; if (nt > 1024) nt = 1024; return (nt / 64) * 64; }

// operates on scratch/flags/seq allocated by radiance_ar_ext; slot_stride_bytes = max_bytes,
// scale_off_bytes = max_bytes/2. bf16/fp16 payloads only; n_elem must be a multiple of GROUP.
static void all_reduce_fp8(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                           int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride_bytes,
                           int64_t scale_off_bytes, int64_t inp, int64_t out, int64_t n_elem,
                           int64_t dtype, int64_t stream_i, int64_t nblocks, int64_t nthreads,
                           int64_t drain, int64_t acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (dtype != 0 && dtype != 1) throw std::runtime_error("fp8 ar: only bf16/fp16 payloads");
  if (n_elem % GROUP != 0) throw std::runtime_error("fp8 ar: n_elem not a multiple of GROUP");
  const int n_groups = (int)(n_elem / GROUP);
  const int nb = clamp_blocks((int)nblocks, n_groups);
  const int nt = clamp_threads((int)nthreads);
#define LF(T) radiance_ar_mb_fp8<T><<<nb, nt, 0, st>>>( \
    (const T*)inp, (char*)peer_scratch, (const char*)my_scratch, (T*)out, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n_groups, \
    (long)slot_stride_bytes, (long)scale_off_bytes, (int)drain, (int)acq)
  if (dtype == 0) LF(__hip_bfloat16);
  else LF(__half);
#undef LF
}

PYBIND11_MODULE(radiance_ar_quant_ext, m) {
  m.def("all_reduce_fp8", &all_reduce_fp8);
  m.attr("MAX_BLOCKS") = RADIANCE_MAX_BLOCKS;
  m.attr("GROUP") = GROUP;
}
