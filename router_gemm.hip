// router_gemm.hip — memory-bound bf16 skinny GEMM for the MoE router on gfx1201/RDNA4 (R9700).
// C[M,N] = A[M,K] @ W[N,K]^T bf16 (torch Linear). Target M(=n)<=16, K=2048, N=256. Pure pybind11+HIP.
//
// Design (synthesized from wvSplitK + aiter/RDNA4 research):
//  * The kernel is memory-bound on the 1MB weight; the ceiling is machine UNDER-FILL, not bandwidth.
//  * SPLIT-K (primary lever): each (column, k-split) is its own wave -> N*SK waves fill all 32 CUs.
//  * A's matching K-slice staged in a SMALL LDS tile (M*Ks) -> high occupancy; weight streamed with
//    non-temporal dwordx4 loads (read-once, L2-bypass). SK partials reduced deterministically (fp32
//    buffer + reduce kernel) -> bit-reproducible (matters for router->topk).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace py = pybind11;
#ifndef WAVE
#define WAVE 32          // override -DWAVE=64 (+ -mwavefrontsize64) for wave64
#endif
#define VEC 8            // bf16 per int4 (dwordx4) load
#define MAXM 16
#define PAD 8            // LDS bank-conflict pad
typedef int int4v __attribute__((ext_vector_type(4)));   // native 128-bit vec (nontemporal-load capable)
typedef __bf16 bf162 __attribute__((ext_vector_type(2)));   // packed bf16 pair for v_dot2_f32_bf16
// weight load: non-temporal by default (stream, read-once); -DTEMPORAL keeps it cache-resident.
#ifdef TEMPORAL
#define WLOAD(p) (*(p))
#else
#define WLOAD(p) __builtin_nontemporal_load(p)
#endif

// v2 (reference): stage full A[M,K] in LDS, wave-per-column, no split-K.
__global__ void rgemm_v2(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                         __hip_bfloat16* __restrict__ C, int M, int K, int N) {
  extern __shared__ __hip_bfloat16 s2[];
  int tid = threadIdx.x, nthreads = blockDim.x;
  for (int idx = tid * VEC; idx < M * K; idx += nthreads * VEC)
    *(int4*)(s2 + idx) = *(const int4*)(A + idx);
  __syncthreads();
  int wave = tid / WAVE, lane = tid % WAVE, nwaves = nthreads / WAVE;
  const int niter = K / (WAVE * VEC);            // 8 for K=2048
  const int PF = 4;                              // prefetch window depth
  for (int j = blockIdx.x * nwaves + wave; j < N; j += gridDim.x * nwaves) {
    float acc[MAXM]; for (int i = 0; i < MAXM; i++) acc[i] = 0.f;
    const __hip_bfloat16* Wj = W + (size_t)j * K + lane * VEC;
    int4v wbuf[PF];
    #pragma unroll
    for (int p = 0; p < PF; p++) wbuf[p] = *(const int4v*)(Wj + p * WAVE * VEC);
    for (int it = 0; it < niter; it++) {
      int k0 = lane * VEC + it * WAVE * VEC;
      int4v wv = wbuf[it & (PF - 1)];
      int nxt = it + PF;
      if (nxt < niter) wbuf[it & (PF - 1)] = *(const int4v*)(Wj + nxt * WAVE * VEC);
      const __hip_bfloat16* wb = (const __hip_bfloat16*)&wv;
      for (int i = 0; i < M; i++) {
        int4 av = *(const int4*)(s2 + i * K + k0); const __hip_bfloat16* ab = (const __hip_bfloat16*)&av;
        for (int e = 0; e < VEC; e++) acc[i] += (float)ab[e] * (float)wb[e];
      }
    }
    for (int i = 0; i < M; i++) { float v = acc[i];
      for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) C[(size_t)i * N + j] = (__hip_bfloat16)v; }
  }
}

// v4: split-K. grid=(cdiv(N,WV), SK). block=WV waves. Each wave = one column, one k-split.
__global__ void rgemm_v4(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                         float* __restrict__ partial, int M, int K, int N, int SK) {
  extern __shared__ __hip_bfloat16 sA[];       // M*(Ks+PAD)
  int Ks = K / SK, sk = blockIdx.y, kbase = sk * Ks;
  int tid = threadIdx.x, nthreads = blockDim.x;
  int Kp = Ks + PAD;
  for (int idx = tid * VEC; idx < M * Ks; idx += nthreads * VEC) {
    int r = idx / Ks, cc = idx % Ks;
    *(int4*)(&sA[r * Kp + cc]) = *(const int4*)(A + (size_t)r * K + kbase + cc);
  }
  __syncthreads();
  int wave = tid / WAVE, lane = tid % WAVE, nwaves = nthreads / WAVE;
  int j = blockIdx.x * nwaves + wave;
  if (j >= N) return;
  float acc[MAXM]; for (int i = 0; i < MAXM; i++) acc[i] = 0.f;
  const __hip_bfloat16* Wj = W + (size_t)j * K + kbase;
  for (int kk = lane * VEC; kk < Ks; kk += WAVE * VEC) {
    int4v wv = __builtin_nontemporal_load((const int4v*)(Wj + kk));
    const __hip_bfloat16* wb = (const __hip_bfloat16*)&wv;
    for (int i = 0; i < M; i++) {
      int4 av = *(const int4*)(&sA[i * Kp + kk]); const __hip_bfloat16* ab = (const __hip_bfloat16*)&av;
      for (int e = 0; e < VEC; e++) acc[i] += (float)ab[e] * (float)wb[e];
    }
  }
  for (int i = 0; i < M; i++) { float v = acc[i];
    for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    if (lane == 0) partial[(size_t)sk * M * N + (size_t)i * N + j] = v; }
}

__global__ void reduce_sk(const float* __restrict__ partial, __hip_bfloat16* __restrict__ C,
                          int SK, int MN) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= MN) return;
  float v = 0.f;
  for (int s = 0; s < SK; s++) v += partial[(size_t)s * MN + idx];
  C[idx] = (__hip_bfloat16)v;
}

// v6: split-K WITHIN a block (no separate reduce kernel). Block = WV columns x SK k-splits waves; A[M,K]
// staged once in LDS; each wave streams its (column, k-slice) of W; SK partials reduced via a small LDS
// scratch. High occupancy (WV*SK waves/block) + zero reduce-launch overhead. Fits M s.t. M*K*2 < ~63KB.
template <int MT>
__global__ void rgemm_v6(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                         __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
  extern __shared__ char smem[];
  __hip_bfloat16* sA = (__hip_bfloat16*)smem;          // MT*K
  float* red = (float*)(sA + (size_t)MT * K);          // WV*SK*MT
  int tid = threadIdx.x, nthreads = blockDim.x;
  for (int idx = tid * VEC; idx < MT * K; idx += nthreads * VEC)
    *(int4*)(sA + idx) = *(const int4*)(A + idx);
  __syncthreads();
  int wave = tid / WAVE, lane = tid % WAVE;
  int col_local = wave / SK, ksplit = wave % SK;
  int j = blockIdx.x * WV + col_local;
  int Ks = K / SK, kbase = ksplit * Ks;
  float acc[MT];
  #pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  if (j < N) {
    const __hip_bfloat16* Wj = W + (size_t)j * K + kbase + lane * VEC;
    int niter = Ks / (WAVE * VEC);               // <=8
    int4v wbuf[8];
    #pragma unroll
    for (int t = 0; t < 8; t++)                   // issue ALL weight loads up-front (max in-flight)
      if (t < niter) wbuf[t] = WLOAD((const int4v*)(Wj + t * WAVE * VEC));
    for (int t = 0; t < niter; t++) {
      const bf162* wp = (const bf162*)&wbuf[t];
      int kk = lane * VEC + t * WAVE * VEC;
      #pragma unroll
      for (int i = 0; i < MT; i++) {
        int4 av = *(const int4*)(sA + i * K + kbase + kk); const bf162* ap = (const bf162*)&av;
        #pragma unroll
        for (int p = 0; p < VEC / 2; p++)
          acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
      }
    }
    #pragma unroll
    for (int i = 0; i < MT; i++) { float v = acc[i];
      for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v; }
  }
  __syncthreads();
  if (ksplit == 0 && j < N && lane < MT) {
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col_local * SK + s) * MT + lane];
    C[(size_t)lane * N + j] = (__hip_bfloat16)v;
  }
}

// v7: like v6 but A read straight from L2 (no LDS staging, no first barrier) -> works for ALL M, tests
// whether staging/barrier was the bottleneck. LDS only for the tiny SK-reduction scratch.
template <int MT>
__global__ void rgemm_v7(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                         __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
  extern __shared__ float red[];                       // WV*SK*MT
  int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
  int col_local = wave / SK, ksplit = wave % SK;
  int j = blockIdx.x * WV + col_local;
  int Ks = K / SK, kbase = ksplit * Ks;
  float acc[MT];
  #pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  if (j < N) {
    const __hip_bfloat16* Wj = W + (size_t)j * K + kbase + lane * VEC;
    int niter = Ks / (WAVE * VEC);
    int4v wbuf[8];
    #pragma unroll
    for (int t = 0; t < 8; t++)
      if (t < niter) wbuf[t] = WLOAD((const int4v*)(Wj + t * WAVE * VEC));
    for (int t = 0; t < niter; t++) {
      const bf162* wp = (const bf162*)&wbuf[t];
      int kk = kbase + lane * VEC + t * WAVE * VEC;
      #pragma unroll
      for (int i = 0; i < MT; i++) {
        int4 av = *(const int4*)(A + (size_t)i * K + kk); const bf162* ap = (const bf162*)&av;
        #pragma unroll
        for (int p = 0; p < VEC / 2; p++)
          acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
      }
    }
    #pragma unroll
    for (int i = 0; i < MT; i++) { float v = acc[i];
      for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v; }
  }
  __syncthreads();
  if (ksplit == 0 && j < N && lane < MT) {
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col_local * SK + s) * MT + lane];
    C[(size_t)lane * N + j] = (__hip_bfloat16)v;
  }
}

void launch_v7(int64_t a, int64_t w, int64_t c, int M, int K, int N, int WV, int SK, int64_t stream) {
  if (M > MAXM) throw std::runtime_error("M>MAXM");
  if (K % SK != 0 || (K / SK) % (WAVE * VEC) != 0) throw std::runtime_error("bad SK");
  if (WV * SK * WAVE > 1024) throw std::runtime_error("threads/block>1024");
  dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
  size_t smem = (size_t)WV * SK * M * sizeof(float);
  auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w; auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  #define V7(MT) hipLaunchKernelGGL(rgemm_v7<MT>, grid, block, smem, st, A, Wp, Cp, K, N, WV, SK)
  switch (M) {
    case 1: V7(1); break; case 2: V7(2); break; case 3: V7(3); break; case 4: V7(4); break;
    case 5: V7(5); break; case 6: V7(6); break; case 7: V7(7); break; case 8: V7(8); break;
    case 9: V7(9); break; case 10: V7(10); break; case 11: V7(11); break; case 12: V7(12); break;
    case 13: V7(13); break; case 14: V7(14); break; case 15: V7(15); break; case 16: V7(16); break;
    default: throw std::runtime_error("M unsupported");
  }
  #undef V7
}

void launch_v6(int64_t a, int64_t w, int64_t c, int M, int K, int N, int WV, int SK, int64_t stream) {
  if (M > MAXM) throw std::runtime_error("M>MAXM");
  if (K % SK != 0 || (K / SK) % (WAVE * VEC) != 0) throw std::runtime_error("bad SK");
  if (WV * SK * WAVE > 1024) throw std::runtime_error("threads/block>1024");
  dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
  size_t smem = (size_t)M * K * sizeof(__hip_bfloat16) + (size_t)WV * SK * M * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("router_gemm v6: LDS>64KB");
  auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w; auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  #define V6(MT) hipLaunchKernelGGL(rgemm_v6<MT>, grid, block, smem, st, A, Wp, Cp, K, N, WV, SK)
  switch (M) {
    case 1: V6(1); break; case 2: V6(2); break; case 3: V6(3); break; case 4: V6(4); break;
    case 5: V6(5); break; case 6: V6(6); break; case 7: V6(7); break; case 8: V6(8); break;
    case 9: V6(9); break; case 10: V6(10); break; case 11: V6(11); break; case 12: V6(12); break;
    case 13: V6(13); break; case 14: V6(14); break; case 15: V6(15); break; case 16: V6(16); break;
    default: throw std::runtime_error("M unsupported");
  }
  #undef V6
}

void launch(int64_t a, int64_t w, int64_t c, int M, int K, int N,
            int waves_per_block, int grid_blocks, int64_t stream, int ver) {
  if (M > MAXM) throw std::runtime_error("M>MAXM");
  if (K % (WAVE * VEC) != 0) throw std::runtime_error("K%256!=0");
  dim3 block(waves_per_block * WAVE), grid(grid_blocks);
  size_t smem = (size_t)M * K * sizeof(__hip_bfloat16);
  hipLaunchKernelGGL(rgemm_v2, grid, block, smem, (hipStream_t)stream,
                     (const __hip_bfloat16*)a, (const __hip_bfloat16*)w, (__hip_bfloat16*)c, M, K, N);
}

void launch_v4(int64_t a, int64_t w, int64_t partial, int64_t c, int M, int K, int N,
               int waves_per_block, int SK, int64_t stream, int do_reduce) {
  if (M > MAXM) throw std::runtime_error("M>MAXM");
  if (K % SK != 0 || (K / SK) % (WAVE * VEC) != 0) throw std::runtime_error("bad SK");
  int WV = waves_per_block, Ks = K / SK;
  dim3 grid((N + WV - 1) / WV, SK), block(WV * WAVE);
  size_t smem = (size_t)M * (Ks + PAD) * sizeof(__hip_bfloat16);
  if (smem > 64 * 1024) throw std::runtime_error("router_gemm v4: LDS>64KB (raise SK)");
  hipLaunchKernelGGL(rgemm_v4, grid, block, smem, (hipStream_t)stream,
                     (const __hip_bfloat16*)a, (const __hip_bfloat16*)w, (float*)partial, M, K, N, SK);
  if (do_reduce) {
    int MN = M * N;
    hipLaunchKernelGGL(reduce_sk, dim3((MN + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)partial, (__hip_bfloat16*)c, SK, MN);
  }
}

PYBIND11_MODULE(router_gemm, m) {
  m.def("launch", &launch, py::arg("a"), py::arg("w"), py::arg("c"), py::arg("M"), py::arg("K"),
        py::arg("N"), py::arg("waves_per_block"), py::arg("grid_blocks"), py::arg("stream"), py::arg("ver") = 2);
  m.def("launch_v4", &launch_v4, py::arg("a"), py::arg("w"), py::arg("partial"), py::arg("c"),
        py::arg("M"), py::arg("K"), py::arg("N"), py::arg("waves_per_block"), py::arg("SK"),
        py::arg("stream"), py::arg("do_reduce") = 1);
  m.def("launch_v6", &launch_v6, py::arg("a"), py::arg("w"), py::arg("c"), py::arg("M"), py::arg("K"),
        py::arg("N"), py::arg("WV"), py::arg("SK"), py::arg("stream"));
  m.def("launch_v7", &launch_v7, py::arg("a"), py::arg("w"), py::arg("c"), py::arg("M"), py::arg("K"),
        py::arg("N"), py::arg("WV"), py::arg("SK"), py::arg("stream"));
}
