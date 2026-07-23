// router_gemm.hip - bf16 skinny GEMM for the MoE router (gate) on gfx1201 / RDNA4 (R9700).
// C[M,N] = A[M,K] @ W[N,K]^T bf16 (a torch Linear), for M(=n) <= 16, K=2048, N=256. Pure pybind11 + HIP
// (no torch/extension.h). The gate weight is a ~1 MB read that rocBLAS serves poorly at these skinny
// decode shapes; the kernel is memory-bound and fills the 32 CUs via split-K.
//
// Design:
//  * Split-K within a block: block = WV columns x SK k-splits waves, so WV*SK waves per block fill the
//    machine (one wave per column leaves the 32 CUs under-filled). The SK partials are reduced through a
//    small LDS scratch in-block - no separate reduce kernel, so no extra launch latency - and the reduction
//    is deterministic, keeping the router -> top-k reproducible.
//  * A is read straight from L2 (no LDS staging); the weight is streamed with 128-bit loads issued up-front
//    for maximum in-flight, and the inner product uses the packed v_dot2_f32_bf16 builtin.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
namespace py = pybind11;

#ifndef WAVE
#define WAVE 32          // wave32 (RDNA4); -DWAVE=64 (+ -mwavefrontsize64) for wave64
#endif
#define VEC 8            // bf16 per 128-bit (dwordx4) load
#define MAXM 16          // max supported M (=n)
typedef int int4v __attribute__((ext_vector_type(4)));      // native 128-bit vector
typedef __bf16 bf162 __attribute__((ext_vector_type(2)));   // packed bf16 pair for v_dot2_f32_bf16
// weight load: non-temporal by default (stream, read-once); -DTEMPORAL keeps it cache-resident (the gate
// weights fit the 64 MB Infinity Cache, so cache-resident is faster - the shipped build passes -DTEMPORAL).
#ifdef TEMPORAL
#define WLOAD(p) (*(p))
#else
#define WLOAD(p) __builtin_nontemporal_load(p)
#endif

template <int MT>
__global__ void rgemm(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                      __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
  extern __shared__ float red[];                       // WV*SK*MT SK-reduction scratch
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
    for (int t = 0; t < 8; t++)                          // issue all weight loads up-front (max in-flight)
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
  if (ksplit == 0 && j < N && lane < MT) {              // SK-reduce + write one column of C
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col_local * SK + s) * MT + lane];
    C[(size_t)lane * N + j] = (__hip_bfloat16)v;
  }
}

// grid = ceil(N/WV) blocks; block = WV*SK waves. Compile-time-M dispatch keeps exactly M accumulators.
void launch(int64_t a, int64_t w, int64_t c, int M, int K, int N, int WV, int SK, int64_t stream) {
  if (M > MAXM) throw std::runtime_error("M>MAXM");
  if (K % SK != 0 || (K / SK) % (WAVE * VEC) != 0) throw std::runtime_error("bad SK");
  if (WV * SK * WAVE > 1024) throw std::runtime_error("threads/block>1024");
  dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
  size_t smem = (size_t)WV * SK * M * sizeof(float);
  auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w; auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  #define LAUNCH(MT) hipLaunchKernelGGL(rgemm<MT>, grid, block, smem, st, A, Wp, Cp, K, N, WV, SK)
  switch (M) {
    case 1: LAUNCH(1); break;   case 2: LAUNCH(2); break;   case 3: LAUNCH(3); break;   case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;   case 6: LAUNCH(6); break;   case 7: LAUNCH(7); break;   case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;   case 10: LAUNCH(10); break; case 11: LAUNCH(11); break; case 12: LAUNCH(12); break;
    case 13: LAUNCH(13); break; case 14: LAUNCH(14); break; case 15: LAUNCH(15); break; case 16: LAUNCH(16); break;
    default: throw std::runtime_error("M unsupported");
  }
  #undef LAUNCH
}

PYBIND11_MODULE(router_gemm, m) {
  m.def("launch", &launch, py::arg("a"), py::arg("w"), py::arg("c"), py::arg("M"), py::arg("K"),
        py::arg("N"), py::arg("WV"), py::arg("SK"), py::arg("stream"));
}
