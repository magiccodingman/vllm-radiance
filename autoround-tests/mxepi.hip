// Does the epilogue fast path help the SHIPPED MXFP4 kernel, and is it bit-identical?
//
// The change touches only addressing and predication, so anything other than an EXACT match is a
// bug rather than a tolerance question -- hence a bit-compare of the raw bf16 words, not a
// relative error. Ragged shapes are included deliberately: they are the ones that take the
// fallback path, and a wrong `full` predicate would show up there and nowhere else.
#include "../radiance_autoround_kernels.h"   // HIP_CHECK
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
#include <cstdio>

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 6;
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"},
                          {5120, 5120, "out"}, {200, 640, "ragged"}};
  const int Ms[] = {512, 2048, 4096, 300, 17};
  printf("%-9s %-6s %10s %10s %8s  %s\n", "shape", "M", "old us", "fast us", "gain", "bit-exact");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, Mmax = 4096;
    std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2));
      HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    unsigned char *dA, *dWs, *dWref; float *dAs; __bf16 *dC1, *dC2;
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));      HIP_CHECK(hipMemset(dA, 0x39, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 128, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));
    std::vector<float> hAs(Mmax);
    for (int i = 0; i < Mmax; ++i) hAs[i] = 0.5f + (i % 97) / 100.f;
    HIP_CHECK(hipMemcpy(dAs, hAs.data(), Mmax * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dC1, (size_t)Mmax * N * 2));
    HIP_CHECK(hipMalloc(&dC2, (size_t)Mmax * N * 2));
    for (int M : Ms) {
      if (M > Mmax) continue;
      const int TNsel = (M >= 2048) ? 4 : 2;
      const int BNsel = (TNsel == 4) ? BNF_OF(4) : BNF_OF(2);
      const dim3 g((N + BNsel - 1) / BNsel, (M + BMF - 1) / BMF), b(NTHREADS);
      HIP_CHECK(hipMemset(dC1, 0xAB, (size_t)M * N * 2));
      HIP_CHECK(hipMemset(dC2, 0xAB, (size_t)M * N * 2));
#define RUN(DST, FAST, TN_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<TN_, false, FAST>), \
        g, b, 0, 0, dA, Wm[0], dWs, dWref, dAs, DST, M, N, K)
      if (TNsel == 4) { RUN(dC1, false, 4); RUN(dC2, true, 4); }
      else            { RUN(dC1, false, 2); RUN(dC2, true, 2); }
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned short> h1((size_t)M * N), h2((size_t)M * N);
      HIP_CHECK(hipMemcpy(h1.data(), dC1, h1.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h2.data(), dC2, h2.size() * 2, hipMemcpyDeviceToHost));
      size_t diff = 0;
      for (size_t i = 0; i < h1.size(); ++i) if (h1[i] != h2[i]) ++diff;

      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float bo = 1e30f, bf = 1e30f, ms;
      for (int rep = 0; rep < 3; ++rep) {
#define TIME(BEST, FAST)                                                                   \
        { for (int it = 0; it < 2; ++it) { if (TNsel == 4) RUN(dC1, FAST, 4); else RUN(dC1, FAST, 2); } \
          HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));                 \
          for (int it = 0; it < 10; ++it) { if (TNsel == 4) RUN(dC1, FAST, 4); else RUN(dC1, FAST, 2); } \
          HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));                \
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); BEST = std::min(BEST, ms * 100.f); }
        TIME(bo, false); TIME(bf, true);
#undef TIME
      }
      printf("%-9s %-6d %10.1f %10.1f %7.1f%%  %s\n", sh.name, M, bo, bf, 100.0 * (1 - bf / bo),
             diff ? "MISMATCH" : "yes");
      hipEventDestroy(e0); hipEventDestroy(e1);
#undef RUN
    }
    for (int c = 0; c < NCOPY; ++c) hipFree(Wm[c]);
    hipFree(dA); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC1); hipFree(dC2);
  }
  return 0;
}
