// Census the SHIPPED MXFP4 folded kernel's ISA, to see whether it carries the same epilogue
// address/predication overhead the int4 prefill kernel did.
#include "mxfp4_kernels.h"
template __global__ void radiance_mxfp4_fp8_gemm_folded<2, false>(
    const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *,
    const float *, __bf16 *, int, int, int);
template __global__ void radiance_mxfp4_fp8_gemm_folded<4, false>(
    const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *,
    const float *, __bf16 *, int, int, int);
int main() { return 0; }
