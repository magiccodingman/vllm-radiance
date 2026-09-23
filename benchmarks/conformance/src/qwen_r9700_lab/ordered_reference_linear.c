#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <fenv.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "Requires IEEE binary32");

/* Each output follows the Python reference's increasing-k multiply/add order.
 * No BLAS, fused multiply-add, reassociation or cross-output reduction.
 * The caller owns disjoint, contiguous arrays and validates all lengths.
 */
int qwen_ordered_linear(const float *x, const float *w, float *out,
                        size_t batches, size_t rows, size_t width) {
    if (fegetround() != FE_TONEAREST) return 1;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (size_t k = 0; k < width; ++k) {
                float product = x[batch * width + k] * w[row * width + k];
                sum = sum + product;
            }
            out[batch * rows + row] = sum;
        }
    }
    return 0;
}
