#pragma once

#include "FlatDistanceComputer.h"
#include "NeonDotProduct.h"

#include <arm_neon.h>

namespace rnndescent {

struct NeonFP32Kernel : FP32DistanceKernel {
    static constexpr size_t block_elements = 4;

    static float32x4_t zero() { return vdupq_n_f32(0.0f); }
    static float32x4_t load(const float *row) { return vld1q_f32(row); }
    static float32x4_t multiply_add(float32x4_t sum, float32x4_t a, float32x4_t b) { return vfmaq_f32(sum, a, b); }
    static float reduce(float32x4_t sum) { return vaddvq_f32(sum); }
};

using SimdDistanceComputerFP32L2 = FlatDistanceComputer<NeonDotProduct<NeonFP32Kernel>>;

} // namespace rnndescent
