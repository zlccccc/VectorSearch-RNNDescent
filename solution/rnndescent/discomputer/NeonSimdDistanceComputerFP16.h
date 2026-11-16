#pragma once

#include "NeonSimdDistanceComputerFP32.h"

namespace rnndescent {

struct NeonFP16Kernel : NeonFP32Kernel {
    using Code = float16_t;

    static Code encode(float value) { return static_cast<Code>(value); }
    // Keep half-precision storage and accumulate the products in FP32.
    static float32x4_t load(const Code *row) { return vcvt_f32_f16(vld1_f16(row)); }
};

using SimdDistanceComputerFP16L2 = FlatDistanceComputer<NeonDotProduct<NeonFP16Kernel>>;

} // namespace rnndescent
