#pragma once

#include "FlatDistanceComputer.h"
#include "NeonDotProduct.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>

namespace rnndescent {

struct NeonInt16Kernel {
    using Code = int16_t;
    using Dot = int64_t;
    static constexpr size_t block_elements = 8;
    static constexpr double max_code = 32767.0;
    double center = 0.0;
    double scale = 1.0;
    double scale_squared = 1.0;

    void train(const float *data, size_t count) {
        const auto [min_value, max_value] = std::minmax_element(data, data + count);
        center = (double(*min_value) + double(*max_value)) * 0.5;
        const double radius = (double(*max_value) - double(*min_value)) * 0.5;
        scale = radius > 0.0 ? radius / max_code : 1.0;
        scale_squared = scale * scale;
    }

    Code encode(float value) const {
        const double code = std::round((value - center) / scale);
        return static_cast<Code>(std::clamp(code, -max_code, max_code));
    }

    static int64x2_t zero() { return vdupq_n_s64(0); }
    static int16x8_t load(const Code *row) { return vld1q_s16(row); }

    static int64x2_t multiply_add(int64x2_t sum, int16x8_t a, int16x8_t b) {
        sum = vpadalq_s32(sum, vmull_s16(vget_low_s16(a), vget_low_s16(b)));
        return vpadalq_s32(sum, vmull_high_s16(a, b));
    }

    static int64_t reduce(int64x2_t sum) { return vaddvq_s64(sum); }
};

using SimdDistanceComputerInt16L2 = FlatDistanceComputer<NeonDotProduct<NeonInt16Kernel>>;

} // namespace rnndescent
