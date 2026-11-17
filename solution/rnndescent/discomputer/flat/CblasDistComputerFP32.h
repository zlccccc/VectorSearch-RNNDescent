#pragma once

#include "FlatDistanceComputer.h"

#include <array>
#include <cblas.h>

namespace rnndescent {

struct CblasFP32Kernel : FP32DistanceKernel {
    static float dot(const float *a, const float *b, size_t dim) { return cblas_sdot(dim, a, 1, b, 1); }

    static std::array<float, 4> dot4(const float *q, const float *x0, const float *x1, const float *x2, const float *x3, size_t dim) {
        return {dot(q, x0, dim), dot(q, x1, dim), dot(q, x2, dim), dot(q, x3, dim)};
    }
};

using CblasDistanceComputerFP32L2 = FlatDistanceComputer<CblasFP32Kernel>;

} // namespace rnndescent
