#pragma once

#include <memory>

// Optional manual overrides for benchmarking / deployment:
//   RNNDESCENT_FORCE_AVX512
//   RNNDESCENT_FORCE_AVX2
//   RNNDESCENT_FORCE_NEON

#if defined(RNNDESCENT_FORCE_AVX512)
#include "Avx512SimdDistanceComputerFP32.h"
#include "Avx512SimdDistanceComputerUInt8.h"
#elif defined(RNNDESCENT_FORCE_AVX2)
#include "Avx2SimdDistanceComputerFP32.h"
#include "Avx2SimdDistanceComputerInt8.h"
#elif defined(RNNDESCENT_FORCE_NEON)
#include "NeonSimdDistanceComputerFP32.h"
#include "NeonSimdDistanceComputerInt8.h"
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include "Avx512SimdDistanceComputerFP32.h"
#include "Avx512SimdDistanceComputerUInt8.h"
#else
#include "Avx2SimdDistanceComputerFP32.h"
#include "Avx2SimdDistanceComputerInt8.h"
#endif
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#include "NeonSimdDistanceComputerFP32.h"
#include "NeonSimdDistanceComputerInt8.h"
#else
#error "Unsupported platform for SIMD distance computer. Please add a fallback implementation."
#endif

namespace rnndescent {
#if defined(RNNDESCENT_FORCE_AVX512) ||                                                                                                                        \
    ((defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && defined(__AVX512F__) && defined(__AVX512BW__))
using SelectedNeighborsContainerType = UInt8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerUInt8L2;
#else
using SelectedNeighborsContainerType = Int8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerInt8L2;
#endif

struct SelectedDistanceComputerFactory {
    static std::unique_ptr<MyDistanceComputer> create_build_graph(const float *data, int n, int d) {
        return std::make_unique<SimdDistanceComputerInt8L2>(data, n, d);
    }

    static std::unique_ptr<MyDistanceComputer> create_cached_graph(const float *data, int n, int d) {
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d);
    }

    static std::unique_ptr<MyDistanceComputer> create_refine_search(const float *data, int n, int d) {
        return std::make_unique<SimdDistanceComputerFP32L2>(data, n, d);
    }
};
} // namespace rnndescent
