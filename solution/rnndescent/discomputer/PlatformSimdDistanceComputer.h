#pragma once

#include <memory>

// Optional manual overrides for benchmarking / deployment:
//   RNNDESCENT_FORCE_AVX512
//   RNNDESCENT_FORCE_AVX2
//   RNNDESCENT_FORCE_AVX2_UINT8
//   RNNDESCENT_FORCE_AVX_VNNI
//   RNNDESCENT_FORCE_NEON

#if defined(RNNDESCENT_FORCE_AVX_VNNI)
#include "Avx2SimdDistanceComputerFP32.h"
#include "AvxVnniSimdDistanceComputerUInt8.h"
#elif defined(RNNDESCENT_FORCE_AVX2_UINT8)
#include "Avx2SimdDistanceComputerFP32.h"
#include "Avx2SimdDistanceComputerUInt8.h"
#elif defined(RNNDESCENT_FORCE_AVX512) || \
    (!defined(RNNDESCENT_FORCE_AVX2) && !defined(RNNDESCENT_FORCE_NEON) && \
     defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__))
#define RNNDESCENT_USE_AVX512
#include "Avx512SimdDistanceComputerFP32.h"
#include "Avx512SimdDistanceComputerUInt8.h"
#elif defined(RNNDESCENT_FORCE_AVX2)
#include "Avx2SimdDistanceComputerFP32.h"
#include "Avx2SimdDistanceComputerInt8.h"
#elif defined(RNNDESCENT_FORCE_NEON)
#include "NeonSimdDistanceComputerFP32.h"
#include "NeonSimdDistanceComputerInt8.h"
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "Avx2SimdDistanceComputerFP32.h"
#include "Avx2SimdDistanceComputerInt8.h"
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#include "NeonSimdDistanceComputerFP32.h"
#include "NeonSimdDistanceComputerInt8.h"
#else
#error "Unsupported platform for SIMD distance computer. Please add a fallback implementation."
#endif

namespace rnndescent {
#if defined(RNNDESCENT_FORCE_AVX_VNNI)
using SelectedNeighborsContainerType = QuantizedNeighbors<AvxVnniUInt8DotProduct>;
using SelectedSaveNeighborDiscomputer = AvxVnniSimdDistanceComputerUInt8L2;
#elif defined(RNNDESCENT_FORCE_AVX2_UINT8)
using SelectedNeighborsContainerType = QuantizedNeighbors<Avx2UInt8DotProduct>;
using SelectedSaveNeighborDiscomputer = Avx2SimdDistanceComputerUInt8L2;
#elif defined(RNNDESCENT_USE_AVX512)
#undef RNNDESCENT_USE_AVX512
using SelectedNeighborsContainerType = UInt8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerUInt8L2;
#else
using SelectedNeighborsContainerType = Int8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerInt8L2;
#endif

struct SelectedDistanceComputerFactory {
    static std::unique_ptr<MyDistanceComputer> create_build_graph(const float *data, int n, int d) {
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d);
    }

    static std::unique_ptr<MyDistanceComputer> create_cached_graph(const float *data, int n, int d, const int *row_order = nullptr) {
        // row_order maps graph node IDs to rows in the original matrix.
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d, row_order);
    }

    static std::unique_ptr<MyDistanceComputer> create_refine_search(const float *data, int n, int d) {
        return std::make_unique<SimdDistanceComputerFP32L2>(data, n, d);
    }
};
} // namespace rnndescent
