#pragma once

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
using SelectedNeighborsContainerType = Int8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerInt8L2;
} // namespace rnndescent
