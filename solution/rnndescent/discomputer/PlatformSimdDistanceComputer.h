#pragma once

#include "../common/QuantizedFormat.h"

#include <memory>
#include <stdexcept>

// Optional manual overrides for benchmarking / deployment:
//   RNNDESCENT_FORCE_AVX512
//   RNNDESCENT_FORCE_AVX2
//   RNNDESCENT_FORCE_AVX2_UINT8
//   RNNDESCENT_FORCE_AVX_VNNI
//   RNNDESCENT_FORCE_NEON

#if defined(RNNDESCENT_FORCE_AVX_VNNI)
#include "flat/Avx2SimdDistanceComputerFP32.h"
#include "quantized/AvxVnniSimdDistanceComputerUInt8.h"
#include "quantized/AvxVnniUInt4Kernel.h"
#define RNNDESCENT_HAS_UINT4_KERNEL
#elif defined(RNNDESCENT_FORCE_AVX2_UINT8)
#include "flat/Avx2SimdDistanceComputerFP32.h"
#include "quantized/Avx2SimdDistanceComputerUInt8.h"
#elif defined(RNNDESCENT_FORCE_AVX512) || \
    (!defined(RNNDESCENT_FORCE_AVX2) && !defined(RNNDESCENT_FORCE_NEON) && \
     defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__))
#define RNNDESCENT_USE_AVX512
#include "flat/Avx512SimdDistanceComputerFP32.h"
#include "quantized/Avx512SimdDistanceComputerUInt8.h"
#include "quantized/Avx512UInt4Kernel.h"
#define RNNDESCENT_HAS_UINT4_KERNEL
#elif defined(RNNDESCENT_FORCE_AVX2)
#include "flat/Avx2SimdDistanceComputerFP32.h"
#include "quantized/Avx2SimdDistanceComputerInt8.h"
#elif defined(RNNDESCENT_FORCE_NEON)
#include "flat/NeonSimdDistanceComputerFP32.h"
#include "quantized/NeonSimdDistanceComputerInt8.h"
#include "quantized/NeonUInt4Kernel.h"
#define RNNDESCENT_HAS_UINT4_KERNEL
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "flat/Avx2SimdDistanceComputerFP32.h"
#include "quantized/Avx2SimdDistanceComputerInt8.h"
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#include "flat/NeonSimdDistanceComputerFP32.h"
#include "quantized/NeonSimdDistanceComputerInt8.h"
#if defined(__ARM_FEATURE_DOTPROD)
#include "quantized/NeonUInt4Kernel.h"
#define RNNDESCENT_HAS_UINT4_KERNEL
#endif
#else
#error "Unsupported platform for SIMD distance computer. Please add a fallback implementation."
#endif

namespace rnndescent {
#if defined(RNNDESCENT_FORCE_AVX_VNNI)
using SelectedUInt4Kernel = AvxVnniUInt4Kernel;
using SelectedNeighborsContainerType = QuantizedNeighbors<AvxVnniUInt8DotProduct>;
using SelectedSaveNeighborDiscomputer = AvxVnniSimdDistanceComputerUInt8L2;
#elif defined(RNNDESCENT_FORCE_AVX2_UINT8)
using SelectedNeighborsContainerType = QuantizedNeighbors<Avx2UInt8DotProduct>;
using SelectedSaveNeighborDiscomputer = Avx2SimdDistanceComputerUInt8L2;
#elif defined(RNNDESCENT_USE_AVX512)
#undef RNNDESCENT_USE_AVX512
using SelectedUInt4Kernel = Avx512UInt4Kernel;
using SelectedNeighborsContainerType = UInt8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerUInt8L2;
#else
#if defined(RNNDESCENT_HAS_UINT4_KERNEL)
using SelectedUInt4Kernel = NeonUInt4Kernel;
#endif
using SelectedNeighborsContainerType = Int8Neighbors;
using SelectedSaveNeighborDiscomputer = SimdDistanceComputerInt8L2;
#endif

struct SelectedDistanceComputerFactory {
    static void validate_quantization([[maybe_unused]] QuantizedFormat format) {
#if !defined(RNNDESCENT_HAS_UINT4_KERNEL)
        if (format == QuantizedFormat::UInt4)
            throw std::invalid_argument("uint4 quantization requires AVX-VNNI, AVX-512 VNNI, or NEON Dot Product");
#endif
    }

    static std::unique_ptr<MyDistanceComputer> create_build_graph(const float *data, int n, int d) {
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d);
    }

    static std::unique_ptr<MyDistanceComputer> create_cached_graph(const float *data, int n, int d, const int *row_order = nullptr,
                                                                  QuantizedFormat format = QuantizedFormat::Byte) {
        // row_order maps graph node IDs to rows in the original matrix.
        validate_quantization(format);
#if defined(RNNDESCENT_HAS_UINT4_KERNEL)
        using PackedGraph = QuantizedDistanceComputer<SelectedUInt4Kernel, SelectedUInt4Kernel::Encoding>;
        if (format == QuantizedFormat::UInt4)
            return std::make_unique<PackedGraph>(data, n, d, row_order);
#endif
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d, row_order);
    }

    static std::unique_ptr<MyDistanceComputer> create_refine_search(const float *data, int n, int d) {
        return std::make_unique<SimdDistanceComputerFP32L2>(data, n, d);
    }

    static std::unique_ptr<MyDistanceComputer> create_quantized_refine(const float *data, int n, int d,
                                                                      const int *row_order = nullptr,
                                                                      QuantizedFormat format = QuantizedFormat::Byte) {
        validate_quantization(format);
#if defined(RNNDESCENT_HAS_UINT4_KERNEL)
        if (format == QuantizedFormat::UInt4) {
            using Encoding = RowwiseUInt4Quantization<SelectedUInt4Kernel::block_bytes>;
            return std::make_unique<QuantizedDistanceComputer<SelectedUInt4Kernel, Encoding>>(data, n, d, row_order);
        }
#endif
        return std::make_unique<SelectedSaveNeighborDiscomputer>(data, n, d, row_order);
    }
};
} // namespace rnndescent
