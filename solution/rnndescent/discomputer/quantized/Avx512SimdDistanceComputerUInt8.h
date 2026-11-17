#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <immintrin.h>

namespace rnndescent {

struct Avx512UInt8DotProduct {
    using Quantization = UInt8Quantization;
    static constexpr size_t block_bytes = sizeof(__m512i);

    static int32_t dot(const uint8_t *q, const uint8_t *x, size_t stride) {
        // The supported dimension bounds all INT32 accumulators, so saturation is unnecessary.
        auto sum = _mm512_setzero_si512();
        for (size_t i = 0; i < stride; i += block_bytes)
            sum = _mm512_dpbusd_epi32(sum, _mm512_loadu_si512(q + i), _mm512_loadu_si512(x + i));
        return _mm512_reduce_add_epi32(sum);
    }

    template <bool Interleaved = false>
    static std::array<int32_t, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1,
                                       const uint8_t *x2, const uint8_t *x3, size_t stride) {
        // Expose the common 64- and 128-byte strides to the compiler.
        if (stride == block_bytes)
            return dot4_blocks<Interleaved>(q, x0, x1, x2, x3, block_bytes);
        if (stride == 2 * block_bytes)
            return dot4_blocks<Interleaved>(q, x0, x1, x2, x3, 2 * block_bytes);
        return dot4_blocks<Interleaved>(q, x0, x1, x2, x3, stride);
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *result) {
        for (int i = 0; i < count; i += 4) {
            const auto *x = codes + static_cast<size_t>(i) * stride;
            const auto dots = dot4<true>(q, x, x + block_bytes, x + 2 * block_bytes, x + 3 * block_bytes, stride);
            for (int k = 0; k < 4; ++k)
                result[i + k] = static_cast<float>(query_norm + norms[i + k] - 2 * dots[k]);
        }
    }

  private:
    static __m256i sum_halves(__m512i sum) {
        return _mm256_add_epi32(_mm512_castsi512_si256(sum), _mm512_extracti64x4_epi64(sum, 1));
    }

    static std::array<int32_t, 4> reduce4(__m512i s0, __m512i s1, __m512i s2, __m512i s3) {
        const auto p01 = _mm256_hadd_epi32(sum_halves(s0), sum_halves(s1));
        const auto p23 = _mm256_hadd_epi32(sum_halves(s2), sum_halves(s3));
        // Each 128-bit half now holds one partial sum for each of the four rows.
        const auto lanes = _mm256_hadd_epi32(p01, p23);
        const auto totals = _mm_add_epi32(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
        std::array<int32_t, 4> result;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(result.data()), totals);
        return result;
    }

    template <bool Interleaved>
    static std::array<int32_t, 4> dot4_blocks(const uint8_t *q, const uint8_t *x0, const uint8_t *x1,
                                              const uint8_t *x2, const uint8_t *x3, size_t stride) {
        auto s0 = _mm512_setzero_si512();
        auto s1 = _mm512_setzero_si512();
        auto s2 = _mm512_setzero_si512();
        auto s3 = _mm512_setzero_si512();
        for (size_t i = 0; i < stride; i += block_bytes) {
            const size_t offset = Interleaved ? 4 * i : i;
            const auto block = _mm512_loadu_si512(q + i);
            s0 = _mm512_dpbusd_epi32(s0, block, _mm512_loadu_si512(x0 + offset));
            s1 = _mm512_dpbusd_epi32(s1, block, _mm512_loadu_si512(x1 + offset));
            s2 = _mm512_dpbusd_epi32(s2, block, _mm512_loadu_si512(x2 + offset));
            s3 = _mm512_dpbusd_epi32(s3, block, _mm512_loadu_si512(x3 + offset));
        }
        return reduce4(s0, s1, s2, s3);
    }
};

using SimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<Avx512UInt8DotProduct>;
using UInt8Neighbors = QuantizedNeighbors<Avx512UInt8DotProduct>;

} // namespace rnndescent
