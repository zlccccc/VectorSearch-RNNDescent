#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <immintrin.h>

namespace rnndescent {

inline __m256i avx2_accumulate_byte_products(__m256i sum, __m256i q, __m256i x) {
    // Codes are in [0, 126], so two products fit in a signed 16-bit lane.
    const auto pairs = _mm256_maddubs_epi16(q, x);
    return _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

template <__m256i (*Accumulate)(__m256i, __m256i, __m256i)> struct AvxUInt8DotProduct {
    using Quantization = UInt8Quantization;
    static constexpr size_t block_bytes = sizeof(__m256i);

    static __m256i load(const uint8_t *data) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data));
    }

    static int32_t reduce(__m256i sum) {
        auto halves = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        halves = _mm_hadd_epi32(halves, halves);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(halves, halves));
    }

    static int32_t dot(const uint8_t *q, const uint8_t *x, size_t stride) {
        auto sum = _mm256_setzero_si256();
        for (size_t i = 0; i < stride; i += block_bytes)
            sum = Accumulate(sum, load(q + i), load(x + i));
        return reduce(sum);
    }

    template <bool Interleaved = false>
    static std::array<int32_t, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2, const uint8_t *x3, size_t stride) {
        // 64-byte codes are common with PCA. Expose the two loads per row to the compiler.
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
    template <bool Interleaved>
    static std::array<int32_t, 4> dot4_blocks(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2, const uint8_t *x3, size_t stride) {
        auto s0 = _mm256_setzero_si256();
        auto s1 = _mm256_setzero_si256();
        auto s2 = _mm256_setzero_si256();
        auto s3 = _mm256_setzero_si256();
        for (size_t i = 0; i < stride; i += block_bytes) {
            // In the cache, one block from each of the four rows sits side by side.
            const size_t offset = Interleaved ? 4 * i : i;
            const auto block = load(q + i);
            s0 = Accumulate(s0, block, load(x0 + offset));
            s1 = Accumulate(s1, block, load(x1 + offset));
            s2 = Accumulate(s2, block, load(x2 + offset));
            s3 = Accumulate(s3, block, load(x3 + offset));
        }
        return {reduce(s0), reduce(s1), reduce(s2), reduce(s3)};
    }
};

using Avx2UInt8DotProduct = AvxUInt8DotProduct<avx2_accumulate_byte_products>;
using Avx2SimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<Avx2UInt8DotProduct>;

} // namespace rnndescent
