#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <immintrin.h>

namespace {

inline int32_t ReduceAddEpi32(const __m256i &sums) {
    __m128i ab_sum = _mm_add_epi32(_mm256_extracti128_si256(sums, 0), _mm256_extracti128_si256(sums, 1));
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    return _mm_extract_epi32(ab_sum, 0);
}

inline __m256i Correlation(const __m128i &a, const __m128i &b) { return _mm256_madd_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)); }

} // namespace

namespace rnndescent {

struct Avx2Int8DotProduct {
    using Quantization = SignedInt8Quantization;
    static constexpr size_t block_bytes = sizeof(__m128i);
    static constexpr size_t unroll_bytes = 4 * block_bytes;

    static int32_t dot(const int8_t *a, const int8_t *b, size_t stride) {
        __m256i sums = _mm256_setzero_si256();
        for (size_t i = 0; i < stride; i += block_bytes) {
            sums = _mm256_add_epi32(sums, Correlation(_mm_loadu_si128((__m128i const *)(a + i)), _mm_loadu_si128((__m128i const *)(b + i))));
        }
        return ReduceAddEpi32(sums);
    }

    static std::array<int32_t, 4> dot4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2,
                                       const int8_t *x3, size_t stride) {
        __m128i loadq;
        __m256i sums0 = _mm256_setzero_si256();
        __m256i sums1 = _mm256_setzero_si256();
        __m256i sums2 = _mm256_setzero_si256();
        __m256i sums3 = _mm256_setzero_si256();
        const size_t unrolled_size = stride - stride % unroll_bytes;
        size_t i = 0;
        for (; i < unrolled_size; i += unroll_bytes) {
            loadq = _mm_loadu_si128((__m128i const *)(q + i));
            sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i)), loadq));
            sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i)), loadq));
            sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i)), loadq));
            sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i)), loadq));
            loadq = _mm_loadu_si128((__m128i const *)(q + i + 16));
            sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 16)), loadq));
            sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 16)), loadq));
            sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 16)), loadq));
            sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 16)), loadq));
            loadq = _mm_loadu_si128((__m128i const *)(q + i + 32));
            sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 32)), loadq));
            sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 32)), loadq));
            sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 32)), loadq));
            sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 32)), loadq));
            loadq = _mm_loadu_si128((__m128i const *)(q + i + 48));
            sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 48)), loadq));
            sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 48)), loadq));
            sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 48)), loadq));
            sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 48)), loadq));
        }
        for (; i < stride; i += block_bytes) {
            loadq = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + i));
            sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x0 + i)), loadq));
            sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x1 + i)), loadq));
            sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x2 + i)), loadq));
            sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x3 + i)), loadq));
        }
        return {ReduceAddEpi32(sums0), ReduceAddEpi32(sums1), ReduceAddEpi32(sums2), ReduceAddEpi32(sums3)};
    }

    // A group of four cached neighbors is interleaved by SIMD block.
    static void cached_distances(const int8_t *__restrict q, int query_norm, int size,
                                 uint32_t code_stride,
                                 const int8_t *__restrict xvalue, const int *__restrict l2norms, float *result) {
        const size_t unrolled_size = code_stride - code_stride % unroll_bytes;
        for (int id = 0; id < size; id += 4) {
            const int8_t *__restrict x = xvalue + static_cast<size_t>(id) * code_stride;

            __m256i sums0 = _mm256_setzero_si256();
            __m256i sums1 = _mm256_setzero_si256();
            __m256i sums2 = _mm256_setzero_si256();
            __m256i sums3 = _mm256_setzero_si256();
            size_t i = 0;
            for (; i < unrolled_size; i += unroll_bytes) {
#pragma unroll
                for (int k = 0; k < 64; k += 16) {
                    __m128i loadq = _mm_loadu_si128((__m128i const *)(q + i + k));
                    sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 0)), loadq));
                    sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 16)), loadq));
                    sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 32)), loadq));
                    sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 48)), loadq));
                }
            }
            for (; i < code_stride; i += block_bytes) {
                const auto loadq = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + i));
                const auto *chunk = x + i * 4;
                sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(chunk)), loadq));
                sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(chunk + block_bytes)), loadq));
                sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(chunk + 2 * block_bytes)), loadq));
                sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128(reinterpret_cast<const __m128i *>(chunk + 3 * block_bytes)), loadq));
            }
            result[id] = static_cast<float>(query_norm + l2norms[id] - 2 * ReduceAddEpi32(sums0));
            result[id + 1] = static_cast<float>(query_norm + l2norms[id + 1] - 2 * ReduceAddEpi32(sums1));
            result[id + 2] = static_cast<float>(query_norm + l2norms[id + 2] - 2 * ReduceAddEpi32(sums2));
            result[id + 3] = static_cast<float>(query_norm + l2norms[id + 3] - 2 * ReduceAddEpi32(sums3));
        }
    }
};

using SimdDistanceComputerInt8L2 = QuantizedDistanceComputer<Avx2Int8DotProduct>;
using Int8Neighbors = QuantizedNeighbors<Avx2Int8DotProduct>;

} // namespace rnndescent
