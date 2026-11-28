#pragma once

#include "QuantizedKernelBase.h"
#include "UInt4Quantization.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace rnndescent {

struct AvxVnniUInt4Kernel {
    static constexpr size_t block_bytes = sizeof(__m256i);
    using Encoding = PackedUInt4Quantization<block_bytes>;
    using Tile4 = QuantizedKernelBase<Encoding, block_bytes, 4>;
    using Tile16 = QuantizedKernelBase<Encoding, block_bytes, 16, 4>;

    // Each word packs two groups of four features, 32 dimensions apart, into low and high nibbles.
    // The 16-row layout interleaves those words across neighbors so each lane
    // accumulates one neighbor. Four-row tails use contiguous 32-byte blocks.
    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    uint8_t *codes, int *norms, uint8_t *scratch) {
        Tile16::pack_tiles_with_tail<Tile4>(distance, ids, count, stride, codes, norms, scratch);
    }

    static __m256i accumulate(__m256i sum, const uint8_t *code, __m256i qlow, __m256i qhigh) {
        const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(code));
        const auto mask = _mm256_set1_epi8(15);
        const auto low = _mm256_and_si256(packed, mask);
        const auto high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        sum = _mm256_dpbusd_avx_epi32(sum, low, qlow);
        return _mm256_dpbusd_avx_epi32(sum, high, qhigh);
    }

    static int reduce(__m256i sum) {
        auto halves = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        halves = _mm_hadd_epi32(halves, halves);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(halves, halves));
    }

    static int dot(const uint8_t *q, const uint8_t *row, size_t stride) {
        auto sum = _mm256_setzero_si256();
        for (size_t i = 0; i < stride; i += block_bytes) {
            const auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * i));
            const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * i + block_bytes));
            sum = accumulate(sum, row + i, low, high);
        }
        return reduce(sum);
    }

    static std::array<int, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                                   const uint8_t *x3, size_t stride) {
        auto s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (size_t feature = 0; feature < stride; feature += block_bytes) {
            const auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * feature));
            const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * feature + block_bytes));
            s0 = accumulate(s0, x0 + feature, low, high);
            s1 = accumulate(s1, x1 + feature, low, high);
            s2 = accumulate(s2, x2 + feature, low, high);
            s3 = accumulate(s3, x3 + feature, low, high);
        }
        std::array<int, 4> products;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(products.data()), reduce4(s0, s1, s2, s3));
        return products;
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *out) {
        const auto qnorm = _mm256_set1_epi32(query_norm);
        const int split = count & ~15;
        const size_t tile16_stride = 16 * size_t(stride);
        const size_t tile4_stride = 4 * size_t(stride);
        int i = 0;
        for (; i < split; i += 16, codes += tile16_stride) {
            auto s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
            const auto *query = q;
            const auto *block = codes;
            const auto *end = codes + tile16_stride;
            // Advance in full blocks of 16 packed rows and matching query bytes.
            do {
                for (size_t j = 0; j < block_bytes; j += 8) {
                    const auto qlo0 = broadcast(query + j);
                    const auto qhi0 = broadcast(query + block_bytes + j);
                    const auto qlo1 = broadcast(query + j + 4);
                    const auto qhi1 = broadcast(query + block_bytes + j + 4);
                    const auto *tile = block + 16 * j;
                    s0 = accumulate(s0, tile, qlo0, qhi0);
                    s1 = accumulate(s1, tile + block_bytes, qlo0, qhi0);
                    s2 = accumulate(s2, tile + 2 * block_bytes, qlo1, qhi1);
                    s3 = accumulate(s3, tile + 3 * block_bytes, qlo1, qhi1);
                }
                query += 2 * block_bytes;
                block += 16 * block_bytes;
            } while (block != end);
            store8(out + i, norms + i, qnorm, _mm256_add_epi32(s0, s2));
            store8(out + i + 8, norms + i + 8, qnorm, _mm256_add_epi32(s1, s3));
        }
        for (; i < count; i += 4, codes += tile4_stride) {
            auto s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
            for (size_t feature = 0; feature < stride; feature += block_bytes) {
                const auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * feature));
                const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 2 * feature + block_bytes));
                const auto *tile = codes + feature * 4;
                s0 = accumulate(s0, tile, low, high);
                s1 = accumulate(s1, tile + block_bytes, low, high);
                s2 = accumulate(s2, tile + 2 * block_bytes, low, high);
                s3 = accumulate(s3, tile + 3 * block_bytes, low, high);
            }
            const auto xnorm = _mm_loadu_si128(reinterpret_cast<const __m128i *>(norms + i));
            _mm_storeu_ps(out + i, squared_distances4(_mm256_castsi256_si128(qnorm), xnorm, reduce4(s0, s1, s2, s3)));
        }
    }

  private:
    static __m128i reduce4(__m256i s0, __m256i s1, __m256i s2, __m256i s3) {
        const auto pairs01 = _mm256_hadd_epi32(s0, s1);
        const auto pairs23 = _mm256_hadd_epi32(s2, s3);
        const auto halves = _mm256_hadd_epi32(pairs01, pairs23);
        return _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
    }

    static __m128 squared_distances4(__m128i query_norm, __m128i norms, __m128i dots) {
        const auto sum = _mm_add_epi32(query_norm, norms);
        return _mm_cvtepi32_ps(_mm_sub_epi32(sum, _mm_slli_epi32(dots, Encoding::twice_dot_shift)));
    }

    static __m256i broadcast(const uint8_t *query) {
        int32_t word;
        std::memcpy(&word, query, sizeof(word));
        return _mm256_set1_epi32(word);
    }

    static void store8(float *out, const int *norms, __m256i qnorm, __m256i dot) {
        const auto xnorm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(norms));
        const auto score = _mm256_sub_epi32(_mm256_add_epi32(qnorm, xnorm), _mm256_slli_epi32(dot, Encoding::twice_dot_shift));
        _mm256_storeu_ps(out, _mm256_cvtepi32_ps(score));
    }
};

} // namespace rnndescent
