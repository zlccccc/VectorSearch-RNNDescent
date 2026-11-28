#pragma once

#include "QuantizedKernelBase.h"
#include "UInt4Quantization.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace rnndescent {

struct Avx512UInt4Kernel {
    // A 64-feature block occupies 32 bytes before expanding to a ZMM register.
    static constexpr size_t block_bytes = 32;
    using Encoding = PackedUInt4Quantization<block_bytes>;
    using Tile4 = QuantizedKernelBase<Encoding, block_bytes, 4>;
    using Tile16 = QuantizedKernelBase<Encoding, block_bytes, 16, 4>;

    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    uint8_t *codes, int *norms, uint8_t *scratch) {
        Tile16::pack_tiles_with_tail<Tile4>(distance, ids, count, stride, codes, norms, scratch);
    }

    static int dot(const uint8_t *q, const uint8_t *row, size_t stride) {
        auto sum = _mm512_setzero_si512();
        for (size_t i = 0; i < stride; i += block_bytes)
            sum = _mm512_dpbusd_epi32(sum, expand(row + i), _mm512_loadu_si512(q + 2 * i));
        return _mm512_reduce_add_epi32(sum);
    }

    static std::array<int, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                                   const uint8_t *x3, size_t stride) {
        std::array<int, 4> result;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(result.data()), dots4<false>(q, x0, x1, x2, x3, stride));
        return result;
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *out) {
        const auto qnorm = _mm_set1_epi32(query_norm);
        const auto norm16 = _mm512_set1_epi32(query_norm);
        const int split = count & ~15;
        const size_t tile16_stride = 16 * size_t(stride);
        const size_t tile4_stride = 4 * size_t(stride);
        int i = 0;
        for (; i < split; i += 16, codes += tile16_stride)
            score16(q, codes, codes + tile16_stride, norm16, norms + i, out + i);
        for (; i < count; i += 4, codes += tile4_stride)
            store4(dots4<true>(q, codes, codes + block_bytes, codes + 2 * block_bytes, codes + 3 * block_bytes, stride),
                   qnorm, norms + i, out + i);
    }

  private:
    static __m512i expand(const uint8_t *row) {
        const auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row));
        const auto mask = _mm256_set1_epi8(15);
        const auto low = _mm256_and_si256(bytes, mask);
        const auto high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
        return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    }

    template <bool Interleaved>
    static __m128i dots4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                         const uint8_t *x3, size_t stride) {
        auto s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
        for (size_t i = 0; i < stride; i += block_bytes) {
            const auto query = _mm512_loadu_si512(q + 2 * i);
            const size_t offset = Interleaved ? 4 * i : i;
            s0 = _mm512_dpbusd_epi32(s0, expand(x0 + offset), query);
            s1 = _mm512_dpbusd_epi32(s1, expand(x1 + offset), query);
            s2 = _mm512_dpbusd_epi32(s2, expand(x2 + offset), query);
            s3 = _mm512_dpbusd_epi32(s3, expand(x3 + offset), query);
        }
        const auto halves = [](__m512i sum) {
            return _mm256_add_epi32(_mm512_castsi512_si256(sum), _mm512_extracti64x4_epi64(sum, 1));
        };
        const auto pairs01 = _mm256_hadd_epi32(halves(s0), halves(s1));
        const auto pairs23 = _mm256_hadd_epi32(halves(s2), halves(s3));
        const auto totals = _mm256_hadd_epi32(pairs01, pairs23);
        return _mm_add_epi32(_mm256_castsi256_si128(totals), _mm256_extracti128_si256(totals, 1));
    }

    static void store4(__m128i dots, __m128i qnorm, const int *norms, float *out) {
        const auto xnorm = _mm_loadu_si128(reinterpret_cast<const __m128i *>(norms));
        const auto squared = _mm_sub_epi32(_mm_add_epi32(qnorm, xnorm), _mm_slli_epi32(dots, Encoding::twice_dot_shift));
        _mm_storeu_ps(out, _mm_cvtepi32_ps(squared));
    }

    static void score16(const uint8_t *query, const uint8_t *block, const uint8_t *end, __m512i qnorm,
                        const int *norms, float *out) {
        auto s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
        const auto mask = _mm512_set1_epi8(15);
        constexpr size_t lane_group_bytes = sizeof(__m512i);
        // Each ZMM lane accumulates one neighbor from the interleaved tile.
        const auto update = [&](const uint8_t *query, const uint8_t *tile, __m512i sum) {
            int low, high;
            std::memcpy(&low, query, sizeof(low));
            std::memcpy(&high, query + block_bytes, sizeof(high));
            const auto packed = _mm512_loadu_si512(tile);
            sum = _mm512_dpbusd_epi32(sum, _mm512_and_si512(packed, mask), _mm512_set1_epi32(low));
            return _mm512_dpbusd_epi32(sum, _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask), _mm512_set1_epi32(high));
        };
        do {
            for (size_t j = 0; j < block_bytes; j += 16) {
                const auto *tile = block + 16 * j;
                s0 = update(query + j, tile, s0);
                s1 = update(query + j + 4, tile + lane_group_bytes, s1);
                s2 = update(query + j + 8, tile + 2 * lane_group_bytes, s2);
                s3 = update(query + j + 12, tile + 3 * lane_group_bytes, s3);
            }
            query += 2 * block_bytes;
            block += 16 * block_bytes;
        } while (block != end);
        const auto dots = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
        const auto squared = _mm512_sub_epi32(_mm512_add_epi32(qnorm, _mm512_loadu_si512(norms)),
                                             _mm512_slli_epi32(dots, Encoding::twice_dot_shift));
        _mm512_storeu_ps(out, _mm512_cvtepi32_ps(squared));
    }
};

} // namespace rnndescent
