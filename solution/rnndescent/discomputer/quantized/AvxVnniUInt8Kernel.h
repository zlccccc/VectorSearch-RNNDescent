#pragma once

#include "Avx2UInt8Kernel.h"

namespace rnndescent {

inline __m256i avx_vnni_accumulate_byte_products(__m256i sum, __m256i q, __m256i x) {
    return _mm256_dpbusd_avx_epi32(sum, q, x);
}

struct AvxVnniUInt8Kernel : AvxUInt8DotProduct<avx_vnni_accumulate_byte_products> {
    using Tile4 = QuantizedKernelBase<UInt8Quantization, sizeof(__m256i), 4>;
    using Tile16 = QuantizedKernelBase<UInt8Quantization, sizeof(__m256i), 16, 4>;

    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    uint8_t *codes, int *norms, uint8_t *scratch) {
        avx_uint8::pack_neighbors<Tile4, Tile16>(distance, ids, count, stride, codes, norms, scratch);
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *result) {
        if (stride == 64)
            avx_uint8::score_tiles<avx_vnni_accumulate_byte_products, 64>(q, query_norm, count, 64, codes, norms, result);
        else if (stride == 32)
            avx_uint8::score_tiles<avx_vnni_accumulate_byte_products, 32>(q, query_norm, count, 32, codes, norms, result);
        else if (stride == 128)
            cached_distances128(q, query_norm, count, codes, norms, result);
        else
            avx_uint8::score_tiles<avx_vnni_accumulate_byte_products, 0>(q, query_norm, count, stride, codes, norms, result);
    }

  private:
    // Keep the unrolled 128-byte kernel out of the graph traversal's instruction stream.
    [[gnu::noinline]] static void cached_distances128(const uint8_t *q, int query_norm, int count,
                                                     const uint8_t *codes, const int *norms, float *result) {
        constexpr size_t stride = 128;
        const auto zero = _mm256_setzero_si256();
        const auto qnorm = _mm256_set1_epi32(query_norm);
        const int split = count & ~15;
        int i = 0;
        for (; i < split; i += 16, codes += 16 * stride) {
            auto lo0 = zero, lo1 = zero, lo2 = zero, lo3 = zero;
            auto hi0 = zero, hi1 = zero, hi2 = zero, hi3 = zero;
            const auto accumulate_quad = [&](size_t feature, __m256i &lo, __m256i &hi) {
                int32_t word;
                std::memcpy(&word, q + feature, sizeof(word));
                const auto broadcast = _mm256_set1_epi32(word);
                const auto *quad = codes + feature * 16;
                lo = _mm256_dpbusd_avx_epi32(lo, broadcast, load(quad));
                hi = _mm256_dpbusd_avx_epi32(hi, broadcast, load(quad + block_bytes));
            };
#pragma GCC unroll 8
            for (size_t feature = 0; feature < stride; feature += 16) {
                accumulate_quad(feature, lo0, hi0);
                accumulate_quad(feature + 4, lo1, hi1);
                accumulate_quad(feature + 8, lo2, hi2);
                accumulate_quad(feature + 12, lo3, hi3);
            }
            const auto lo = _mm256_add_epi32(_mm256_add_epi32(lo0, lo1), _mm256_add_epi32(lo2, lo3));
            const auto hi = _mm256_add_epi32(_mm256_add_epi32(hi0, hi1), _mm256_add_epi32(hi2, hi3));
            avx_uint8::store_scores8(result + i, norms + i, qnorm, lo);
            avx_uint8::store_scores8(result + i + 8, norms + i + 8, qnorm, hi);
        }
        const auto qnorm4 = _mm256_castsi256_si128(qnorm);
        for (; i < count; i += 4, codes += 4 * stride)
            avx_uint8::compute4<avx_vnni_accumulate_byte_products, stride>(
                q, codes, qnorm4, norms + i, result + i, stride);
    }
};

} // namespace rnndescent
