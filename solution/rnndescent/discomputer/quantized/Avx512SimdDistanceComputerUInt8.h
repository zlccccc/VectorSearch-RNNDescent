#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace rnndescent {

struct Avx512UInt8Kernel : QuantizedKernelBase<UInt8Quantization, sizeof(__m512i), 4> {
    using Tile16 = QuantizedKernelBase<UInt8Quantization, block_bytes, 16, 4>;

    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    Code *codes, int *norms, uint8_t *scratch) {
        const int split = stride == block_bytes ? count & ~15 : 0;
        Tile16::pack_tiles(distance, ids, split, stride, codes, norms, scratch);
        if (split < count)
            pack_tiles(distance, ids + split, count - split, stride, codes + size_t(split) * stride, norms + split, scratch);
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

    static int32_t dot(const uint8_t *q, const uint8_t *x, size_t stride) {
        // The supported dimension bounds all INT32 accumulators, so saturation is unnecessary.
        auto sum = _mm512_setzero_si512();
        for (size_t i = 0; i < stride; i += block_bytes)
            sum = _mm512_dpbusd_epi32(sum, _mm512_loadu_si512(q + i), _mm512_loadu_si512(x + i));
        return _mm512_reduce_add_epi32(sum);
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *result) {
        if (stride == block_bytes)
            score_cached<block_bytes>(q, query_norm, count, stride, codes, norms, result);
        else if (stride == 2 * block_bytes)
            score_cached<2 * block_bytes>(q, query_norm, count, stride, codes, norms, result);
        else
            score_cached<0>(q, query_norm, count, stride, codes, norms, result);
    }

  private:
    template <size_t Fixed>
    static void score_cached(const uint8_t *q, int query_norm, int count, size_t stride,
                             const uint8_t *codes, const int *norms, float *out) {
        const size_t bytes = Fixed ? Fixed : stride;
        int i = 0;
        if constexpr (Fixed == block_bytes) {
            const auto qnorm = _mm512_set1_epi32(query_norm);
            const int split = count & ~15;
            for (; i < split; i += 16, codes += 16 * bytes)
                score16(q, codes, qnorm, norms + i, out + i);
        }
        const auto qnorm = _mm_set1_epi32(query_norm);
        for (; i < count; i += 4, codes += 4 * bytes)
            score4<Fixed>(q, codes, qnorm, norms + i, out + i, bytes);
    }

    static void score16(const uint8_t *q, const uint8_t *codes, __m512i qnorm, const int *norms, float *out) {
        const auto zero = _mm512_setzero_si512();
        auto s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        const auto update = [&](size_t j, __m512i &sum) {
            int32_t word;
            std::memcpy(&word, q + j, sizeof(word));
            sum = _mm512_dpbusd_epi32(sum, _mm512_set1_epi32(word), _mm512_loadu_si512(codes + 16 * j));
        };
        // Each lane accumulates one neighbor in the 4-byte interleaved tile.
#pragma GCC unroll 4
        for (size_t j = 0; j < block_bytes; j += 16) {
            update(j, s0);
            update(j + 4, s1);
            update(j + 8, s2);
            update(j + 12, s3);
        }
        const auto dots = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
        const auto squared = _mm512_sub_epi32(_mm512_add_epi32(qnorm, _mm512_loadu_si512(norms)), _mm512_slli_epi32(dots, 1));
        _mm512_storeu_ps(out, _mm512_cvtepi32_ps(squared));
    }

    template <size_t Fixed>
    static void score4(const uint8_t *q, const uint8_t *codes, __m128i qnorm, const int *norms, float *out, size_t stride) {
        const size_t bytes = Fixed ? Fixed : stride;
        const auto zero = _mm512_setzero_si512();
        auto s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        for (size_t j = 0; j < bytes; j += block_bytes) {
            const auto query = _mm512_loadu_si512(q + j);
            const auto *block = codes + 4 * j;
            s0 = _mm512_dpbusd_epi32(s0, query, _mm512_loadu_si512(block));
            s1 = _mm512_dpbusd_epi32(s1, query, _mm512_loadu_si512(block + block_bytes));
            s2 = _mm512_dpbusd_epi32(s2, query, _mm512_loadu_si512(block + 2 * block_bytes));
            s3 = _mm512_dpbusd_epi32(s3, query, _mm512_loadu_si512(block + 3 * block_bytes));
        }
        const auto p01 = _mm512_add_epi32(_mm512_shuffle_i32x4(s0, s1, 0x44), _mm512_shuffle_i32x4(s0, s1, 0xee));
        const auto p23 = _mm512_add_epi32(_mm512_shuffle_i32x4(s2, s3, 0x44), _mm512_shuffle_i32x4(s2, s3, 0xee));
        // Put each neighbor's partial sums in its own 128-bit lane, then gather the four totals.
        auto lanes = _mm512_add_epi32(_mm512_shuffle_i32x4(p01, p23, 0x88), _mm512_shuffle_i32x4(p01, p23, 0xdd));
        lanes = _mm512_add_epi32(lanes, _mm512_shuffle_epi32(lanes, _MM_PERM_BADC));
        lanes = _mm512_add_epi32(lanes, _mm512_shuffle_epi32(lanes, _MM_PERM_CDAB));
        const auto dots = _mm512_castsi512_si128(_mm512_maskz_compress_epi32(0x1111, lanes));
        const auto xnorm = _mm_loadu_si128(reinterpret_cast<const __m128i *>(norms));
        const auto squared = _mm_sub_epi32(_mm_add_epi32(qnorm, xnorm), _mm_slli_epi32(dots, 1));
        _mm_storeu_ps(out, _mm_cvtepi32_ps(squared));
    }

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

using Avx512UInt8DotProduct = Avx512UInt8Kernel;
using SimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<Avx512UInt8DotProduct>;
using UInt8Neighbors = QuantizedNeighbors<Avx512UInt8DotProduct>;

} // namespace rnndescent
