#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace rnndescent {

// AVX2-compatible SIMD routines shared with the AVX-VNNI kernel.
using AvxUInt8Accumulate = __m256i (*)(__m256i, __m256i, __m256i);
using AvxUInt8KernelBase = QuantizedKernelBase<UInt8Quantization, sizeof(__m256i), 4>;

template <AvxUInt8Accumulate Accumulate>
struct AvxUInt8DotProduct : AvxUInt8KernelBase {
    using Tile4 = AvxUInt8KernelBase;
    using Tile16 = QuantizedKernelBase<UInt8Quantization, block_bytes, 16, 4>;

    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    uint8_t *codes, int *norms, uint8_t *scratch) {
        Tile16::pack_tiles_with_tail<Tile4>(distance, ids, count, stride, codes, norms, scratch);
    }

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

    static std::array<int32_t, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                                       const uint8_t *x3, size_t stride) {
        // 64-byte codes are common with PCA. Expose the two loads per row to the compiler.
        if (stride == 2 * block_bytes)
            return dot4_blocks(q, x0, x1, x2, x3, 2 * block_bytes);
        return dot4_blocks(q, x0, x1, x2, x3, stride);
    }

  private:
    static std::array<int32_t, 4> dot4_blocks(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2, const uint8_t *x3, size_t stride) {
        auto s0 = _mm256_setzero_si256();
        auto s1 = _mm256_setzero_si256();
        auto s2 = _mm256_setzero_si256();
        auto s3 = _mm256_setzero_si256();
        for (size_t i = 0; i < stride; i += block_bytes) {
            const auto block = load(q + i);
            s0 = Accumulate(s0, block, load(x0 + i));
            s1 = Accumulate(s1, block, load(x1 + i));
            s2 = Accumulate(s2, block, load(x2 + i));
            s3 = Accumulate(s3, block, load(x3 + i));
        }
        return {reduce(s0), reduce(s1), reduce(s2), reduce(s3)};
    }
};

namespace avx_uint8 {

// These routines operate on 256-bit inputs. Each backend chooses its tiles and multiply-add.
inline constexpr size_t block_bytes = sizeof(__m256i);

inline void store_scores8(float *out, const int *norms, __m256i qnorm, __m256i dot) {
    const auto xnorm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(norms));
    const auto l2 = _mm256_sub_epi32(_mm256_add_epi32(qnorm, xnorm), _mm256_slli_epi32(dot, 1));
    _mm256_storeu_ps(out, _mm256_cvtepi32_ps(l2));
}

template <AvxUInt8Accumulate Accumulate, size_t Fixed>
void compute4(const uint8_t *q, const uint8_t *codes, __m128i qnorm, const int *norms, float *out,
              size_t stride) {
    using Dot = AvxUInt8DotProduct<Accumulate>;
    const size_t bytes = Fixed ? Fixed : stride;
    auto s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t feature = 0; feature < bytes; feature += block_bytes) {
        const auto query = Dot::load(q + feature);
        const auto *block = codes + feature * 4;
        s0 = Accumulate(s0, query, Dot::load(block));
        s1 = Accumulate(s1, query, Dot::load(block + block_bytes));
        s2 = Accumulate(s2, query, Dot::load(block + 2 * block_bytes));
        s3 = Accumulate(s3, query, Dot::load(block + 3 * block_bytes));
    }
    // Reduce four dot products together and store four squared distances at once.
    const auto pairs01 = _mm256_hadd_epi32(s0, s1);
    const auto pairs23 = _mm256_hadd_epi32(s2, s3);
    const auto halves = _mm256_hadd_epi32(pairs01, pairs23);
    const auto dots = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
    const auto xnorm = _mm_loadu_si128(reinterpret_cast<const __m128i *>(norms));
    const auto l2 = _mm_sub_epi32(_mm_add_epi32(qnorm, xnorm), _mm_slli_epi32(dots, 1));
    _mm_storeu_ps(out, _mm_cvtepi32_ps(l2));
}

template <AvxUInt8Accumulate Accumulate, size_t Fixed>
inline void compute16(const uint8_t *q, const uint8_t *codes, __m256i qnorm, const int *norms, float *out,
                      size_t bytes) {
    using Dot = AvxUInt8DotProduct<Accumulate>;
    const auto zero = _mm256_setzero_si256();
    auto lo0 = zero, hi0 = zero;
    const auto accumulate_quad = [](__m256i &lo, __m256i &hi, const uint8_t *query, const uint8_t *packed) {
        int32_t word;
        std::memcpy(&word, query, sizeof(word));
        const auto broadcast = _mm256_set1_epi32(word);
        lo = Accumulate(lo, broadcast, Dot::load(packed));
        hi = Accumulate(hi, broadcast, Dot::load(packed + block_bytes));
    };
    if constexpr (Fixed != 0) {
        for (size_t quad = 0; quad < Fixed; quad += 4)
            accumulate_quad(lo0, hi0, q + quad, codes + quad * 16);
    } else {
        // The stride is padded to 32 bytes. Four independent accumulators hide dot-product latency.
        auto lo1 = zero, lo2 = zero, lo3 = zero;
        auto hi1 = zero, hi2 = zero, hi3 = zero;
        for (size_t quad = 0; quad < bytes; quad += 16) {
            accumulate_quad(lo0, hi0, q + quad, codes + quad * 16);
            accumulate_quad(lo1, hi1, q + quad + 4, codes + (quad + 4) * 16);
            accumulate_quad(lo2, hi2, q + quad + 8, codes + (quad + 8) * 16);
            accumulate_quad(lo3, hi3, q + quad + 12, codes + (quad + 12) * 16);
        }
        lo0 = _mm256_add_epi32(_mm256_add_epi32(lo0, lo1), _mm256_add_epi32(lo2, lo3));
        hi0 = _mm256_add_epi32(_mm256_add_epi32(hi0, hi1), _mm256_add_epi32(hi2, hi3));
    }
    store_scores8(out, norms, qnorm, lo0);
    store_scores8(out + 8, norms + 8, qnorm, hi0);
}

template <AvxUInt8Accumulate Accumulate, size_t Fixed>
void score_tiles(const uint8_t *q, int query_norm, int count, size_t stride, const uint8_t *codes,
                 const int *norms, float *out) {
    const size_t bytes = Fixed ? Fixed : stride;
    const auto qnorm = _mm256_set1_epi32(query_norm);
    // The packed 16-row tiles come first, followed by whole four-row tiles.
    const int split = count & ~15;
    int i = 0;
    for (; i < split; i += 16, codes += 16 * bytes)
        compute16<Accumulate, Fixed>(q, codes, qnorm, norms + i, out + i, bytes);

    const auto qnorm4 = _mm256_castsi256_si128(qnorm);
    if constexpr (Fixed == 0) {
        // Expand the four 32-byte feature blocks only when scoring a 128-byte tail.
        if (i < count && bytes == 128) {
            for (; i < count; i += 4, codes += 4 * 128)
                compute4<Accumulate, 128>(q, codes, qnorm4, norms + i, out + i, 128);
            return;
        }
    }
    for (; i < count; i += 4, codes += 4 * bytes)
        compute4<Accumulate, Fixed>(q, codes, qnorm4, norms + i, out + i, bytes);
}

} // namespace avx_uint8

inline __m256i avx2_accumulate_byte_products(__m256i sum, __m256i q, __m256i x) {
    // Codes are in [0, 126], so two products fit in a signed 16-bit lane.
    const auto pairs = _mm256_maddubs_epi16(q, x);
    return _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

struct Avx2UInt8Kernel : AvxUInt8DotProduct<avx2_accumulate_byte_products> {
    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *result) {
        if (stride == 64)
            avx_uint8::score_tiles<avx2_accumulate_byte_products, 64>(q, query_norm, count, 64, codes, norms, result);
        else if (stride == 32)
            avx_uint8::score_tiles<avx2_accumulate_byte_products, 32>(q, query_norm, count, 32, codes, norms, result);
        else
            avx_uint8::score_tiles<avx2_accumulate_byte_products, 0>(q, query_norm, count, stride, codes, norms, result);
    }
};

} // namespace rnndescent
