#pragma once

#include "../discomputer/utils.h"

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rnndescent::detail {

struct NeighborIdSpan {
    const int *ids;
    std::size_t score_end;
};

inline constexpr std::size_t score_mask_bits = 32;

inline constexpr std::size_t score_mask_count(std::size_t count) {
    return (count + score_mask_bits - 1) / score_mask_bits;
}

// Graph degrees are padded to four, so the beam also has a multiple of four scores.
inline void make_neighbor_score_masks(const float *scores, std::size_t count, float cutoff, uint32_t *masks) {
#if defined(__AVX2__)
    const auto cutoff8 = _mm256_set1_ps(cutoff);
    const auto compare_lanes = [&](const float *values) {
        const auto passing = _mm256_cmp_ps(_mm256_loadu_ps(values), cutoff8, _CMP_LT_OQ);
        return static_cast<uint32_t>(_mm256_movemask_ps(passing));
    };
    constexpr std::size_t lanes = 8;
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const auto cutoff4 = vdupq_n_f32(cutoff);
    const uint32x4_t lane_bits = {1, 2, 4, 8};
    const auto compare_lanes = [&](const float *values) {
        const auto passing = vcltq_f32(vld1q_f32(values), cutoff4);
        return vaddvq_u32(vandq_u32(passing, lane_bits));
    };
    constexpr std::size_t lanes = 4;
#else
    const auto compare_lanes = [&](const float *values) { return uint32_t(*values < cutoff); };
    constexpr std::size_t lanes = 1;
#endif

    std::size_t k = 0;
#if defined(__AVX512F__) && !defined(RNNDESCENT_FORCE_AVX2) && \
    !defined(RNNDESCENT_FORCE_AVX2_UINT8) && !defined(RNNDESCENT_FORCE_AVX_VNNI)
    const auto cutoff16 = _mm512_set1_ps(cutoff);
    for (; k + score_mask_bits <= count; k += score_mask_bits) {
        const auto lo = _mm512_cmp_ps_mask(_mm512_loadu_ps(scores + k), cutoff16, _CMP_LT_OQ);
        const auto hi = _mm512_cmp_ps_mask(_mm512_loadu_ps(scores + k + 16), cutoff16, _CMP_LT_OQ);
        *masks++ = uint32_t(lo) | (uint32_t(hi) << 16);
    }
#elif defined(__AVX2__)
    for (; k + score_mask_bits <= count; k += score_mask_bits) {
        *masks++ = compare_lanes(scores + k) | (compare_lanes(scores + k + 8) << 8) |
                   (compare_lanes(scores + k + 16) << 16) | (compare_lanes(scores + k + 24) << 24);
    }
#else
    for (; k + score_mask_bits <= count; k += score_mask_bits) {
        uint32_t word = 0;
        for (std::size_t lane = 0; lane < score_mask_bits; lane += lanes)
            word |= compare_lanes(scores + k + lane) << lane;
        *masks++ = word;
    }
#endif
    if (k == count)
        return;

    const std::size_t base = k;
    uint32_t word = 0;
    for (; k + lanes <= count; k += lanes)
        word |= compare_lanes(scores + k) << (k - base);
#if defined(__AVX2__)
    if (k != count) {
        const auto passing = _mm_cmp_ps(_mm_loadu_ps(scores + k), _mm256_castps256_ps128(cutoff8), _CMP_LT_OQ);
        word |= uint32_t(_mm_movemask_ps(passing)) << (k - base);
    }
#endif
    *masks = word;
}

inline SingleNeighbor *collect_neighbors_from_masks(const float *scores, std::size_t count, const uint32_t *masks,
                                                    const NeighborIdSpan *span, SingleNeighbor *out) {
    if (count == 0)
        return out;

    // Set bits arrive in score order. Empty spans repeat the previous end offset.
    std::size_t span_begin = 0;
    std::size_t span_end = span->score_end;
    for (std::size_t base = 0; base < count; base += score_mask_bits) {
        uint32_t mask = *masks++;
        while (mask) {
            const auto position = base + __builtin_ctz(mask);
            while (position >= span_end) {
                span_begin = span_end;
                span_end = (++span)->score_end;
            }
            *out++ = {span->ids[position - span_begin], scores[position]};
            mask &= mask - 1;
        }
    }
    return out;
}

} // namespace rnndescent::detail
