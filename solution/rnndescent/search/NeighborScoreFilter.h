#pragma once

#include "../discomputer/utils.h"
#include "CandidateBitmap.h"

#include <cstddef>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rnndescent::detail {

// Keep the original adjacency order, including across SIMD blocks.
inline SingleNeighbor *append_neighbors_below_cutoff(const float *scores, const int *ids, std::size_t count, float cutoff,
                                                    CandidateBitmap &visited, SingleNeighbor *out) {
    const auto append_if_new = [&](std::size_t position) {
        const int id = ids[position];
        if (visited.insert(id))
            *out++ = {id, scores[position]};
    };
    [[maybe_unused]] const auto append_mask = [&](unsigned mask, std::size_t offset) {
        while (mask) {
            append_if_new(offset + __builtin_ctz(mask));
            mask &= mask - 1;
        }
    };

    std::size_t k = 0;
#if defined(__AVX512F__) && !defined(RNNDESCENT_FORCE_AVX2) && \
    !defined(RNNDESCENT_FORCE_AVX2_UINT8) && !defined(RNNDESCENT_FORCE_AVX_VNNI)
    // Eight-wide screening gave higher end-to-end QPS on SIFT100K's short rows.
    if (count >= 32) {
        const auto cutoff16 = _mm512_set1_ps(cutoff);
        for (; k + 16 <= count; k += 16) {
            const auto mask = _mm512_cmp_ps_mask(_mm512_loadu_ps(scores + k), cutoff16, _CMP_LT_OQ);
            append_mask(mask, k);
        }
    }
#endif
#if defined(__AVX2__)
    // AVX-VNNI adds integer dot products. FP32 comparisons use the same AVX2 path.
    const auto cutoff8 = _mm256_set1_ps(cutoff);
    for (; k + 8 <= count; k += 8) {
        const auto passing = _mm256_cmp_ps(_mm256_loadu_ps(scores + k), cutoff8, _CMP_LT_OQ);
        append_mask(static_cast<unsigned>(_mm256_movemask_ps(passing)), k);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const auto cutoff4 = vdupq_n_f32(cutoff);
    const uint32x4_t lane_bits = {1, 2, 4, 8};
    for (; k + 4 <= count; k += 4) {
        const auto passing = vcltq_f32(vld1q_f32(scores + k), cutoff4);
        append_mask(vaddvq_u32(vandq_u32(passing, lane_bits)), k);
    }
#endif
    // Packed graph rows have a multiple of four scores, so x86 can leave a four-score tail.
    // Dedicated four-lane comparisons gave comparable end-to-end QPS on SIFT100K.
    for (; k < count; ++k) {
        if (scores[k] < cutoff)
            append_if_new(k);
    }
    return out;
}

} // namespace rnndescent::detail
