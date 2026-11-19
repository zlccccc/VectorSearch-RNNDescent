#pragma once

#include "QuantizedDistanceComputer.h"

#include <arm_neon.h>
#include <array>

namespace rnndescent {

struct NeonInt8DotProduct {
    using Quantization = SignedInt8Quantization;
    static constexpr size_t block_bytes = 16;

    static int32_t dot(const int8_t *a, const int8_t *b, size_t stride) {
        int32x4_t sums = vdupq_n_s32(0);
        for (size_t i = 0; i < stride; i += block_bytes) {
            sums = vdotq_s32(sums, vld1q_s8(a + i), vld1q_s8(b + i));
        }
        return vaddvq_s32(sums);
    }

    static std::array<int32_t, 4> dot4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2,
                                       const int8_t *x3, size_t stride) {
        int8x16_t loadq;
        int32x4_t sums0 = vdupq_n_s32(0);
        int32x4_t sums1 = vdupq_n_s32(0);
        int32x4_t sums2 = vdupq_n_s32(0);
        int32x4_t sums3 = vdupq_n_s32(0);
        const size_t unrolled_size = stride - stride % 64;
        size_t i = 0;
        for (; i < unrolled_size; i += 64) {
            loadq = vld1q_s8(q + i);
            sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i + 00), loadq);
            sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i + 00), loadq);
            sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i + 00), loadq);
            sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i + 00), loadq);
            loadq = vld1q_s8(q + i + 16);
            sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i + 16), loadq);
            sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i + 16), loadq);
            sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i + 16), loadq);
            sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i + 16), loadq);
            loadq = vld1q_s8(q + i + 32);
            sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i + 32), loadq);
            sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i + 32), loadq);
            sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i + 32), loadq);
            sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i + 32), loadq);
            loadq = vld1q_s8(q + i + 48);
            sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i + 48), loadq);
            sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i + 48), loadq);
            sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i + 48), loadq);
            sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i + 48), loadq);
        }
        for (; i < stride; i += block_bytes) {
            loadq = vld1q_s8(q + i);
            sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i), loadq);
            sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i), loadq);
            sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i), loadq);
            sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i), loadq);
        }
        return {vaddvq_s32(sums0), vaddvq_s32(sums1), vaddvq_s32(sums2), vaddvq_s32(sums3)};
    }

    static void cached_distances(const int8_t *q, int query_norm, int count, uint32_t stride, const int8_t *codes,
                                 const int *norms, float *out) {
        const auto qnorm = vdupq_n_s32(query_norm);
        for (int i = 0; i < count; i += 4, codes += 4 * size_t(stride))
            score4(q, codes, qnorm, norms + i, out + i, stride);
    }

  private:
    static void write_scores(int32x4_t dots, int32x4_t qnorm, const int *norms, float *out) {
        const auto squared = vsubq_s32(vaddq_s32(qnorm, vld1q_s32(norms)), vshlq_n_s32(dots, 1));
        vst1q_f32(out, vcvtq_f32_s32(squared));
    }

    static void score4(const int8_t *q, const int8_t *codes, int32x4_t qnorm,
                       const int *norms, float *out, size_t stride) {
        const auto zero = vdupq_n_s32(0);
        auto s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        const auto step = [&](size_t j) {
            const auto query = vld1q_s8(q + j);
            const auto rows = vld1q_s8_x4(codes + 4 * j);
            s0 = vdotq_s32(s0, rows.val[0], query);
            s1 = vdotq_s32(s1, rows.val[1], query);
            s2 = vdotq_s32(s2, rows.val[2], query);
            s3 = vdotq_s32(s3, rows.val[3], query);
        };
        size_t j = 0;
        const size_t unroll_end = stride & ~size_t{63};
        for (; j < unroll_end; j += 64) {
            step(j);
            step(j + 16);
            step(j + 32);
            step(j + 48);
        }
        for (; j < stride; j += block_bytes)
            step(j);
        write_scores(vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3)), qnorm, norms, out);
    }
};

} // namespace rnndescent
