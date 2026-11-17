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

    static void cached_distances(const int8_t *__restrict q, int query_norm, int size,
                                 uint32_t code_stride,
                                 const int8_t *__restrict xvalue, const int *__restrict l2norms, float *result) {
        const size_t unrolled_size = code_stride - code_stride % (4 * block_bytes);
        for (int id = 0; id < size; id += 4) {
            const int8_t *__restrict x = xvalue + static_cast<size_t>(id) * code_stride;

            int32x4_t sums0 = vdupq_n_s32(0);
            int32x4_t sums1 = vdupq_n_s32(0);
            int32x4_t sums2 = vdupq_n_s32(0);
            int32x4_t sums3 = vdupq_n_s32(0);
            const auto dot_chunk = [&](size_t k) {
                const int8x16_t loadq = vld1q_s8(q + k);
                const int8_t *chunk = x + 4 * k;
                sums0 = vdotq_s32(sums0, vld1q_s8(chunk), loadq);
                sums1 = vdotq_s32(sums1, vld1q_s8(chunk + block_bytes), loadq);
                sums2 = vdotq_s32(sums2, vld1q_s8(chunk + 2 * block_bytes), loadq);
                sums3 = vdotq_s32(sums3, vld1q_s8(chunk + 3 * block_bytes), loadq);
            };
            size_t k = 0;
            for (; k < unrolled_size; k += 4 * block_bytes) {
                dot_chunk(k);
                dot_chunk(k + block_bytes);
                dot_chunk(k + 2 * block_bytes);
                dot_chunk(k + 3 * block_bytes);
            }
            for (; k < code_stride; k += block_bytes)
                dot_chunk(k);
            result[id] = static_cast<float>(query_norm + l2norms[id] - 2 * vaddvq_s32(sums0));
            result[id + 1] = static_cast<float>(query_norm + l2norms[id + 1] - 2 * vaddvq_s32(sums1));
            result[id + 2] = static_cast<float>(query_norm + l2norms[id + 2] - 2 * vaddvq_s32(sums2));
            result[id + 3] = static_cast<float>(query_norm + l2norms[id + 3] - 2 * vaddvq_s32(sums3));
        }
    }
};

using SimdDistanceComputerInt8L2 = QuantizedDistanceComputer<NeonInt8DotProduct>;
using Int8Neighbors = QuantizedNeighbors<NeonInt8DotProduct>;

} // namespace rnndescent
