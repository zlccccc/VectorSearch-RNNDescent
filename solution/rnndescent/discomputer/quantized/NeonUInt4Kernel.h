#pragma once

#include "QuantizedKernelBase.h"
#include "UInt4Quantization.h"

#include <arm_neon.h>
#include <array>

namespace rnndescent {

struct NeonUInt4Kernel {
    static constexpr size_t block_bytes = 16;
    using Encoding = PackedUInt4Quantization<block_bytes>;
    using Tile4 = QuantizedKernelBase<Encoding, block_bytes, 4>;

    static void pack_neighbor_cache(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    uint8_t *codes, int *norms, uint8_t *scratch) {
        Tile4::pack_tiles(distance, ids, count, stride, codes, norms, scratch);
    }

    static int dot(const uint8_t *q, const uint8_t *row, size_t stride) {
        auto sum = vdupq_n_s32(0);
        for (size_t i = 0; i < stride; i += block_bytes) {
            const auto low = vreinterpretq_s8_u8(vld1q_u8(q + 2 * i));
            const auto high = vreinterpretq_s8_u8(vld1q_u8(q + 2 * i + block_bytes));
            sum = accumulate(sum, vld1q_u8(row + i), low, high);
        }
        return vaddvq_s32(sum);
    }

    static std::array<int, 4> dot4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                                   const uint8_t *x3, size_t stride) {
        std::array<int, 4> result;
        vst1q_s32(result.data(), dots4<false>(q, x0, x1, x2, x3, stride));
        return result;
    }

    static void cached_distances(const uint8_t *q, int query_norm, int count, uint32_t stride, const uint8_t *codes,
                                 const int *norms, float *out) {
        const auto qnorm = vdupq_n_s32(query_norm);
        for (int i = 0; i < count; i += 4, codes += 4 * size_t(stride)) {
            const auto dots = dots4<true>(q, codes, nullptr, nullptr, nullptr, stride);
            const auto squared = vsubq_s32(vaddq_s32(qnorm, vld1q_s32(norms + i)),
                                           vshlq_n_s32(dots, Encoding::twice_dot_shift));
            vst1q_f32(out + i, vcvtq_f32_s32(squared));
        }
    }

  private:
    static int32x4_t accumulate(int32x4_t sum, uint8x16_t packed, int8x16_t qlow, int8x16_t qhigh) {
        // Both nibbles fit in signed bytes, so SDOT needs no sign correction.
        const auto low = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(15)));
        const auto high = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
        return vdotq_s32(vdotq_s32(sum, low, qlow), high, qhigh);
    }

    template <bool Interleaved>
    static int32x4_t dots4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2,
                           const uint8_t *x3, size_t stride) {
        auto s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0;
        for (size_t i = 0; i < stride; i += block_bytes) {
            const auto low = vreinterpretq_s8_u8(vld1q_u8(q + 2 * i));
            const auto high = vreinterpretq_s8_u8(vld1q_u8(q + 2 * i + block_bytes));
            if constexpr (Interleaved) {
                const auto rows = vld1q_u8_x4(x0 + 4 * i);
                s0 = accumulate(s0, rows.val[0], low, high);
                s1 = accumulate(s1, rows.val[1], low, high);
                s2 = accumulate(s2, rows.val[2], low, high);
                s3 = accumulate(s3, rows.val[3], low, high);
            } else {
                s0 = accumulate(s0, vld1q_u8(x0 + i), low, high);
                s1 = accumulate(s1, vld1q_u8(x1 + i), low, high);
                s2 = accumulate(s2, vld1q_u8(x2 + i), low, high);
                s3 = accumulate(s3, vld1q_u8(x3 + i), low, high);
            }
        }
        return vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
    }
};

} // namespace rnndescent
