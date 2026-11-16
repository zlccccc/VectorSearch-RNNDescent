#pragma once
// #include "faiss_platform_macros.h"
// #include <faiss/impl/DistanceComputer.h>
// #include "simd/dot.h"
#include <arm_neon.h>

#include "../Logger.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/prefetch.h>
#include <cstring>
#include <vector>

#include "MyDistanceComputer.h"
#include "NeighborsStorageBase.h"

namespace {

inline int8x16_t Load(const int8_t *p) {
    // #ifdef __GNUC__
    //   __builtin_prefetch(p + 128);
    // #endif
    // prefetch_L2(p + 256);
    __builtin_prefetch(p + 64, 0, 2);
    return vld1q_s8(p);
}

// FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
constexpr size_t neon_int8_block_bytes = 16;

inline int8x16_t LoadInt8Tail(const int8_t *data, size_t count) {
    int8_t tail[neon_int8_block_bytes]{};
    std::memcpy(tail, data, count);
    return vld1q_s8(tail);
}

inline int32_t CorrelationSum(const int8_t *a, const int8_t *b, size_t size) {
    int32x4_t sums = vdupq_n_s32(0);
    // FAISS_PRAGMA_IMPRECISE_LOOP
    const size_t full_size = size - size % neon_int8_block_bytes;
    for (size_t i = 0; i < full_size; i += neon_int8_block_bytes) {
        sums = vdotq_s32(sums, Load(a + i), Load(b + i));
    }
    if (full_size < size) {
        const size_t tail = size - full_size;
        sums = vdotq_s32(sums, LoadInt8Tail(a + full_size, tail), LoadInt8Tail(b + full_size, tail));
    }
    return vaddvq_s32(sums);
}
// FAISS_PRAGMA_IMPRECISE_FUNCTION_END

inline void CorrelationSum4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2, const int8_t *x3, size_t size, int32_t &res0, int32_t &res1,
                            int32_t &res2, int32_t &res3) {
    int8x16_t loadq;
    int32x4_t sums0 = vdupq_n_s32(0);
    int32x4_t sums1 = vdupq_n_s32(0);
    int32x4_t sums2 = vdupq_n_s32(0);
    int32x4_t sums3 = vdupq_n_s32(0);
    // FAISS_PRAGMA_IMPRECISE_LOOP
    // for (size_t i = 0; i < size; i += 16) {
    //     loadq = Load(q + i);
    //     sums0 = vaddq_s32(sums0, Correlation(Load(x0 + i), loadq));
    //     sums1 = vaddq_s32(sums1, Correlation(Load(x1 + i), loadq));
    //     sums2 = vaddq_s32(sums2, Correlation(Load(x2 + i), loadq));
    //     sums3 = vaddq_s32(sums3, Correlation(Load(x3 + i), loadq));
    // }
    const size_t full_size = size - size % neon_int8_block_bytes;
    const size_t unrolled_size = size - size % 64;
    size_t i = 0;
    for (; i < unrolled_size; i += 64) {
        __builtin_prefetch(q + i + 64, 0, 2);
        __builtin_prefetch(x0 + i + 64, 0, 2);
        __builtin_prefetch(x1 + i + 64, 0, 2);
        __builtin_prefetch(x2 + i + 64, 0, 2);
        __builtin_prefetch(x3 + i + 64, 0, 2);
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
    for (; i < full_size; i += neon_int8_block_bytes) {
        loadq = vld1q_s8(q + i);
        sums0 = vdotq_s32(sums0, vld1q_s8(x0 + i), loadq);
        sums1 = vdotq_s32(sums1, vld1q_s8(x1 + i), loadq);
        sums2 = vdotq_s32(sums2, vld1q_s8(x2 + i), loadq);
        sums3 = vdotq_s32(sums3, vld1q_s8(x3 + i), loadq);
    }
    if (full_size < size) {
        const size_t tail = size - full_size;
        loadq = LoadInt8Tail(q + i, tail);
        sums0 = vdotq_s32(sums0, LoadInt8Tail(x0 + i, tail), loadq);
        sums1 = vdotq_s32(sums1, LoadInt8Tail(x1 + i, tail), loadq);
        sums2 = vdotq_s32(sums2, LoadInt8Tail(x2 + i, tail), loadq);
        sums3 = vdotq_s32(sums3, LoadInt8Tail(x3 + i, tail), loadq);
    }
    res0 = vaddvq_s32(sums0);
    res1 = vaddvq_s32(sums1);
    res2 = vaddvq_s32(sums2);
    res3 = vaddvq_s32(sums3);
}

inline void CorrelationSub(int size, const float *l, float *res) {
    float32x4_t vres;
    for (int i = 0; i < size; i += 4) {
        vres = vsubq_f32(vld1q_f32(l + i), vld1q_f32(res + i));
        vst1q_f32(res + i, vres);
    }
}

} // namespace

namespace rnndescent {

struct SimdDistanceComputerInt8L2 : MyDistanceComputer {
    size_t n, d;
    std::vector<int8_t> matrix;
    std::vector<int8_t> query;
    const int maxscale = 127;
    // const int maxscale = 15;
    float scale = 0, scale2 = 0;
    std::vector<float> matrixl2norms;
    std::vector<float> mean;

    inline void scaled_dot(const int8_t *a, const int8_t *b, float &res) { res = CorrelationSum(a, b, d); }

    inline void scaled_dot_batch4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2, const int8_t *x3, size_t size, float &res0,
                                  float &res1, float &res2, float &res3) {
        int32_t dp0, dp1, dp2, dp3;
        CorrelationSum4(q, x0, x1, x2, x3, size, dp0, dp1, dp2, dp3);
        res0 = dp0;
        res1 = dp1;
        res2 = dp2;
        res3 = dp3;
    }

    inline void vector_sub(int size, const float *l2norms, std::vector<float> &dis) {
        const float *__restrict l = l2norms;
        CorrelationSub(size, l, dis.data());
    }

    // inline void add_l2(const float *l2, const float *)

    inline void cvt_fp32_to_int8(const float *a, int8_t *b) {
#pragma simd
        for (int k = 0; k < d; k++) {
            b[k] = std::max(-maxscale, std::min(maxscale, (int)std::round((a[k] - mean[k]) / scale)));
        }
    }

    explicit SimdDistanceComputerInt8L2(const float *matrix, int n, int d, const int *row_order = nullptr) : n(n), d(d) {
        (this->matrix).resize(n * d);
        mean.resize(d);
        float max_range = 0.0f;
#pragma omp parallel for reduction(max : max_range)
        for (int k = 0; k < d; k++) {
            float minvalue = INFINITY, maxvalue = -INFINITY;
            for (int i = 0; i < n; i++) {
                maxvalue = std::max(maxvalue, matrix[i * d + k]);
                minvalue = std::min(minvalue, matrix[i * d + k]);
            }
            mean[k] = (maxvalue + minvalue) / 2;
            max_range = std::max(max_range, (maxvalue - minvalue) / 2);
        }
        scale = max_range > 0.0f ? max_range / maxscale : 1.0f;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * maxscale);
        scale2 = scale * scale;
        matrixl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const auto *source = matrix + static_cast<size_t>(row_order ? row_order[i] : i) * d;
            cvt_fp32_to_int8(source, this->matrix.data() + i * d);
            matrixl2norms[i] = 0;
            for (int k = 0; k < d; k++) {
                matrixl2norms[i] += (source[k] - mean[k]) * (source[k] - mean[k]);
            }
            matrixl2norms[i] /= scale2 * 2;
        }
    }

    int row_count() const override { return static_cast<int>(n); }

    int dimension() const override { return static_cast<int>(d); }

    float operator()(int idq, int i) final override {
        const int8_t *__restrict x = query.data() + idq * d;
        const int8_t *__restrict y = matrix.data() + i * d;
        return matrixl2norms[i] - CorrelationSum(x, y, d);
    }

    float symmetric_dis(int i, int j) final override {
        const int8_t *__restrict x0 = matrix.data() + i * d;
        const int8_t *__restrict x1 = matrix.data() + j * d;
        return matrixl2norms[i] + matrixl2norms[j] - CorrelationSum(x0, x1, d);
    }

    void set_query(const float *x, int n) override {
        query.resize(d * n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            cvt_fp32_to_int8(x + i * d, query.data() + i * d);
        }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) override final {
        const int8_t *__restrict q = query.data() + idq * d;
        const int8_t *__restrict x0 = matrix.data() + idx0 * d;
        const int8_t *__restrict x1 = matrix.data() + idx1 * d;
        const int8_t *__restrict x2 = matrix.data() + idx2 * d;
        const int8_t *__restrict x3 = matrix.data() + idx3 * d;

        // TODO: prefetch
        // prefetch_L2(matrixl2norms.data() + idx0);
        // prefetch_L2(matrixl2norms.data() + idx1);
        // prefetch_L2(matrixl2norms.data() + idx2);
        // prefetch_L2(matrixl2norms.data() + idx3);
        float32_t dp0, dp1, dp2, dp3;
        scaled_dot_batch4(q, x0, x1, x2, x3, d, dp0, dp1, dp2, dp3);
        dis0 = matrixl2norms[idx0] - dp0;
        dis1 = matrixl2norms[idx1] - dp1;
        dis2 = matrixl2norms[idx2] - dp2;
        dis3 = matrixl2norms[idx3] - dp3;
    }

    void copy_index(int idx0, void *y, float &l2norm) override final {
        memcpy(y, matrix.data() + idx0 * d, d);
        l2norm = matrixl2norms[idx0];
    }

    int8_t *get_query_ptr_int8(int idx0) override final { return query.data() + idx0 * d; }
    ~SimdDistanceComputerInt8L2() override {}
};

// neighbor number, dim
struct Int8Neighbors : NeighborsStorageBase<Int8Neighbors, int8_t, float> { // 全都变成offset!
    using Base = NeighborsStorageBase<Int8Neighbors, int8_t, float>;
    using Base::dim;
    using Base::dis;
    using Base::edges;
    using Base::l2norms;
    using Base::pool;
    using Base::size;

    static constexpr size_t cache_lane_width = neon_int8_block_bytes;
    using Base::Base;

    void compute_distance(int idq, std::vector<float> &result) {
        result.resize(size);
        if (pool != nullptr) {
            const int8_t *__restrict q = dis->get_query_ptr_int8(idq);
            distance_preloaded_all(q, size, pool, l2norms, result.data());
        } else {
            for (int i = 0; i < size; i += 4)
                dis->distances_batch_4(idq, edges[i], edges[i + 1], edges[i + 2], edges[i + 3], result[i], result[i + 1], result[i + 2], result[i + 3]);
        }
    }

    // 这个地方已经加上了preload
    inline void distance_preloaded_all(const int8_t *__restrict q, const int size, const int8_t *__restrict xvalue, const float *__restrict l2norms,
                                       float *dis) {
        const size_t dimension = dim;
        const size_t tail = dimension % cache_lane_width;
        const size_t full_size = dimension - tail;
        const size_t unrolled_size = dimension - dimension % (4 * cache_lane_width);
        for (int id = 0; id < size; id += 4) {
            const int8_t *__restrict x = xvalue + id * dim;
            // __builtin_prefetch(q, 0, 2);
            // __builtin_prefetch(x, 0, 2);

            int32x4_t sums0 = vdupq_n_s32(0);
            int32x4_t sums1 = vdupq_n_s32(0);
            int32x4_t sums2 = vdupq_n_s32(0);
            int32x4_t sums3 = vdupq_n_s32(0);
            const auto dot_chunk = [&](size_t k) {
                const int8x16_t loadq = vld1q_s8(q + k);
                const int8_t *chunk = x + 4 * k;
                sums0 = vdotq_s32(sums0, vld1q_s8(chunk), loadq);
                sums1 = vdotq_s32(sums1, vld1q_s8(chunk + cache_lane_width), loadq);
                sums2 = vdotq_s32(sums2, vld1q_s8(chunk + 2 * cache_lane_width), loadq);
                sums3 = vdotq_s32(sums3, vld1q_s8(chunk + 3 * cache_lane_width), loadq);
            };
            size_t k = 0;
            for (; k < unrolled_size; k += 4 * cache_lane_width) {
                dot_chunk(k);
                dot_chunk(k + cache_lane_width);
                dot_chunk(k + 2 * cache_lane_width);
                dot_chunk(k + 3 * cache_lane_width);
            }
            for (; k < full_size; k += cache_lane_width)
                dot_chunk(k);
            if (tail != 0) {
                const int8x16_t loadq = LoadInt8Tail(q + full_size, tail);
                const int8_t *chunk = x + 4 * full_size;
                sums0 = vdotq_s32(sums0, LoadInt8Tail(chunk, tail), loadq);
                sums1 = vdotq_s32(sums1, LoadInt8Tail(chunk + tail, tail), loadq);
                sums2 = vdotq_s32(sums2, LoadInt8Tail(chunk + 2 * tail, tail), loadq);
                sums3 = vdotq_s32(sums3, LoadInt8Tail(chunk + 3 * tail, tail), loadq);
            }
            dis[id] = vaddvq_s32(sums0);
            dis[id + 1] = vaddvq_s32(sums1);
            dis[id + 2] = vaddvq_s32(sums2);
            dis[id + 3] = vaddvq_s32(sums3);
        }
        // vector_sub(size, l2norms, dis);
        // CorrelationSub(size, l2norms, dis);
        float32x4_t vres;
        for (int i = 0; i < size; i += 4) {
            vres = vsubq_f32(vld1q_f32(l2norms + i), vld1q_f32(dis + i));
            vst1q_f32(dis + i, vres);
        }
        // dis = temp;
    }
};

} // namespace rnndescent
