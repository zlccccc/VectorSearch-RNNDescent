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
#include <string.h>
#include <vector>

#include "MyDistanceComputer.h"

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
inline int32_t CorrelationSum(const int8_t *a, const int8_t *b, size_t size) {
    int32x4_t sums = vdupq_n_s32(0);
    // FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < size; i += 16) {
        sums = vdotq_s32(sums, Load(a + i), Load(b + i));
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
    for (size_t i = 0; i < size; i += 64) {
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

    explicit SimdDistanceComputerInt8L2(const float *matrix, int n, int d) : n(n), d(d) {
        (this->matrix).resize(n * d);
        // scale = std::max(scale, *std::max_element(matrix, matrix + n * d));
        mean.resize(d);
#pragma omp parallel for
        for (int k = 0; k < d; k++) {
            float minvalue = INFINITY, maxvalue = -INFINITY;
            for (int i = 0; i < n; i++) {
                maxvalue = std::max(maxvalue, matrix[i * d + k]);
                minvalue = std::min(minvalue, matrix[i * d + k]);
            }
            mean[k] = (maxvalue + minvalue) / 2;
#pragma single
            { scale = std::max(scale, (maxvalue - minvalue) / 2); }
        }
        // scale = (maxvalue - minvalue) / 2.0;
        // mean = (maxvalue + minvalue) / 2.0;
        scale /= maxscale;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * maxscale);
        scale2 = scale * scale;
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            cvt_fp32_to_int8(matrix + i * d, this->matrix.data() + i * d);
        }

        matrixl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            matrixl2norms[i] = 0;
            for (int k = 0; k < d; k++) {
                matrixl2norms[i] += (matrix[i * d + k] - mean[k]) * (matrix[i * d + k] - mean[k]);
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

    void copy_index(int idx0, int8_t *y, float &l2norm) override final {
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

    Int8Neighbors() = default;
    Int8Neighbors(int dim, std::vector<int> &neighbor, MyDistanceComputer *dis, std::vector<int> &rollback_ids, bool save_neighbor)
        : Base(dim, neighbor, dis, rollback_ids, save_neighbor) {
        reorder_block(size, this->dim, pool);
    }

    static void reorder_block(int size, int dim, int8_t *pool) {
        if (pool == nullptr)
            return;
#pragma omp parallel for
        for (int base = 0; base < size; base += 4) {
            int8_t *x = pool + base * dim;
            std::vector<int8_t> reorder_value(4 * dim);
            for (int i = 0; i < dim; i += 16) {
                for (int id = 0; id < 4; id++) {
                    memcpy(reorder_value.data() + i * 4 + id * 16, x + id * dim + i, 16);
                }
            }
            memcpy(x, reorder_value.data(), 4 * dim);
        }
    }

    void compute_distance(int idq, std::vector<float> &result) { // 计算所有距离
        result.resize(size);
        if (pool != nullptr) {
            const int8_t *__restrict q = dis->get_query_ptr_int8(idq);
            distance_preloaded_all(q, size, pool, l2norms, result.data());
        } else {
            for (int i = 0; i < size; i += 4)
                dis->distances_batch_4(idq, edges[i], edges[i + 1], edges[i + 2], edges[i + 3], result[i], result[i + 1], result[i + 2], result[i + 3]);
        }
        // for (int id = 0; id < size; id++)
        //     printf("%f ", result[id]);
        // puts("<- dis");
    }

    // 这个地方已经加上了preload
    inline void distance_preloaded_all(const int8_t *__restrict q, const int size, const int8_t *__restrict xvalue, const float *__restrict l2norms,
                                       float *dis) {
        // #pragma omp simd
        // __builtin_prefetch(dis, 1, 2);
        for (int id = 0; id < size; id += 4) {
            const int8_t *__restrict x = xvalue + id * dim;
            // __builtin_prefetch(q, 0, 2);
            // __builtin_prefetch(x, 0, 2);

            // int8x16_t loadq;
            int32x4_t sums0 = vdupq_n_s32(0);
            int32x4_t sums1 = vdupq_n_s32(0);
            int32x4_t sums2 = vdupq_n_s32(0);
            int32x4_t sums3 = vdupq_n_s32(0);
            for (int i = 0; i < dim; i += 64) {
                // __builtin_prefetch(q + i + 64, 0, 2);
                // #pragma unroll
                for (int k = 0; k < 64; k += 16) {
                    // __builtin_prefetch(x + (i + k) * 4 + 64, 0, 2);
                    int8x16_t loadq = vld1q_s8(q + i + k);

                    sums0 = vdotq_s32(sums0, vld1q_s8(x + (i + k) * 4), loadq);
                    sums1 = vdotq_s32(sums1, vld1q_s8(x + (i + k) * 4 + 16), loadq);
                    sums2 = vdotq_s32(sums2, vld1q_s8(x + (i + k) * 4 + 32), loadq);
                    sums3 = vdotq_s32(sums3, vld1q_s8(x + (i + k) * 4 + 48), loadq);

                    // int8x16x4_t loadx = vld4q_s8(x + (i + k) * 4);  // 用这个的话reorder要修一下~~
                    // sums0 = vdotq_s32(sums0, loadx.val[0], loadq);
                    // sums1 = vdotq_s32(sums1, loadx.val[1], loadq);
                    // sums2 = vdotq_s32(sums2, loadx.val[2], loadq);
                    // sums3 = vdotq_s32(sums3, loadx.val[3], loadq);
                }
            }
            //             for (int i = 0, j = 0; i < d; i += 16, j += 64) {
            //                 // __builtin_prefetch(q + i + 64, 0, 2);
            // // #pragma unroll
            //                 int8x16_t loadq = vld1q_s8(q + i);
            //                 sums0 = vdotq_s32(sums0, vld1q_s8(x + j), loadq);
            //                 sums1 = vdotq_s32(sums1, vld1q_s8(x + j + 16), loadq);
            //                 sums2 = vdotq_s32(sums2, vld1q_s8(x + j + 32), loadq);
            //                 sums3 = vdotq_s32(sums3, vld1q_s8(x + j + 48), loadq);
            //             }
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
