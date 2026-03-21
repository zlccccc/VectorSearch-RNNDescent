#pragma once
#include <immintrin.h> //AVX(include wmmintrin.h)
#include <wmmintrin.h>
#include <x86intrin.h> //(包含所有相关的头文件)

#include "MyDistanceComputer.h"
#include "NeighborsStorageBase.h"
#include "utils.h"
#include "../Logger.h"
#include <algorithm>
#include <cmath>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/prefetch.h>
#include <fstream>
#include <string.h>
#include <vector>

namespace {

#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

inline int32_t ReduceAddEpi32(const __m256i &sums) {
    __m128i ab_sum = _mm_add_epi32(_mm256_extracti128_si256(sums, 0), _mm256_extracti128_si256(sums, 1));
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    return _mm_extract_epi32(ab_sum, 0);
}

inline __m256i Correlation(const __m128i &a, const __m128i &b) { return _mm256_madd_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)); }

inline int32_t CorrelationSum(const int8_t *a, const int8_t *b, size_t size) {
    __m256i sums = _mm256_setzero_si256();
    for (size_t i = 0; i < size; i += 16) { // length = 16
        sums = _mm256_add_epi32(sums, Correlation(_mm_loadu_si128((__m128i const *)(a + i)), _mm_loadu_si128((__m128i const *)(b + i))));
    }
    return ReduceAddEpi32(sums);
}

inline void CorrelationSum4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2, const int8_t *x3, size_t size, int32_t &res0, int32_t &res1,
                            int32_t &res2, int32_t &res3) {
    __m128i loadq;
    __m256i sums0 = _mm256_setzero_si256();
    __m256i sums1 = _mm256_setzero_si256();
    __m256i sums2 = _mm256_setzero_si256();
    __m256i sums3 = _mm256_setzero_si256();
    for (size_t i = 0; i < size; i += 64) { // 64 cacheline / sizeof(__m128i)
        __builtin_prefetch(q + i + 64, 0, 2);
        __builtin_prefetch(x0 + i + 64, 0, 2);
        __builtin_prefetch(x1 + i + 64, 0, 2);
        __builtin_prefetch(x2 + i + 64, 0, 2);
        __builtin_prefetch(x3 + i + 64, 0, 2);
        loadq = _mm_loadu_si128((__m128i const *)(q + i + 00));
        sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 00)), loadq));
        sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 00)), loadq));
        sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 00)), loadq));
        sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 00)), loadq));
        loadq = _mm_loadu_si128((__m128i const *)(q + i + 16));
        sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 16)), loadq));
        sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 16)), loadq));
        sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 16)), loadq));
        sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 16)), loadq));
        loadq = _mm_loadu_si128((__m128i const *)(q + i + 32));
        sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 32)), loadq));
        sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 32)), loadq));
        sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 32)), loadq));
        sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 32)), loadq));
        loadq = _mm_loadu_si128((__m128i const *)(q + i + 48));
        sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x0 + i + 48)), loadq));
        sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x1 + i + 48)), loadq));
        sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x2 + i + 48)), loadq));
        sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x3 + i + 48)), loadq));
    }
    res0 = ReduceAddEpi32(sums0);
    res1 = ReduceAddEpi32(sums1);
    res2 = ReduceAddEpi32(sums2);
    res3 = ReduceAddEpi32(sums3);

    // int s0=0, s1=0, s2=0, s3=0;
    // for (int i = 0; i < size; i ++) {
    //     s0 += (int)q[i] * x0[i];
    //     s1 += (int)q[i] * x1[i];
    //     s2 += (int)q[i] * x2[i];
    //     s3 += (int)q[i] * x3[i];
    // }
    // printf("%d %d %d %d == %d %d %d %d\n", s0, s1, s2, s3, res0, res1, res2, res3);
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
    std::vector<float> queryl2norms;
    std::vector<float> mean; // 非对称量化

    inline void scaled_dot_batch4(const int8_t *q, const int8_t *x0, const int8_t *x1, const int8_t *x2, const int8_t *x3, size_t size, float &res0,
                                  float &res1, float &res2, float &res3) {
        int32_t dp0, dp1, dp2, dp3;
        CorrelationSum4(q, x0, x1, x2, x3, size, dp0, dp1, dp2, dp3);
        res0 = dp0;
        res1 = dp1;
        res2 = dp2;
        res3 = dp3;
    }

    inline void cvt_fp32_to_int8(const float *a, int8_t *b) {
#pragma simd
        for (int k = 0; k < d; k++) {
            b[k] = std::max(-maxscale, std::min(maxscale, (int)std::round((a[k] - mean[k]) / scale)));
        }
    }

    explicit SimdDistanceComputerInt8L2(const float *matrix, int n, int d) : n(n), d(d) {
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
            {
                scale = std::max(scale, (maxvalue - minvalue) / 2);
            }
        }
        // scale = (maxvalue - minvalue) / 2.0;
        // mean = (maxvalue + minvalue) / 2.0;
        scale /= maxscale;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * maxscale);
        scale2 = scale * scale;

        (this->matrix).resize(n * d);
        matrixl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            cvt_fp32_to_int8(matrix + i * d, this->matrix.data() + i * d);
            matrixl2norms[i] = 0;
            for (int k = 0; k < d; k++)
                matrixl2norms[i] += (matrix[i * d + k] - mean[k]) * (matrix[i * d + k] - mean[k]);
            matrixl2norms[i] /= scale2 * 2;
        }

        // for (int i = 0; i < 10; i++) {
        //     for (int j = 0; j < 10; j++) {
        //         float real = 0;
        //         for (int k = 0; k < d; k++) real += (matrix[i * d + k] - matrix[j * d + k]) * (matrix[i * d + k] - matrix[j * d + k]);
        //         printf("[%f - %f] ",this->symmetric_dis(i, j), real);
        //     }
        // }
        // puts("");
        // assert(0);
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
        float dp0, dp1, dp2, dp3;
        scaled_dot_batch4(q, x0, x1, x2, x3, d, dp0, dp1, dp2, dp3);
        dis0 = matrixl2norms[idx0] - dp0;
        dis1 = matrixl2norms[idx1] - dp1;
        dis2 = matrixl2norms[idx2] - dp2;
        dis3 = matrixl2norms[idx3] - dp3;
        // printf("%f %f - %f * 2\n", queryl2norms[idq], matrixl2norms[idx0], dp0);
    }

    void copy_index(int idx0, void *y, float &l2norm) override final {
        memcpy(y, matrix.data() + idx0 * d, d);
        l2norm = matrixl2norms[idx0];
    }

    void *get_query_ptr(int idx0) override final { return query.data() + idx0 * d; }

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
        // if (false && pool != nullptr) {
        if (pool != nullptr) {
            const int8_t *__restrict q = (int8_t *)dis->get_query_ptr(idq);
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

            __m256i sums0 = _mm256_setzero_si256();
            __m256i sums1 = _mm256_setzero_si256();
            __m256i sums2 = _mm256_setzero_si256();
            __m256i sums3 = _mm256_setzero_si256();
            // for (int i = 0; i < dim; i += 16) {
            //     // __builtin_prefetch(q + i + 64, 0, 2);
            //     // __builtin_prefetch(x + i * 4 + 64, 0, 2);
            //     __m128i loadq = _mm_loadu_si128((__m128i const *)(q + i));
            //     sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x + i * 4 + 0)), loadq));
            //     sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x + i * 4 + 16)), loadq));
            //     sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x + i * 4 + 32)), loadq));
            //     sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x + i * 4 + 48)), loadq));
            // }
            for (int i = 0; i < dim; i += 64) {
                __builtin_prefetch(q + i + 64, 0, 2);
#pragma unroll
                for (int k = 0; k < 64; k += 16) {
                    __builtin_prefetch(x + (i + k) * 4 + 64, 0, 2);
                    __m128i loadq = _mm_loadu_si128((__m128i const *)(q + i + k));
                    sums0 = _mm256_add_epi32(sums0, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 0)), loadq));
                    sums1 = _mm256_add_epi32(sums1, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 16)), loadq));
                    sums2 = _mm256_add_epi32(sums2, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 32)), loadq));
                    sums3 = _mm256_add_epi32(sums3, Correlation(_mm_loadu_si128((__m128i const *)(x + (i + k) * 4 + 48)), loadq));
                }
            }
            dis[id] = ReduceAddEpi32(sums0);
            dis[id + 1] = ReduceAddEpi32(sums1);
            dis[id + 2] = ReduceAddEpi32(sums2);
            dis[id + 3] = ReduceAddEpi32(sums3);
            // printf("%f %f %f %f; %f %f\n", dis[id], dis[id + 1], dis[id + 2], dis[id + 3], l2norms[id], l2normq);
        }
        // float32x4_t vres;
        // for (int i = 0; i < size; i += 4) {
        //     vres = vsubq_f32(vld1q_f32(l2norms + i), vld1q_f32(dis + i));
        //     vst1q_f32(dis + i, vres);
        // }
        for (int i = 0; i < size; i++)
            dis[i] = l2norms[i] - dis[i];
        // for (int id = 0; id < size; id++)
        //     printf("%f ", dis[id]);
        // puts("<- dis");
        // dis = temp;
    }
};

} // namespace rnndescent
