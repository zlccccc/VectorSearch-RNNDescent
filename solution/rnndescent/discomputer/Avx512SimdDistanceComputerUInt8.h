#pragma once
#include <immintrin.h> //AVX(include wmmintrin.h)
#include <wmmintrin.h>
// #include <intrin.h> //(包含所有相关的头文件)

#include "MyDistanceComputer.h"
#include "NeighborsStorageBase.h"
#include "utils.h"
#include "../Logger.h"
#include <algorithm>
#include <cmath>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/prefetch.h>
#include <string.h>
#include <vector>

namespace {

#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), apply_to = function)

constexpr size_t avx512_int8_block_bytes = sizeof(__m512i);

inline __mmask64 low_byte_mask(size_t count) {
    return (uint64_t{1} << count) - 1;
}

inline int32_t CorrelationSum(const uint8_t *a, const uint8_t *b, size_t size) {
    __m512i sums = _mm512_setzero_si512();
    const size_t tail = size % avx512_int8_block_bytes;
    const size_t full_size = size - tail;
    for (size_t i = 0; i < full_size; i += avx512_int8_block_bytes) {
        sums = _mm512_dpbusds_epi32(sums, _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
    }
    if (tail != 0) {
        const auto mask = low_byte_mask(tail);
        sums = _mm512_dpbusds_epi32(sums, _mm512_maskz_loadu_epi8(mask, a + full_size),
                                   _mm512_maskz_loadu_epi8(mask, b + full_size));
    }
    return _mm512_reduce_add_epi32(sums);
}

inline void CorrelationSum4(const uint8_t *q, const uint8_t *x0, const uint8_t *x1, const uint8_t *x2, const uint8_t *x3, size_t size, int32_t &res0,
                            int32_t &res1, int32_t &res2, int32_t &res3) {
    __builtin_prefetch(q, 0, 2);
    __builtin_prefetch(x0, 0, 2);
    __builtin_prefetch(x1, 0, 2);
    __builtin_prefetch(x2, 0, 2);
    __builtin_prefetch(x3, 0, 2);
    __m512i loadq;
    __m512i sums0 = _mm512_setzero_si512();
    __m512i sums1 = _mm512_setzero_si512();
    __m512i sums2 = _mm512_setzero_si512();
    __m512i sums3 = _mm512_setzero_si512();
    const size_t tail = size % avx512_int8_block_bytes;
    const size_t full_size = size - tail;
    for (size_t i = 0; i < full_size; i += avx512_int8_block_bytes) {
        __builtin_prefetch(q + i + 64, 0, 2);
        __builtin_prefetch(x0 + i + 64, 0, 2);
        __builtin_prefetch(x1 + i + 64, 0, 2);
        __builtin_prefetch(x2 + i + 64, 0, 2);
        __builtin_prefetch(x3 + i + 64, 0, 2);
        loadq = _mm512_loadu_si512(q + i);
        sums0 = _mm512_dpbusds_epi32(sums0, _mm512_loadu_si512(x0 + i), loadq);
        sums1 = _mm512_dpbusds_epi32(sums1, _mm512_loadu_si512(x1 + i), loadq);
        sums2 = _mm512_dpbusds_epi32(sums2, _mm512_loadu_si512(x2 + i), loadq);
        sums3 = _mm512_dpbusds_epi32(sums3, _mm512_loadu_si512(x3 + i), loadq);
    }
    if (tail != 0) {
        const auto mask = low_byte_mask(tail);
        loadq = _mm512_maskz_loadu_epi8(mask, q + full_size);
        sums0 = _mm512_dpbusds_epi32(sums0, _mm512_maskz_loadu_epi8(mask, x0 + full_size), loadq);
        sums1 = _mm512_dpbusds_epi32(sums1, _mm512_maskz_loadu_epi8(mask, x1 + full_size), loadq);
        sums2 = _mm512_dpbusds_epi32(sums2, _mm512_maskz_loadu_epi8(mask, x2 + full_size), loadq);
        sums3 = _mm512_dpbusds_epi32(sums3, _mm512_maskz_loadu_epi8(mask, x3 + full_size), loadq);
    }
    res0 = _mm512_reduce_add_epi32(sums0);
    res1 = _mm512_reduce_add_epi32(sums1);
    res2 = _mm512_reduce_add_epi32(sums2);
    res3 = _mm512_reduce_add_epi32(sums3);

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

struct SimdDistanceComputerUInt8L2 : MyDistanceComputer {
    size_t n, d;
    std::vector<uint8_t> matrix;
    std::vector<uint8_t> query;
    // const int maxscale = 127;
    const int maxscale = 63;
    // const int maxscale = 15;
    float scale = 0;
    std::vector<int> matrixl2norms;
    // std::vector<int> queryl2norms;
    std::vector<float> mean; // 非对称量化; 平均值

    explicit SimdDistanceComputerUInt8L2(const float *matrix, int n, int d) : n(n), d(d) {
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
        scale = max_range > 0 ? max_range / maxscale : 1.0f;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * maxscale);

        (this->matrix).resize(n * d);
        matrixl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            matrixl2norms[i] = 0;
            for (int k = 0; k < d; k++) {
                int normval = std::round((matrix[i * d + k] - mean[k]) / scale);
                this->matrix[i * d + k] = std::max(0, std::min(maxscale * 2, normval + maxscale));
                matrixl2norms[i] += this->matrix[i * d + k] * this->matrix[i * d + k];
            }
        }
    }

    int row_count() const override { return static_cast<int>(n); }

    int dimension() const override { return static_cast<int>(d); }

    float operator()(int idq, int i) final override {
        const uint8_t *__restrict x = query.data() + idq * d;
        const uint8_t *__restrict y = matrix.data() + i * d;
        return (matrixl2norms[i] - CorrelationSum(x, y, d) * 2);
    }

    float symmetric_dis(int i, int j) final override {
        const uint8_t *__restrict x0 = matrix.data() + i * d;
        const uint8_t *__restrict x1 = matrix.data() + j * d;
        return (matrixl2norms[i] + matrixl2norms[j] - CorrelationSum(x0, x1, d) * 2);
    }

    void set_query(const float *x, int n) override {
        query.resize(d * n);
        // queryl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            // queryl2norms[i] = 0;
            for (int k = 0; k < d; k++) {
                int normval = std::round((x[i * d + k] - mean[k]) / scale);
                this->query[i * d + k] = std::max(0, std::min(maxscale * 2, normval + maxscale));
                // queryl2norms[i] += this->query[i * d + k] * this->query[i * d + k];
            }
        }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) override final {
        const uint8_t *__restrict q = query.data() + idq * d;
        const uint8_t *__restrict x0 = matrix.data() + idx0 * d;
        const uint8_t *__restrict x1 = matrix.data() + idx1 * d;
        const uint8_t *__restrict x2 = matrix.data() + idx2 * d;
        const uint8_t *__restrict x3 = matrix.data() + idx3 * d;

        // TODO: prefetch
        // prefetch_L2(matrixl2norms.data() + idx0);
        // prefetch_L2(matrixl2norms.data() + idx1);
        // prefetch_L2(matrixl2norms.data() + idx2);
        // prefetch_L2(matrixl2norms.data() + idx3);
        int dp0, dp1, dp2, dp3;
        CorrelationSum4(q, x0, x1, x2, x3, d, dp0, dp1, dp2, dp3);
        dis0 = (matrixl2norms[idx0] - dp0 * 2);
        dis1 = (matrixl2norms[idx1] - dp1 * 2);
        dis2 = (matrixl2norms[idx2] - dp2 * 2);
        dis3 = (matrixl2norms[idx3] - dp3 * 2);
        // printf("%d + %d - %d * 2\n", matrixl2norms[idx0], queryl2norms[idq], dp0);
    }

    void copy_index(int idx0, void *y, int32_t &l2norm) override final {
        memcpy(y, matrix.data() + idx0 * d, d);
        l2norm = matrixl2norms[idx0];
    }

    void *get_query_ptr(int idx0) override final { return query.data() + idx0 * d; }

    ~SimdDistanceComputerUInt8L2() override {}
};

// neighbor number, dim
struct UInt8Neighbors : NeighborsStorageBase<UInt8Neighbors, uint8_t, int> { // 全都变成offset!
    using Base = NeighborsStorageBase<UInt8Neighbors, uint8_t, int>;
    using Base::dim;
    using Base::dis;
    using Base::edges;
    using Base::l2norms;
    using Base::pool;
    using Base::size;

    static constexpr size_t cache_lane_width = avx512_int8_block_bytes;
    using Base::Base;

    void compute_distance(int idq, std::vector<float> &result) { // 计算所有距离
        result.resize(size);
        // if (false && pool != nullptr) {
        if (pool != nullptr) {
            const uint8_t *__restrict q = (uint8_t *)dis->get_query_ptr(idq);
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
    inline void distance_preloaded_all(const uint8_t *__restrict q, const int size, const uint8_t *__restrict xvalue, const int *__restrict l2norms,
                                       float *dis) {
        // #pragma omp simd
        // __builtin_prefetch(dis, 1, 2);
        const size_t dimension = dim;
        const size_t tail = dimension % cache_lane_width;
        const size_t full_size = dimension - tail;
        for (int id = 0; id < size; id += 4) {
            const uint8_t *__restrict x = xvalue + static_cast<size_t>(id) * dimension;
            // __builtin_prefetch(q, 0, 2);
            // __builtin_prefetch(x, 0, 2);

            __m512i sums0 = _mm512_setzero_si512();
            __m512i sums1 = _mm512_setzero_si512();
            __m512i sums2 = _mm512_setzero_si512();
            __m512i sums3 = _mm512_setzero_si512();
            for (size_t i = 0; i < full_size; i += cache_lane_width) {
                const auto loadq = _mm512_loadu_si512(q + i);
                const auto *chunk = x + i * 4;
                sums0 = _mm512_dpbusds_epi32(sums0, _mm512_loadu_si512(chunk), loadq);
                sums1 = _mm512_dpbusds_epi32(sums1, _mm512_loadu_si512(chunk + cache_lane_width), loadq);
                sums2 = _mm512_dpbusds_epi32(sums2, _mm512_loadu_si512(chunk + 2 * cache_lane_width), loadq);
                sums3 = _mm512_dpbusds_epi32(sums3, _mm512_loadu_si512(chunk + 3 * cache_lane_width), loadq);
            }
            if (tail != 0) {
                const auto mask = low_byte_mask(tail);
                const auto loadq = _mm512_maskz_loadu_epi8(mask, q + full_size);
                const auto *chunk = x + full_size * 4;
                sums0 = _mm512_dpbusds_epi32(sums0, _mm512_maskz_loadu_epi8(mask, chunk), loadq);
                sums1 = _mm512_dpbusds_epi32(sums1, _mm512_maskz_loadu_epi8(mask, chunk + tail), loadq);
                sums2 = _mm512_dpbusds_epi32(sums2, _mm512_maskz_loadu_epi8(mask, chunk + 2 * tail), loadq);
                sums3 = _mm512_dpbusds_epi32(sums3, _mm512_maskz_loadu_epi8(mask, chunk + 3 * tail), loadq);
            }
            dis[id] = _mm512_reduce_add_epi32(sums0);
            dis[id + 1] = _mm512_reduce_add_epi32(sums1);
            dis[id + 2] = _mm512_reduce_add_epi32(sums2);
            dis[id + 3] = _mm512_reduce_add_epi32(sums3);
            // printf("%f %f %f %f; %f %f\n", dis[id], dis[id + 1], dis[id + 2], dis[id + 3], l2norms[id], l2normq);
        }
        for (int i = 0; i < size; i++)
            dis[i] = l2norms[i] - dis[i] * 2;
        // for (int id = 0; id < size; id++)
        //     printf("%f ", dis[id]);
        // puts("<- dis");
        // dis = temp;
    }
};

} // namespace rnndescent

#pragma clang attribute pop
#pragma GCC pop_options
