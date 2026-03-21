#pragma once

#include <immintrin.h>
#include "MyDistanceComputer.h"
#include "utils.h"
#include <cblas.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>
#include <vector>

namespace {

#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), apply_to = function)

inline float CorrelationSum(const float * a, const float* b, size_t n) {
    __m512 sums = _mm512_setzero_ps();
    for (size_t i = 0; i < n; i += 16) {
        sums = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), sums);
    }
    return _mm512_reduce_add_ps(sums);
}

inline void CorrelationSum4(const float *q, const float *x0, const float *x1, const float *x2, const float *x3, size_t size,
                                 float &res0, float &res1, float &res2, float &res3) {
    __m512 sums0 = _mm512_setzero_ps();
    __m512 sums1 = _mm512_setzero_ps();
    __m512 sums2 = _mm512_setzero_ps();
    __m512 sums3 = _mm512_setzero_ps();
    for (size_t i = 0; i < size; i += 16) {
        __m512 q_vec = _mm512_loadu_ps(&q[i]);
        __m512 x0_vec = _mm512_loadu_ps(&x0[i]);
        __m512 x1_vec = _mm512_loadu_ps(&x1[i]);
        __m512 x2_vec = _mm512_loadu_ps(&x2[i]);
        __m512 x3_vec = _mm512_loadu_ps(&x3[i]);
        sums0 = _mm512_fmadd_ps(q_vec, x0_vec, sums0);
        sums1 = _mm512_fmadd_ps(q_vec, x1_vec, sums1);
        sums2 = _mm512_fmadd_ps(q_vec, x2_vec, sums2);
        sums3 = _mm512_fmadd_ps(q_vec, x3_vec, sums3);
    }
    res0 = _mm512_reduce_add_ps(sums0);
    res1 = _mm512_reduce_add_ps(sums1);
    res2 = _mm512_reduce_add_ps(sums2);
    res3 = _mm512_reduce_add_ps(sums3);
}
} // namespace

namespace rnndescent {

struct SimdDistanceComputerFP32L2 : MyDistanceComputer {
    const float *matrix;
    size_t n, d;
    const float *query;
    std::vector<float> matrixl2norms;
    std::vector<float> queryl2norms;

    explicit SimdDistanceComputerFP32L2(const float *matrix, int n, int d) : matrix(matrix), n(n), d(d) {
        matrixl2norms.resize(n);
        faiss::fvec_norms_L2sqr(matrixl2norms.data(), matrix, d, n);
    }

    int row_count() const override { return static_cast<int>(n); }

    float operator()(int idq, int i) final override {
        const float *__restrict q = query + idq * d;
        const float *__restrict y = matrix + i * d;

        // prefetch_L2(matrixl2norms.data() + i);
        // const float dp0 = faiss::fvec_inner_product(query, y, d);
        // const float dp0 = cblas_sdot(d, query, 1, y, 1);
        const float dp0 = CorrelationSum(query, y, d);
        return queryl2norms[idq] + matrixl2norms[i] - 2 * dp0;
    }

    float symmetric_dis(int i, int j) final override {
        const float *__restrict yi = matrix + i * d;
        const float *__restrict yj = matrix + j * d;

        // prefetch_L2(matrixl2norms.data() + i);
        // prefetch_L2(matrixl2norms.data() + j);
        // const float dp0 = faiss::fvec_inner_product(yi, yj, d);
        // const float dp0 = cblas_sdot(d, yi, 1, yj, 1);
        const float dp0 = CorrelationSum(yi, yj, d);
        return matrixl2norms[i] + matrixl2norms[j] - 2 * dp0;
    }

    void set_query(const float *x, int n) override {
        query = x;
        queryl2norms.resize(n); // fvec_norms_L2sqr
        // faiss::fvec_norms_L2sqr(queryl2norms.data(), query, d, n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            queryl2norms[i]=0;
            for (int k = 0; k < d; k++)
                queryl2norms[i] += x[i * d + k] * x[i * d + k];
        }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) override final {
        // compute first, assign next
        const float *__restrict q = query + idq * d;
        const float *__restrict y0 = matrix + idx0 * d;
        const float *__restrict y1 = matrix + idx1 * d;
        const float *__restrict y2 = matrix + idx2 * d;
        const float *__restrict y3 = matrix + idx3 * d;

        prefetch_L2(matrixl2norms.data() + idx0);
        prefetch_L2(matrixl2norms.data() + idx1);
        prefetch_L2(matrixl2norms.data() + idx2);
        prefetch_L2(matrixl2norms.data() + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;

        CorrelationSum4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);

        dis0 = queryl2norms[idq] + matrixl2norms[idx0] - 2 * dp0;
        dis1 = queryl2norms[idq] + matrixl2norms[idx1] - 2 * dp1;
        dis2 = queryl2norms[idq] + matrixl2norms[idx2] - 2 * dp2;
        dis3 = queryl2norms[idq] + matrixl2norms[idx3] - 2 * dp3;
    }
};
} // namespace rnndescent