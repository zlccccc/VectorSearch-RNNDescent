#pragma once
#include "utils.h"
#include "MyDistanceComputer.h"
#include <faiss/utils/distances.h>
#include <vector>
#include <faiss/utils/prefetch.h>
#include <faiss/impl/NNDescent.h>

#include <arm_neon.h>

namespace {
inline float32x4_t Load(const float32_t *p) {
    // #ifdef __GNUC__
    //   __builtin_prefetch(p + 128);
    // #endif
    // prefetch_L2(p + 256);
    return vld1q_f32(p);
}

inline float32_t simde_dot_f32(float32_t const* a, float32_t const* b, size_t n) {
    float32x4_t ab_vec {0};
// #pragma unroll simd
    for (size_t i = 0; i < n; i += 4) {
        float32x4_t a_vec = Load(a + i);
        float32x4_t b_vec = Load(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    }
    return vaddvq_f32(ab_vec);
}

inline void simde_dot_f32_batch4(const float32_t *q, const float32_t *x0, const float32_t *x1, const float32_t *x2, const float32_t *x3, size_t size,
                                 float &res0, float &res1, float &res2, float &res3) {
    float32x4_t sums0{0};
    float32x4_t sums1{0};
    float32x4_t sums2{0};
    float32x4_t sums3{0};
// #pragma unroll simd
    for (size_t i = 0; i < size; i += 4) {
        float32x4_t q_vec = Load(q + i);
        float32x4_t x0_vec = Load(x0 + i);
        float32x4_t x1_vec = Load(x1 + i);
        float32x4_t x2_vec = Load(x2 + i);
        float32x4_t x3_vec = Load(x3 + i);
        sums0 = vfmaq_f32(sums0, q_vec, x0_vec);
        sums1 = vfmaq_f32(sums1, q_vec, x1_vec);
        sums2 = vfmaq_f32(sums2, q_vec, x2_vec);
        sums3 = vfmaq_f32(sums3, q_vec, x3_vec);
    }
    // return vaddvq_f16(ab_vec);
    res0 = vaddvq_f32(sums0);
    res1 = vaddvq_f32(sums1);
    res2 = vaddvq_f32(sums2);
    res3 = vaddvq_f32(sums3);
}
} // namespace

namespace rnndescent {

struct SimdDistanceComputerFP32L2 : MyDistanceComputer {
    const float32_t *matrix;
    size_t n, d;
    const float32_t* query;
    size_t ndis;
    std::vector<float> matrixpool;
    std::vector<float> matrixl2norms;
    std::vector<float> queryl2norms;

    explicit SimdDistanceComputerFP32L2(const float *matrix, faiss::idx_t n, int d) : n(n), d(d), ndis(0) {
        this->matrix = matrix;
        matrixl2norms.resize(n);
        faiss::fvec_norms_L2sqr(matrixl2norms.data(), matrix, d, n);
    }

    int row_count() const override { return static_cast<int>(n); }

    int dimension() const override { return static_cast<int>(d); }

    float operator()(int idq, int i) final override {
        const float32_t *__restrict y0 = matrix + i * d;
        const float32_t *__restrict q = query + idq * d;
        float dp0 = simde_dot_f32(y0, q, d);
        return queryl2norms[idq] + matrixl2norms[i] - 2 * dp0;
    }

    float symmetric_dis(int i, int j) final override {
        float32_t result;
        const float32_t *__restrict y0 = matrix + i * d;
        const float32_t *__restrict y1 = matrix + j * d;
        float dp0 = simde_dot_f32(y0, y1, d);

        return matrixl2norms[i] + matrixl2norms[j] - 2 * dp0;
    }

    void set_query(const float *x, int n) override {
        this->query = x;
        queryl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            queryl2norms[i] = 0;
            for (int k = 0; k < d; k++)
                queryl2norms[i] += x[i * d + k] * x[i * d + k];
        }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) {
        const float32_t *__restrict q = query + idq * d;
        const float32_t *__restrict y0 = matrix + idx0 * d;
        const float32_t *__restrict y1 = matrix + idx1 * d;
        const float32_t *__restrict y2 = matrix + idx2 * d;
        const float32_t *__restrict y3 = matrix + idx3 * d;

        // prefetch_L2(matrixl2norms.data() + idx0);
        // prefetch_L2(matrixl2norms.data() + idx1);
        // prefetch_L2(matrixl2norms.data() + idx2);
        // prefetch_L2(matrixl2norms.data() + idx3);
        float dp0, dp1, dp2, dp3;
        simde_dot_f32_batch4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        // dp0 = simde_dot_f32(matrix + idx0 * d, q, d);
        // dp1 = simde_dot_f32(matrix + idx1 * d, q, d);
        // dp2 = simde_dot_f32(matrix + idx2 * d, q, d);
        // dp3 = simde_dot_f32(matrix + idx3 * d, q, d);
        dis0 = queryl2norms[idq] + matrixl2norms[idx0] - 2 * dp0;
        dis1 = queryl2norms[idq] + matrixl2norms[idx1] - 2 * dp1;
        dis2 = queryl2norms[idq] + matrixl2norms[idx2] - 2 * dp2;
        dis3 = queryl2norms[idq] + matrixl2norms[idx3] - 2 * dp3;
    }
    ~SimdDistanceComputerFP32L2() override {}
};
} // namespace rnndescent