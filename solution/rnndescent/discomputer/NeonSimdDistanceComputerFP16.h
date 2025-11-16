#pragma once
#include "utils.h"
#include "MyDistanceComputer.h"
#include <faiss/utils/distances.h>
#include <vector>
#include <faiss/utils/prefetch.h>
#include <faiss/impl/NNDescent.h>

#include <arm_neon.h>

namespace {
inline float16x8_t Load(const float16_t *p) {
    // #ifdef __GNUC__
    //   __builtin_prefetch(p + 128);
    // #endif
    prefetch_L2(p + 256);
    return vld1q_f16(p);
}

inline float simde_dot_f16(float16_t const *a, float16_t const *b, size_t n) {
    float16x8_t ab_vec{0};
// #pragma unroll simd
    for (size_t i = 0; i < n; i += 8) {
        float16x8_t a_vec = Load(a + i);
        float16x8_t b_vec = Load(b + i);
        // ab_vec = ab_vec + a_vec * b_vec;
        // ab_vec = vaddq_f16(ab_vec, vmulq_f16(a_vec, b_vec));
        ab_vec = vfmaq_f16(ab_vec, a_vec, b_vec);
    }
    // return vaddvq_f16(ab_vec);
    float x1 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(ab_vec)));
    float x2 = vaddvq_f32(vcvt_f32_f16(vget_high_f16(ab_vec)));
    return x1 + x2;
}

inline void simde_dot_f16_batch4(const float16_t *q, const float16_t *x0, const float16_t *x1, const float16_t *x2, const float16_t *x3, size_t size,
                                 float &res0, float &res1, float &res2, float &res3) {
    float16x8_t sums0{0};
    float16x8_t sums1{0};
    float16x8_t sums2{0};
    float16x8_t sums3{0};
// #pragma unroll simd
    for (size_t i = 0; i < size; i += 8) {
        float16x8_t q_vec = Load(q + i);
        float16x8_t x0_vec = Load(x0 + i);
        float16x8_t x1_vec = Load(x1 + i);
        float16x8_t x2_vec = Load(x2 + i);
        float16x8_t x3_vec = Load(x3 + i);
        // ab_vec = ab_vec + a_vec * b_vec;
        // ab_vec = vaddq_f16(ab_vec, vmulq_f16(a_vec, b_vec));
        sums0 = vfmaq_f16(sums0, q_vec, x0_vec);
        sums1 = vfmaq_f16(sums1, q_vec, x1_vec);
        sums2 = vfmaq_f16(sums2, q_vec, x2_vec);
        sums3 = vfmaq_f16(sums3, q_vec, x3_vec);
    }
    // return vaddvq_f16(ab_vec);
    res0 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(sums0))) + vaddvq_f32(vcvt_f32_f16(vget_high_f16(sums0)));
    res1 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(sums1))) + vaddvq_f32(vcvt_f32_f16(vget_high_f16(sums1)));
    res2 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(sums2))) + vaddvq_f32(vcvt_f32_f16(vget_high_f16(sums2)));
    res3 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(sums3))) + vaddvq_f32(vcvt_f32_f16(vget_high_f16(sums3)));
}

inline void simde_cvt_f32_to_f16(float32_t const *a, float16_t *b, size_t n) {
#pragma unroll simd
    for (size_t i = 0; i < n; i += 4) {
        vst1_f16(b + i, vcvt_f16_f32(vld1q_f32(a + i)));
    }
}
} // namespace

namespace rnndescent {

struct SimdDistanceComputerFP16L2 : MyDistanceComputer {
    std::vector<float16_t> matrix;
    size_t n, d;
    std::vector<float16_t> query;
    size_t ndis;
    std::vector<float> matrixpool;
    std::vector<float> matrixl2norms;
    std::vector<float> queryl2norms;

    explicit SimdDistanceComputerFP16L2(const float *matrix, faiss::idx_t n, int d) : n(n), d(d), ndis(0) {
        (this->matrix).resize(n * d);
        simde_cvt_f32_to_f16(matrix, (float16_t *)(this->matrix).data(), n * d);
        matrixl2norms.resize(n);
        faiss::fvec_norms_L2sqr(matrixl2norms.data(), matrix, d, n);
    }

    float operator()(int idq, int i) final override {
        const float16_t *__restrict y0 = matrix.data() + i * d;
        const float16_t *__restrict q = query.data() + idq * d;
        float dp0 = simde_dot_f16(y0, q, d);
        return queryl2norms[idq] + matrixl2norms[i] - 2 * dp0;
    }

    float symmetric_dis(int i, int j) final override {
        float32_t result;
        const float16_t *__restrict y0 = matrix.data() + i * d;
        const float16_t *__restrict y1 = matrix.data() + j * d;
        float dp0 = simde_dot_f16(y0, y1, d);

        return matrixl2norms[i] + matrixl2norms[j] - 2 * dp0;
    }

    void set_query(const float *x, int n) override {
        query.resize(n * d);
        queryl2norms.resize(n);
        simde_cvt_f32_to_f16(x, (float16_t *)(this->query).data(), n * d);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            queryl2norms[i] = 0;
            for (int k = 0; k < d; k++)
                queryl2norms[i] += x[i * d + k] * x[i * d + k];
        }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) {
        const float16_t *__restrict q = query.data() + idq * d;
        const float16_t *__restrict y0 = matrix.data() + idx0 * d;
        const float16_t *__restrict y1 = matrix.data() + idx1 * d;
        const float16_t *__restrict y2 = matrix.data() + idx2 * d;
        const float16_t *__restrict y3 = matrix.data() + idx3 * d;

        // prefetch_L2(matrixl2norms.data() + idx0);
        // prefetch_L2(matrixl2norms.data() + idx1);
        // prefetch_L2(matrixl2norms.data() + idx2);
        // prefetch_L2(matrixl2norms.data() + idx3);
        float dp0, dp1, dp2, dp3;
        simde_dot_f16_batch4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        // dp0 = simde_dot_f16(matrix.data() + idx0 * d, q, d);
        // dp1 = simde_dot_f16(matrix.data() + idx1 * d, q, d);
        // dp2 = simde_dot_f16(matrix.data() + idx2 * d, q, d);
        // dp3 = simde_dot_f16(matrix.data() + idx3 * d, q, d);
        dis0 = queryl2norms[idq] + matrixl2norms[idx0] - 2 * dp0;
        dis1 = queryl2norms[idq] + matrixl2norms[idx1] - 2 * dp1;
        dis2 = queryl2norms[idq] + matrixl2norms[idx2] - 2 * dp2;
        dis3 = queryl2norms[idq] + matrixl2norms[idx3] - 2 * dp3;
    }

    // void distance_preloaded_batch4(int idq, const float16_t *x, float &dis0, float &dis1, float &dis2, float &dis3, const float *l2norms) { assert(0); }
    void distance_preloaded_batch4(int idq, const float16_t *x, float &dis0, float &dis1, float &dis2, float &dis3, const float *l2norms) override final {
        const float16_t *__restrict q = query.data() + idq * d;
        const float16_t *__restrict x0 = x + 0 * d;
        const float16_t *__restrict x1 = x + 1 * d;
        const float16_t *__restrict x2 = x + 2 * d;
        const float16_t *__restrict x3 = x + 3 * d;
        float32_t dp0, dp1, dp2, dp3;
        simde_dot_f16_batch4(q, x0, x1, x2, x3, d, dp0, dp1, dp2, dp3);
        // dp0 = simde_dot_f16(x0, q, d);
        // dp1 = simde_dot_f16(x1, q, d);
        // dp2 = simde_dot_f16(x2, q, d);
        // dp3 = simde_dot_f16(x3, q, d);
        dis0 = queryl2norms[idq] + l2norms[0] - 2 * dp0;
        dis1 = queryl2norms[idq] + l2norms[1] - 2 * dp1;
        dis2 = queryl2norms[idq] + l2norms[2] - 2 * dp2;
        dis3 = queryl2norms[idq] + l2norms[3] - 2 * dp3;
    }

    void copy_index(int idx0, float16_t *y, float &l2norm) override final {
        memcpy(y, matrix.data() + idx0 * d, d * sizeof(float16_t));
        l2norm = matrixl2norms[idx0];
    }
    ~SimdDistanceComputerFP16L2() override {}
};
} // namespace rnndescent