#pragma once
// #include "faiss_platform_macros.h"
// #include <faiss/impl/DistanceComputer.h>
// #include "simd/dot.h"
#include <arm_neon.h>

#include "utils.h"
#include <algorithm>
#include <cmath>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/prefetch.h>
#include <string.h>
#include <vector>

#include "MyDistanceComputer.h"

namespace {

inline int16x8_t Int16Load(const int16_t *p) {
    // #ifdef __GNUC__
    //   __builtin_prefetch(p + 128);
    // #endif
    // prefetch_L2(p + 256);
    return vld1q_s16(p);
}

inline int64x2_t Int16Correlation(const int16x8_t &a, const int16x8_t &b) {
    int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    return vaddq_s64(vpaddlq_s32(lo), vpaddlq_s32(hi));
}
// FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
inline int64_t Int16CorrelationSum(const int16_t *a, const int16_t *b, size_t size) {
    int64x2_t sums = vdupq_n_s64(0);
    // FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < size; i += 8) {
        sums = vaddq_s64(sums, Int16Correlation(Int16Load(a + i), Int16Load(b + i)));
    }
    return vaddvq_s64(sums);
}
// FAISS_PRAGMA_IMPRECISE_FUNCTION_END

inline void Int16CorrelationSum4(const int16_t *q, const int16_t *x0, const int16_t *x1, const int16_t *x2, const int16_t *x3, size_t size, int64_t &res0,
                                 int64_t &res1, int64_t &res2, int64_t &res3) {
    int64x2_t sums0 = vdupq_n_s64(0);
    int64x2_t sums1 = vdupq_n_s64(0);
    int64x2_t sums2 = vdupq_n_s64(0);
    int64x2_t sums3 = vdupq_n_s64(0);
    // FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < size; i += 8) {
        int16x8_t loadq = Int16Load(q + i);
        sums0 = vaddq_s64(sums0, Int16Correlation(Int16Load(x0 + i), loadq));
        sums1 = vaddq_s64(sums1, Int16Correlation(Int16Load(x1 + i), loadq));
        sums2 = vaddq_s64(sums2, Int16Correlation(Int16Load(x2 + i), loadq));
        sums3 = vaddq_s64(sums3, Int16Correlation(Int16Load(x3 + i), loadq));
    }
    res0 = vaddvq_s64(sums0);
    res1 = vaddvq_s64(sums1);
    res2 = vaddvq_s64(sums2);
    res3 = vaddvq_s64(sums3);
}
} // namespace

namespace rnndescent {

struct SimdDistanceComputerInt16L2 : MyDistanceComputer {
    size_t n, d;
    std::vector<int16_t> matrix;
    std::vector<int16_t> query;
    const int maxscale = 32767;
    // const int maxscale = 2048;
    // const int maxscale = 127;
    // const int maxscale = 15;
    double scale = 0, scale2 = 0, mean = 0;
    std::vector<double> matrixl2norms;
    // std::vector<double> queryl2norms;

    inline void scaled_dot(const int16_t *a, const int16_t *b, float &res) {
        res = Int16CorrelationSum(a, b, d) * scale2;
        // res = Int16CorrelationSum(a, b, d);
    }

    inline void scaled_dot_batch4(const int16_t *q, const int16_t *x0, const int16_t *x1, const int16_t *x2, const int16_t *x3, size_t size, double &res0,
                                  double &res1, double &res2, double &res3) {
        int64_t dp0, dp1, dp2, dp3;
        Int16CorrelationSum4(q, x0, x1, x2, x3, size, dp0, dp1, dp2, dp3);
        res0 = dp0 * scale2;
        res1 = dp1 * scale2;
        res2 = dp2 * scale2;
        res3 = dp3 * scale2;
    }

    inline void cvt_fp32_to_int16(const float *a, int16_t *b) {
#pragma simd
        for (int k = 0; k < d; k++) {
            b[k] = std::max(-maxscale, std::min(maxscale, (int)std::round((a[k] - mean) / scale)));
        }
    }

    explicit SimdDistanceComputerInt16L2(const float *matrix, int n, int d) : n(n), d(d) {
        (this->matrix).resize(n * d);
        double maxvalue = *std::max_element(matrix, matrix + n * d);
        double minvalue = *std::min_element(matrix, matrix + n * d);
        scale = (maxvalue - minvalue) / 2.0;
        mean = (maxvalue + minvalue) / 2.0;
        scale /= maxscale;
        printf("Int16 QT scale = %f; mean = %f\n", scale, mean);
        scale2 = scale * scale;
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            cvt_fp32_to_int16(matrix + i * d, this->matrix.data() + i * d);
        }

        matrixl2norms.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                matrixl2norms[i] += (matrix[i * d + k] - mean) * (matrix[i * d + k] - mean);
            }
        }
    }

    int row_count() const override { return static_cast<int>(n); }

    int dimension() const override { return static_cast<int>(d); }

    float operator()(int idq, int i) final override {
        float32_t result;
        const int16_t *__restrict q = query.data() + idq * d;
        const int16_t *__restrict y = matrix.data() + i * d;
        scaled_dot(y, q, result);
        return matrixl2norms[i] - 2 * result;
    }

    float symmetric_dis(int i, int j) final override {
        float32_t dp0;
        const int16_t *__restrict x = matrix.data() + i * d;
        const int16_t *__restrict y = matrix.data() + j * d;
        scaled_dot(x, y, dp0);
        return matrixl2norms[i] + matrixl2norms[j] - 2 * dp0;
    }

    void set_query(const float *x, int n) override {
        query.resize(d * n);
//         queryl2norms.resize(n);
// #pragma omp parallel for
//         for (int i = 0; i < n; i++) {
//             queryl2norms[i] = 0;
//             for (int k = 0; k < d; k++)
//                 queryl2norms[i] += (x[i * d + k] - mean) * (x[i * d + k] - mean);
//             cvt_fp32_to_int16(x + i * d, query.data() + i * d);
//         }
    }

    // compute four distances
    void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) override final {
        const int16_t *__restrict q = query.data() + idq * d;
        const int16_t *__restrict x0 = matrix.data() + idx0 * d;
        const int16_t *__restrict x1 = matrix.data() + idx1 * d;
        const int16_t *__restrict x2 = matrix.data() + idx2 * d;
        const int16_t *__restrict x3 = matrix.data() + idx3 * d;

        // TODO: prefetch
        // prefetch_L2(matrixl2norms.data() + idx0);
        // prefetch_L2(matrixl2norms.data() + idx1);
        // prefetch_L2(matrixl2norms.data() + idx2);
        // prefetch_L2(matrixl2norms.data() + idx3);
        double dp0, dp1, dp2, dp3;
        scaled_dot_batch4(q, x0, x1, x2, x3, d, dp0, dp1, dp2, dp3);
        // dis0 = queryl2norms[idq] + matrixl2norms[idx0] - 2 * dp0;
        // dis1 = queryl2norms[idq] + matrixl2norms[idx1] - 2 * dp1;
        // dis2 = queryl2norms[idq] + matrixl2norms[idx2] - 2 * dp2;
        // dis3 = queryl2norms[idq] + matrixl2norms[idx3] - 2 * dp3;
        dis0 = matrixl2norms[idx0] - 2 * dp0;
        dis1 = matrixl2norms[idx1] - 2 * dp1;
        dis2 = matrixl2norms[idx2] - 2 * dp2;
        dis3 = matrixl2norms[idx3] - 2 * dp3;
    }

    ~SimdDistanceComputerInt16L2() override {}
};

} // namespace rnndescent