#pragma once

#include "FlatDistanceComputer.h"

#include <array>
#include <immintrin.h>

namespace rnndescent {

struct Avx512FP32Kernel : FP32DistanceKernel {
    static constexpr size_t lanes = sizeof(__m512) / sizeof(float);

    static float dot(const float *a, const float *b, size_t n) {
        __m512 sums = _mm512_setzero_ps();
        const size_t tail = n % lanes;
        const size_t full_size = n - tail;
        for (size_t i = 0; i < full_size; i += lanes) {
            sums = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sums);
        }
        if (tail != 0) {
            const auto mask = static_cast<__mmask16>((1u << tail) - 1u);
            const auto a_vec = _mm512_maskz_loadu_ps(mask, a + full_size);
            const auto b_vec = _mm512_maskz_loadu_ps(mask, b + full_size);
            sums = _mm512_fmadd_ps(a_vec, b_vec, sums);
        }
        return _mm512_reduce_add_ps(sums);
    }

    static std::array<float, 4> dot4(const float *q, const float *x0, const float *x1, const float *x2, const float *x3, size_t size) {
        __m512 sums0 = _mm512_setzero_ps();
        __m512 sums1 = _mm512_setzero_ps();
        __m512 sums2 = _mm512_setzero_ps();
        __m512 sums3 = _mm512_setzero_ps();
        const size_t tail = size % lanes;
        const size_t full_size = size - tail;
        for (size_t i = 0; i < full_size; i += lanes) {
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
        if (tail != 0) {
            const auto mask = static_cast<__mmask16>((1u << tail) - 1u);
            const auto q_vec = _mm512_maskz_loadu_ps(mask, q + full_size);
            sums0 = _mm512_fmadd_ps(q_vec, _mm512_maskz_loadu_ps(mask, x0 + full_size), sums0);
            sums1 = _mm512_fmadd_ps(q_vec, _mm512_maskz_loadu_ps(mask, x1 + full_size), sums1);
            sums2 = _mm512_fmadd_ps(q_vec, _mm512_maskz_loadu_ps(mask, x2 + full_size), sums2);
            sums3 = _mm512_fmadd_ps(q_vec, _mm512_maskz_loadu_ps(mask, x3 + full_size), sums3);
        }
        return {_mm512_reduce_add_ps(sums0), _mm512_reduce_add_ps(sums1), _mm512_reduce_add_ps(sums2), _mm512_reduce_add_ps(sums3)};
    }

    static float query_norm(const float *row, size_t dim) { return dot(row, row, dim); }
};

using SimdDistanceComputerFP32L2 = FlatDistanceComputer<Avx512FP32Kernel>;

} // namespace rnndescent
