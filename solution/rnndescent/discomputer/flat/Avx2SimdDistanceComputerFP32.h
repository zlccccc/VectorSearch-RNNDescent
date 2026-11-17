#pragma once

#include "FlatDistanceComputer.h"

#include <array>
#include <immintrin.h>

namespace rnndescent {

struct Avx2FP32Kernel : FP32DistanceKernel {
    static float reduce(__m256 sums) {
        __m128 ab_sum = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
        ab_sum = _mm_hadd_ps(ab_sum, ab_sum);
        ab_sum = _mm_hadd_ps(ab_sum, ab_sum);
        return _mm_cvtss_f32(ab_sum);
    }

    static constexpr size_t lanes = sizeof(__m256) / sizeof(float);

    static __m256i tail_mask(size_t count) {
        return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    }

    static float dot(const float *a, const float *b, size_t n) {
        __m256 sums = _mm256_setzero_ps();
        const size_t tail = n % lanes;
        const size_t full_size = n - tail;
        for (size_t i = 0; i < full_size; i += lanes) {
            sums = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sums);
        }
        if (tail != 0) {
            const auto mask = tail_mask(tail);
            sums = _mm256_fmadd_ps(_mm256_maskload_ps(a + full_size, mask), _mm256_maskload_ps(b + full_size, mask), sums);
        }
        return reduce(sums);
    }

    static std::array<float, 4> dot4(const float *q, const float *x0, const float *x1, const float *x2, const float *x3, size_t size) {
        __m256 sums0 = _mm256_setzero_ps();
        __m256 sums1 = _mm256_setzero_ps();
        __m256 sums2 = _mm256_setzero_ps();
        __m256 sums3 = _mm256_setzero_ps();
        const size_t tail = size % lanes;
        const size_t full_size = size - tail;
        for (size_t i = 0; i < full_size; i += lanes) {
            __m256 q_vec = _mm256_loadu_ps(q + i);
            __m256 x0_vec = _mm256_loadu_ps(x0 + i);
            __m256 x1_vec = _mm256_loadu_ps(x1 + i);
            __m256 x2_vec = _mm256_loadu_ps(x2 + i);
            __m256 x3_vec = _mm256_loadu_ps(x3 + i);
            sums0 = _mm256_fmadd_ps(q_vec, x0_vec, sums0);
            sums1 = _mm256_fmadd_ps(q_vec, x1_vec, sums1);
            sums2 = _mm256_fmadd_ps(q_vec, x2_vec, sums2);
            sums3 = _mm256_fmadd_ps(q_vec, x3_vec, sums3);
        }
        if (tail != 0) {
            const auto mask = tail_mask(tail);
            const auto q_vec = _mm256_maskload_ps(q + full_size, mask);
            sums0 = _mm256_fmadd_ps(q_vec, _mm256_maskload_ps(x0 + full_size, mask), sums0);
            sums1 = _mm256_fmadd_ps(q_vec, _mm256_maskload_ps(x1 + full_size, mask), sums1);
            sums2 = _mm256_fmadd_ps(q_vec, _mm256_maskload_ps(x2 + full_size, mask), sums2);
            sums3 = _mm256_fmadd_ps(q_vec, _mm256_maskload_ps(x3 + full_size, mask), sums3);
        }
        return {reduce(sums0), reduce(sums1), reduce(sums2), reduce(sums3)};
    }
};

using SimdDistanceComputerFP32L2 = FlatDistanceComputer<Avx2FP32Kernel>;

} // namespace rnndescent
