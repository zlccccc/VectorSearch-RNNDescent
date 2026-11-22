#pragma once

#include "../../common/HugePages.h"
#include "../MyDistanceComputer.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace rnndescent {

struct FP32DistanceKernel {
    using Code = float;
    using Dot = float;

    static float query_norm(const float *row, size_t dim) {
        float norm = 0.0f;
        for (size_t k = 0; k < dim; ++k)
            norm += row[k] * row[k];
        return norm;
    }
};

// Distance computers that read rows directly from their original or encoded storage.
template <class Kernel> class FlatDistanceComputer final : public MyDistanceComputer {
    using Code = typename Kernel::Code;
    using Dot = typename Kernel::Dot;
    static constexpr bool uses_fp32_rows = std::is_same_v<Code, float>;
    using Rows = std::conditional_t<uses_fp32_rows, const float *, std::vector<Code>>;

  public:
    FlatDistanceComputer(const float *input, int64_t count, int dim) : n(count), d(dim) {
        if constexpr (uses_fp32_rows) {
            matrix = input;
            matrix_norms.resize(n);
#pragma omp parallel for if (n > 10000)
            for (size_t i = 0; i < n; ++i) {
                const auto *row = input + i * d;
                matrix_norms[i] = Kernel::dot(row, row, d);
            }
        } else {
            if constexpr (std::is_integral_v<Code>)
                kernel.train(input, n * d);
            encode_rows(input, n, matrix, matrix_norms);
            detail::try_huge_pages(matrix.data(), matrix.size() * sizeof(Code));
        }
        detail::try_huge_pages(matrix_norms.data(), matrix_norms.size() * sizeof(Dot));
    }

    int row_count() const override { return static_cast<int>(n); }
    int dimension() const override { return static_cast<int>(d); }

    void set_query(const float *input, int count) override {
        if constexpr (uses_fp32_rows) {
            query = input;
            query_norms.resize(count);
#pragma omp parallel for
            for (int i = 0; i < count; ++i)
                query_norms[i] = Kernel::query_norm(input + i * d, d);
        } else {
            encode_rows(input, count, query, query_norms);
        }
    }

    float operator()(int idq, int i) override {
        const auto product = Kernel::dot(data(query) + idq * d, data(matrix) + i * d, d);
        return squared_distance(query_norms[idq] + matrix_norms[i], product);
    }

    float symmetric_dis(int i, int j) override {
        const auto *base = data(matrix);
        return squared_distance(matrix_norms[i] + matrix_norms[j], Kernel::dot(base + i * d, base + j * d, d));
    }

    void distances_batch_4(int idq, int i0, int i1, int i2, int i3, float &dis0, float &dis1, float &dis2, float &dis3) override {
        const auto *base = data(matrix);
        const auto dots = Kernel::dot4(data(query) + idq * d, base + i0 * d, base + i1 * d, base + i2 * d, base + i3 * d, d);
        const auto norm = query_norms[idq];
        dis0 = squared_distance(norm + matrix_norms[i0], dots[0]);
        dis1 = squared_distance(norm + matrix_norms[i1], dots[1]);
        dis2 = squared_distance(norm + matrix_norms[i2], dots[2]);
        dis3 = squared_distance(norm + matrix_norms[i3], dots[3]);
    }

  private:
    const size_t n, d;
    Kernel kernel;
    Rows matrix{}, query{};
    std::vector<Dot> matrix_norms, query_norms;

    // FP32 reads the caller's rows directly. The low-precision variants own their encoded rows.
    static const Code *data(const Rows &rows) {
        if constexpr (uses_fp32_rows)
            return rows;
        else
            return rows.data();
    }

    void encode_rows(const float *input, size_t count, std::vector<Code> &rows, std::vector<Dot> &norms) {
        rows.resize(count * d);
        norms.resize(count);
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            const auto *source = input + i * d;
            auto *row = rows.data() + i * d;
            for (size_t k = 0; k < d; ++k)
                row[k] = kernel.encode(source[k]);
            norms[i] = Kernel::dot(row, row, d);
        }
    }

    float squared_distance(Dot norm_sum, Dot product) const {
        const auto squared = norm_sum - 2 * product;
        if constexpr (std::is_integral_v<Code>)
            return static_cast<float>(squared * kernel.scale_squared);
        else
            return squared;
    }
};

} // namespace rnndescent
