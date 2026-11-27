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
            if (count == 1) {
                query_norms[0] = Kernel::query_norm(input, d);
            } else if (count > 1) {
#pragma omp parallel for if (parallel_query_preparation(count))
                for (int i = 0; i < count; ++i)
                    query_norms[i] = Kernel::query_norm(input + i * d, d);
            }
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

    void score_candidates(int idq, SingleNeighbor *candidates, size_t count) override {
        if (count == 0)
            return;

        const auto *q = data(query) + idq * d;
        const auto *base = data(matrix);
        const auto norm = query_norms[idq];
        const size_t end4 = count / 4 * 4;
        for (size_t i = 0; i < end4; i += 4) {
            auto *group = candidates + i;
            const auto dots = Kernel::dot4(q, base + group[0].id * d, base + group[1].id * d,
                                          base + group[2].id * d, base + group[3].id * d, d);
            for (int lane = 0; lane < 4; ++lane)
                group[lane].distance = squared_distance(norm + matrix_norms[group[lane].id], dots[lane]);
        }
        for (size_t i = end4; i < count; ++i) {
            auto &candidate = candidates[i];
            const auto product = Kernel::dot(q, base + candidate.id * d, d);
            candidate.distance = squared_distance(norm + matrix_norms[candidate.id], product);
        }
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
        const auto encode_row = [&](size_t i) {
            const auto *source = input + i * d;
            auto *row = rows.data() + i * d;
            for (size_t k = 0; k < d; ++k)
                row[k] = kernel.encode(source[k]);
            norms[i] = Kernel::dot(row, row, d);
        };
        if (count == 1) {
            encode_row(0);
        } else if (count > 1) {
#pragma omp parallel for
            for (size_t i = 0; i < count; ++i)
                encode_row(i);
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
