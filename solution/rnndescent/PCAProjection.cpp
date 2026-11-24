#include "PCAProjection.h"

#include <algorithm>
#include <array>
#include <cblas.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>

extern "C" void dsyev_(const char *job, const char *triangle, const blasint *n, double *matrix,
                       const blasint *stride, double *eigenvalues, double *work,
                       const blasint *work_size, blasint *info);

namespace rnndescent {
namespace {

int hadamard_block_size(int remaining) {
    int size = 1;
    while (size <= remaining / 2)
        size *= 2;
    return size;
}

std::uint32_t spread_coordinate_bits(std::uint32_t bits) {
    // Insert three zero bits between each pair of bits for a four-axis Morton key.
    bits = (bits | bits << 12) & 0x000f000f;
    bits = (bits | bits << 6) & 0x03030303;
    return (bits | bits << 3) & 0x11111111;
}

std::vector<float> leading_eigenvectors(const std::vector<float> &symmetric, blasint dimension, int count) {
    std::vector<double> vectors(symmetric.begin(), symmetric.end());
    std::vector<double> eigenvalues(dimension);
    constexpr char job = 'V', triangle = 'U';
    blasint work_size = -1, info = 0;
    double query = 0;
    dsyev_(&job, &triangle, &dimension, vectors.data(), &dimension, eigenvalues.data(), &query, &work_size, &info);
    RNNDESCENT_ASSERT_MSG(info == 0, "PCA eigensolver workspace query failed");
    work_size = static_cast<blasint>(query);
    std::vector<double> work(work_size);
    dsyev_(&job, &triangle, &dimension, vectors.data(), &dimension, eigenvalues.data(), work.data(), &work_size, &info);
    RNNDESCENT_ASSERT_MSG(info == 0, "PCA eigensolver failed");

    // LAPACK stores eigenvectors by column in ascending eigenvalue order.
    std::vector<float> leading(static_cast<size_t>(count) * dimension);
    for (int i = 0; i < count; ++i)
        std::copy_n(vectors.data() + static_cast<size_t>(dimension - 1 - i) * dimension, dimension,
                    leading.data() + static_cast<size_t>(i) * dimension);
    return leading;
}

void rotate_pca_basis(std::vector<float> &components, std::vector<float> &bias, int input_dimension) {
    // A block-diagonal Hadamard keeps the rotation orthogonal at any output
    // dimension. Sign flips and butterflies act on the projection rows and bias.
    const int dimension = static_cast<int>(bias.size());
    std::mt19937_64 random(64);
    std::uint64_t signs = 0x6d2ed559ca31b498ull;
    for (int start = 0; start < dimension;) {
        const int size = hadamard_block_size(dimension - start);

        const float normalization = 1.0f / std::sqrt(static_cast<float>(size));
        const int end = start + size;
        for (int i = start; i < end; ++i) {
            const int bit = i % 64;
            if (i != 0 && bit == 0)
                signs = random();
            const float scale = ((signs >> bit) & 1) ? -normalization : normalization;
            auto *row = components.data() + static_cast<size_t>(i) * input_dimension;
            for (int j = 0; j < input_dimension; ++j)
                row[j] *= scale;
            bias[i] *= scale;
        }
        for (int half = 1; half < size; half *= 2) {
            for (int tile = start; tile < end; tile += 2 * half) {
                for (int i = tile; i < tile + half; ++i) {
                    auto *left = components.data() + static_cast<size_t>(i) * input_dimension;
                    auto *right = left + static_cast<size_t>(half) * input_dimension;
                    for (int j = 0; j < input_dimension; ++j) {
                        const float a = left[j], b = right[j];
                        left[j] = a + b;
                        right[j] = a - b;
                    }
                    const float a = bias[i], b = bias[i + half];
                    bias[i] = a + b;
                    bias[i + half] = a - b;
                }
            }
        }
        start = end;
    }
}

} // namespace

PCAProjection::PCAProjection(const FloatMatrixView &base, int output_dimension)
    : input_dimension_(base.dimension()), output_dimension_(output_dimension) {
    base.validate("PCA training data");
    const int d = input_dimension_, rows = base.row_count();
    RNNDESCENT_ASSERT_MSG(output_dimension > 0 && output_dimension <= std::min(d, rows),
                          "PCA output dimension exceeds the input dimension or row count");

    // Preserve the previous training sample size and the deterministic sampling order.
    constexpr int rows_per_dimension = 1000;
    const int n = static_cast<int>(std::min<long long>(rows, static_cast<long long>(rows_per_dimension) * d));
    const float *training = base.data_ptr();
    std::vector<int> sampled_rows;
    if (n < rows) {
        sampled_rows.resize(rows);
        std::iota(sampled_rows.begin(), sampled_rows.end(), 0);
        std::mt19937 random(1234);
        for (int i = 0; i < n; ++i) {
            const int other = i + static_cast<int>(random() % (rows - i));
            std::swap(sampled_rows[i], sampled_rows[other]);
        }
        sampled_rows.resize(n);
    }

    std::vector<float> mean(d, 0.0f);
    for (int i = 0; i < n; ++i) {
        const auto *row = base.row_ptr(sampled_rows.empty() ? i : sampled_rows[i]);
        for (int j = 0; j < d; ++j)
            mean[j] += row[j];
    }
    for (float &value : mean)
        value /= n;

    if (n >= d) {
        // Center the covariance by subtracting n * mean * mean^T from X * X^T.
        std::vector<float> covariance(static_cast<size_t>(d) * d);
        for (int col = 0; col < d; ++col)
            for (int row = 0; row <= col; ++row)
                covariance[static_cast<size_t>(col) * d + row] = -n * mean[col] * mean[row];
        if (sampled_rows.empty()) {
            cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, d, n, 1.0f, training, d, 1.0f, covariance.data(), d);
        } else {
            // Gather only one BLAS batch, instead of retaining a full FP32 training copy.
            constexpr int batch_rows = 16384;
            std::vector<float> sample(static_cast<size_t>(std::min(n, batch_rows)) * d);
            for (int offset = 0; offset < n; offset += batch_rows) {
                const int count = std::min(batch_rows, n - offset);
#pragma omp parallel for
                for (int i = 0; i < count; ++i)
                    std::copy_n(base.row_ptr(sampled_rows[offset + i]), d, sample.data() + static_cast<size_t>(i) * d);
                cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, d, count, 1.0f, sample.data(), d,
                            1.0f, covariance.data(), d);
            }
        }
        components_ = leading_eigenvectors(covariance, d, output_dimension);
    } else {
        // With fewer rows than dimensions, solve the smaller Gram matrix.
        std::vector<float> centered(static_cast<size_t>(n) * d);
        for (int i = 0; i < n; ++i) {
            const auto offset = static_cast<size_t>(i) * d;
            for (int j = 0; j < d; ++j)
                centered[offset + j] = training[offset + j] - mean[j];
        }
        std::vector<float> gram(static_cast<size_t>(n) * n);
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, d, 1.0f, centered.data(), d, 0.0f, gram.data(), n);

        const auto vectors = leading_eigenvectors(gram, n, output_dimension);
        components_.resize(static_cast<size_t>(output_dimension) * d);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, output_dimension, n, 1.0f,
                    centered.data(), d, vectors.data(), n, 0.0f, components_.data(), d);
        for (int i = 0; i < output_dimension; ++i) {
            auto *component = components_.data() + static_cast<size_t>(i) * d;
            const float norm = cblas_snrm2(d, component, 1);
            if (norm > 0.0f)
                cblas_sscal(d, 1.0f / norm, component, 1);
        }
    }

    bias_.resize(output_dimension);
    for (int i = 0; i < output_dimension; ++i) {
        const auto *component = components_.data() + static_cast<size_t>(i) * d;
        for (int j = 0; j < d; ++j)
            bias_[i] -= component[j] * mean[j];
    }
    rotate_pca_basis(components_, bias_, d);
}

void PCAProjection::apply(const FloatMatrixView &input, const MutableFloatMatrixView &output) const {
    input.validate("PCA input");
    output.validate("PCA output");
    RNNDESCENT_ASSERT_MSG(input.row_count() == output.row_count(), "PCA input/output row count mismatch");
    RNNDESCENT_ASSERT_MSG(input.dimension() == input_dimension_, "PCA input dimension mismatch");
    RNNDESCENT_ASSERT_MSG(output.dimension() == output_dimension_, "PCA output dimension mismatch");

#pragma omp parallel for
    for (int i = 0; i < input.row_count(); ++i)
        std::copy(bias_.begin(), bias_.end(), output.row_ptr(i));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, output_dimension_, input.row_count(), input_dimension_,
                1.0f, components_.data(), input_dimension_, input.data_ptr(), input_dimension_,
                1.0f, output.data_ptr(), output_dimension_);
}

FloatMatrixView PCAProjection::project(const FloatMatrixView &input, std::vector<float> &buffer) const {
    input.validate("PCA input");
    RNNDESCENT_ASSERT_MSG(input.dimension() == input_dimension_, "PCA input dimension mismatch");
    buffer.resize(static_cast<size_t>(input.row_count()) * output_dimension_);
    apply(input, MutableFloatMatrixView::from_vector(buffer, output_dimension_));
    return FloatMatrixView::from_vector(buffer, output_dimension_);
}

FloatMatrixView PCAProjection::project_queries(const FloatMatrixView &queries) {
    return project(queries, query_buffer_);
}

std::vector<int> PCAProjection::query_order(const FloatMatrixView &projected) const {
    constexpr int axes = 4;
    const int count = projected.row_count();
    if (count <= 1 || output_dimension_ < axes)
        return {};
    projected.validate("projected queries");
    RNNDESCENT_ASSERT_MSG(projected.dimension() == output_dimension_, "projected query dimension mismatch");

    const int block_size = hadamard_block_size(output_dimension_);
    std::vector<std::array<float, axes>> coordinates(count);
    for (int i = 0; i < count; ++i) {
        const float *row = projected.row_ptr(i);
        float a = 0, b = 0, c = 0, d = 0;
        for (int j = 0; j < block_size; j += axes) {
            a += row[j];
            b += row[j + 1];
            c += row[j + 2];
            d += row[j + 3];
        }
        // Recover the leading principal coordinates up to a fixed sign and scale.
        // The first Hadamard block contains all four when its size is at least four.
        coordinates[i] = {a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d};
    }

    auto minima = coordinates.front();
    auto maxima = minima;
    for (const auto &row : coordinates) {
        for (int j = 0; j < axes; ++j) {
            if (!std::isfinite(row[j]))
                return {};
            minima[j] = std::min(minima[j], row[j]);
            maxima[j] = std::max(maxima[j], row[j]);
        }
    }
    constexpr double max_coordinate = 255;
    std::array<double, axes> scales;
    for (int j = 0; j < axes; ++j) {
        const double range = static_cast<double>(maxima[j]) - minima[j];
        scales[j] = range > 0 ? max_coordinate / range : 0;
    }

    std::vector<std::pair<std::uint32_t, int>> keys(count);
    for (int i = 0; i < count; ++i) {
        std::uint32_t key = 0;
        for (int j = 0; j < axes; ++j) {
            const double scaled = (static_cast<double>(coordinates[i][j]) - minima[j]) * scales[j];
            const auto bin = static_cast<std::uint32_t>(std::clamp(scaled, 0.0, max_coordinate));
            key |= spread_coordinate_bits(bin) << j;
        }
        keys[i] = {key, i};
    }
    std::sort(keys.begin(), keys.end());

    std::vector<int> order(count);
    for (int i = 0; i < count; ++i)
        order[i] = keys[i].second;
    return order;
}

} // namespace rnndescent
