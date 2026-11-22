#pragma once

#include "QuantizedDistanceComputer.h"

#include <array>
#include <omp.h>
#include <utility>

namespace rnndescent {

// One byte stores matching coordinates from two consecutive blocks of
// features. Each backend selects the block width consumed by its dot kernel.
template <size_t BlockBytes>
struct PackedUInt4Layout {
    static_assert(BlockBytes > 0);
    static constexpr size_t features_per_block = 2 * BlockBytes;

    static constexpr size_t stride(size_t dimension) {
        return (dimension + features_per_block - 1) / features_per_block * BlockBytes;
    }

    static constexpr size_t logical_bytes(size_t dimension) {
        return dimension / features_per_block * BlockBytes + std::min(dimension % features_per_block, BlockBytes);
    }

    static int code(const uint8_t *row, size_t feature) {
        const uint8_t byte = row[feature / features_per_block * BlockBytes + feature % BlockBytes];
        return feature % features_per_block < BlockBytes ? byte & 15 : byte >> 4;
    }

    template <class EncodePair>
    static void encode(size_t dimension, uint8_t *row, EncodePair encode_pair) {
        for (size_t tile = 0; tile < dimension; tile += features_per_block) {
            for (size_t j = 0; j < BlockBytes; ++j) {
                const auto [low, high] = encode_pair(tile + j, tile + BlockBytes + j);
                row[tile / 2 + j] = static_cast<uint8_t>(low | (high << 4));
            }
        }
    }
};

template <class Kernel>
class UInt4GraphDistanceComputer final : public MyDistanceComputer {
    static constexpr int scale_ratio = Kernel::query_multiplier;
    static constexpr int center_code = 7;
    using Layout = PackedUInt4Layout<Kernel::block_bytes>;

  public:
    static constexpr int pca_dimension = 64;

    UInt4GraphDistanceComputer(const float *data, int n, int d, const int *order = nullptr)
        : n_(n), d_(d), stride_(packed_stride(n, d)),
          matrix_(static_cast<size_t>(n) * stride_), base_norms_(n), centers_(d) {
        train(data);
        Logger::info("PCA UINT4 quantization step = %g, row bytes = %zu\n", step_, stride_);
#pragma omp parallel for
        for (int i = 0; i < n_; ++i) {
            auto *packed = matrix_.data() + static_cast<size_t>(i) * stride_;
            const auto *row = data + static_cast<size_t>(order ? order[i] : i) * d_;
            int norm = 0;
            const auto encode = [&](size_t j) {
                if (j >= static_cast<size_t>(d_))
                    return 0;
                const int code = static_cast<int>(std::clamp(std::round((row[j] - centers_[j]) / step_),
                                                            -float(center_code), float(center_code))) + center_code;
                norm += code * code;
                return code;
            };
            Layout::encode(d_, packed, [&](size_t low, size_t high) {
                return std::pair{encode(low), encode(high)};
            });
            base_norms_[i] = norm * scale_ratio * scale_ratio;
        }
        detail::try_huge_pages(matrix_.data(), matrix_.size());
        detail::try_huge_pages(base_norms_.data(), base_norms_.size() * sizeof(int));
    }

    int row_count() const override { return n_; }
    int dimension() const override { return d_; }
    uint32_t code_stride() const override { return static_cast<uint32_t>(stride_); }

    void set_query(const float *data, int count) override {
        const size_t query_stride = 2 * stride_;
        queries_.resize(static_cast<size_t>(count) * query_stride);
        query_norms_.resize(count);
        // Quantize queries directly to signed bytes for the dot kernel.
        // The base row reconstructs to 8 * code in query units.
#pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            const auto *row = data + static_cast<size_t>(i) * d_;
            auto *query = queries_.data() + static_cast<size_t>(i) * query_stride;
            int norm = 0;
            for (int j = 0; j < d_; ++j) {
                const float position = (row[j] - centers_[j]) / step_ + center_code;
                const int code = static_cast<int>(std::clamp(std::round(position * scale_ratio), -127.0f, 127.0f));
                query[j] = static_cast<uint8_t>(code);
                norm += code * code;
            }
            query_norms_[i] = norm;
        }
    }

    QuantizedQuery get_quantized_query(int i) const override {
        return {queries_.data() + static_cast<size_t>(i) * 2 * stride_, query_norms_[i]};
    }

    void *get_query_ptr(int i) override {
        return queries_.data() + static_cast<size_t>(i) * 2 * stride_;
    }

    void copy_index(int i, void *out, int &norm, bool include_padding) override {
        std::memcpy(out, row_codes(i), include_padding ? stride_ : Layout::logical_bytes(d_));
        norm = base_norms_[i];
    }

    void pack_neighbor_cache(const int *ids, int count, void *codes, int *norms, uint8_t *scratch) override {
        Kernel::pack_neighbor_cache(*this, ids, count, code_stride(), static_cast<uint8_t *>(codes), norms, scratch);
    }

    void compute_cached_distances(int idq, int count, const void *codes, const int *norms, float *result) const override {
        const auto q = get_quantized_query(idq);
        Kernel::cached_distances(static_cast<const uint8_t *>(q.code), q.squared_norm, count, code_stride(),
                                 static_cast<const uint8_t *>(codes), norms, result);
    }

    float operator()(int idq, int i) override {
        const auto q = get_quantized_query(idq);
        const auto *query = static_cast<const uint8_t *>(q.code);
        return static_cast<float>(q.squared_norm + base_norms_[i] - 2 * scale_ratio * Kernel::dot(query, row_codes(i), stride_));
    }

    float symmetric_dis(int i, int j) override {
        const auto *x = row_codes(i);
        const auto *y = row_codes(j);
        int norm = 0;
        for (int k = 0; k < d_; ++k) {
            const int delta = Layout::code(x, k) - Layout::code(y, k);
            norm += delta * delta;
        }
        return static_cast<float>(norm * scale_ratio * scale_ratio);
    }

    void distances_batch_4(int q, int a, int b, int c, int d, float &da, float &db, float &dc, float &dd) override {
        const auto *query = static_cast<const uint8_t *>(get_quantized_query(q).code);
        std::array<float, 4> distances;
        Kernel::graph_distances4(query, row_codes(a), row_codes(b), row_codes(c), row_codes(d), stride_,
                                 query_norms_[q], {base_norms_[a], base_norms_[b], base_norms_[c], base_norms_[d]}, distances.data());
        da = distances[0];
        db = distances[1];
        dc = distances[2];
        dd = distances[3];
    }

  private:
    const int n_, d_;
    const size_t stride_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> matrix_, queries_;
    std::vector<int> base_norms_, query_norms_;
    std::vector<float> centers_;
    float step_ = 1;

    const uint8_t *row_codes(int i) const { return matrix_.data() + static_cast<size_t>(i) * stride_; }

    static size_t packed_stride(int rows, int dimension) {
        constexpr int max_delta = 127 + scale_ratio * 2 * center_code;
        constexpr int max_dimension = std::numeric_limits<int32_t>::max() / (max_delta * max_delta);
        if (rows <= 0 || dimension <= 0 || dimension > max_dimension)
            throw std::invalid_argument("unsupported UINT4 graph matrix size");
        return Layout::stride(dimension);
    }

    void train(const float *data) {
        // A bounded deterministic sample keeps training small and fits the
        // clipping range to reconstruction error rather than rare extrema.
        const int count = std::min(n_, 4096);
        std::vector<float> values(static_cast<size_t>(count) * d_);
        for (int i = 0; i < count; ++i) {
            const auto *source = data + (static_cast<size_t>(i) * n_ / count) * d_;
            auto *sample = values.data() + static_cast<size_t>(i) * d_;
            std::copy_n(source, d_, sample);
        }
        for (int j = 0; j < d_; ++j) {
            double sum = 0;
            for (int i = 0; i < count; ++i)
                sum += values[static_cast<size_t>(i) * d_ + j];
            centers_[j] = static_cast<float>(sum / count);
        }
        double squared_sum = 0;
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < d_; ++j) {
                auto &x = values[static_cast<size_t>(i) * d_ + j];
                x = std::abs(x - centers_[j]);
                squared_sum += static_cast<double>(x) * x;
            }
        }
        if (squared_sum == 0)
            return;
        const float rms = static_cast<float>(std::sqrt(squared_sum / values.size()));
        constexpr int trials = 25;
        std::array<double, trials> errors;
#pragma omp parallel for
        for (int k = 0; k < trials; ++k) {
            const float step = rms * (0.15f + k * 0.04f);
            const float inverse = 1.0f / step;
            double error = 0;
            for (float x : values) {
                const float code = std::min(std::floor(x * inverse + 0.5f), float(center_code));
                const float diff = x - code * step;
                error += static_cast<double>(diff) * diff;
            }
            errors[k] = error;
        }
        const auto best = std::min_element(errors.begin(), errors.end()) - errors.begin();
        step_ = rms * (0.15f + best * 0.04f);
    }
};

// Affine per-row codes for shortlisting in the original vector space. The
// selected backend supplies integer dot products on packed base and byte queries.
template <class Kernel, int Dimension>
class UInt4RerankDistanceComputer final : public MyDistanceComputer {
    using Layout = PackedUInt4Layout<Kernel::block_bytes>;
    static constexpr size_t packed_stride = Layout::stride(Dimension);
    static constexpr size_t query_stride = 2 * packed_stride;
    static_assert(Dimension > 0 && Dimension <= std::numeric_limits<int32_t>::max() / (127 * 127));

  public:
    UInt4RerankDistanceComputer(const float *data, int n, const int *row_order)
        : n_(n), center_(find_center(data, n)), codes_(static_cast<size_t>(n) * packed_stride), rows_(n) {
#pragma omp parallel for
        for (int i = 0; i < n_; ++i) {
            const auto *source = data + static_cast<size_t>(row_order ? row_order[i] : i) * Dimension;
            alignas(32) float centered[Dimension];
            for (int j = 0; j < Dimension; ++j)
                centered[j] = source[j] - center_[j];

            const auto extrema = std::minmax_element(centered, centered + Dimension);
            const float offset = *extrema.first;
            const float scale = (*extrema.second - offset) / 15;
            const float inverse = scale > 0 ? 1 / scale : 0;
            const auto encode = [&](size_t feature) {
                return feature < Dimension
                    ? static_cast<int>(std::clamp((centered[feature] - offset) * inverse + 0.5f, 0.0f, 15.0f))
                    : 0;
            };
            float norm = 0;
            Layout::encode(Dimension, codes_.data() + static_cast<size_t>(i) * packed_stride, [&](size_t a, size_t b) {
                const int low = encode(a), high = encode(b);
                const float x0 = a < Dimension ? offset + scale * low : 0;
                const float x1 = b < Dimension ? offset + scale * high : 0;
                norm += x0 * x0 + x1 * x1;
                return std::pair{low, high};
            });
            rows_[i] = {scale, offset, norm};
        }
        detail::try_huge_pages(codes_.data(), codes_.size());
        detail::try_huge_pages(rows_.data(), rows_.size() * sizeof(Row));
    }

    int row_count() const override { return n_; }
    int dimension() const override { return Dimension; }

    void set_query(const float *data, int count) override {
        queries_.resize(static_cast<size_t>(count) * query_stride);
        query_params_.resize(count);
#pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            const auto *source = data + static_cast<size_t>(i) * Dimension;
            auto *code = queries_.data() + static_cast<size_t>(i) * query_stride;
            alignas(32) float centered[Dimension];
            float max_abs = 0;
            for (int j = 0; j < Dimension; ++j) {
                centered[j] = source[j] - center_[j];
                max_abs = std::max(max_abs, std::abs(centered[j]));
            }
            const float scale = max_abs / 127;
            const float inverse = max_abs > 0 ? 127 / max_abs : 0;
            int sum = 0, squared_sum = 0;
            for (int j = 0; j < Dimension; ++j) {
                const int value = static_cast<int>(std::clamp(std::round(centered[j] * inverse), -127.0f, 127.0f));
                code[j] = static_cast<uint8_t>(value);
                sum += value;
                squared_sum += value * value;
            }
            query_params_[i] = {scale, sum, scale * scale * squared_sum};
        }
    }

    float operator()(int q, int i) override {
        const auto &params = query_params_[q];
        return params.norm + score(params, i, Kernel::template dot<packed_stride>(query(q), row(i), packed_stride));
    }

    float symmetric_dis(int i, int j) override {
        const auto *a = row(i), *b = row(j);
        const auto &x = rows_[i], &y = rows_[j];
        float sum = 0;
        for (int k = 0; k < Dimension; ++k) {
            const float delta = (x.offset + x.scale * Layout::code(a, k)) - (y.offset + y.scale * Layout::code(b, k));
            sum += delta * delta;
        }
        return sum;
    }

    void distances_batch_4(int q, int a, int b, int c, int d, float &da, float &db, float &dc, float &dd) override {
        const auto dots = products4(query(q), a, b, c, d);
        const auto &params = query_params_[q];
        da = params.norm + score(params, a, dots[0]);
        db = params.norm + score(params, b, dots[1]);
        dc = params.norm + score(params, c, dots[2]);
        dd = params.norm + score(params, d, dots[3]);
    }

    void score_candidates(int q, std::vector<SingleNeighbor> &candidates) override {
        const auto *encoded = query(q);
        const auto &params = query_params_[q];
        // The query norm is constant across this shortlist.
        size_t i = 0;
        for (; i + 4 <= candidates.size(); i += 4) {
            auto *group = candidates.data() + i;
            const auto dots = products4(encoded, group[0].id, group[1].id, group[2].id, group[3].id);
            for (int j = 0; j < 4; ++j)
                group[j].distance = score(params, group[j].id, dots[j]);
        }
        for (; i < candidates.size(); ++i) {
            auto &candidate = candidates[i];
            const int dot = Kernel::template dot<packed_stride>(encoded, row(candidate.id), packed_stride);
            candidate.distance = score(params, candidate.id, dot);
        }
    }

  private:
    struct Row {
        float scale, offset, norm;
    };
    struct Query {
        float scale;
        int sum;
        float norm;
    };

    const int n_;
    const std::array<float, Dimension> center_;
    std::vector<uint8_t, AlignedAllocator<uint8_t>> codes_, queries_;
    std::vector<Row, AlignedAllocator<Row>> rows_;
    std::vector<Query> query_params_;

    const uint8_t *row(int i) const { return codes_.data() + static_cast<size_t>(i) * packed_stride; }
    const uint8_t *query(int q) const { return queries_.data() + static_cast<size_t>(q) * query_stride; }

    std::array<int, 4> products4(const uint8_t *q, int a, int b, int c, int d) const {
        return Kernel::template dot4<packed_stride>(q, row(a), row(b), row(c), row(d), packed_stride);
    }

    float score(const Query &q, int id, int dot) const {
        const auto &x = rows_[id];
        const float product = x.scale * dot + x.offset * q.sum;
        return x.norm - 2 * q.scale * product;
    }

    static std::array<float, Dimension> find_center(const float *data, int n) {
        RNNDESCENT_ASSERT_MSG(data != nullptr && n > 0, "UINT4 rerank requires a nonempty base matrix");
        std::vector<std::array<double, Dimension>> sums(omp_get_max_threads());
#pragma omp parallel
        {
            auto &local = sums[omp_get_thread_num()];
#pragma omp for
            for (int i = 0; i < n; ++i) {
                const auto *row = data + static_cast<size_t>(i) * Dimension;
                for (int j = 0; j < Dimension; ++j)
                    local[j] += row[j];
            }
        }

        std::array<float, Dimension> center;
        for (int j = 0; j < Dimension; ++j) {
            double sum = 0;
            for (const auto &local : sums)
                sum += local[j];
            center[j] = static_cast<float>(sum / n);
        }
        return center;
    }
};

} // namespace rnndescent
