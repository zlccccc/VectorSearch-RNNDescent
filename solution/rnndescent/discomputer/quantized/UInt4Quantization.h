#pragma once

#include "../../common/Logger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <utility>
#include <vector>

namespace rnndescent {

namespace detail {

// Round halves away from zero, matching round() followed by saturation.
inline int round_signed_query_code(float value) {
    const float bounded = std::clamp(value, -127.0f, 127.0f);
    const int integral = static_cast<int>(bounded);
    const float fraction = bounded - static_cast<float>(integral);
    return integral + (fraction >= 0.5f) - (fraction <= -0.5f);
}

} // namespace detail

// One byte stores matching coordinates from two consecutive blocks of
// features. Each backend selects the block width consumed by its dot kernel.
template <size_t BlockBytes>
struct PackedUInt4Layout {
    static_assert(BlockBytes > 0);
    static constexpr size_t features_per_block = 2 * BlockBytes;

    // DistanceComputer uses this stride when allocating codes. All SIMD
    // kernels receive whole blocks, including the zero padding.
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

template <size_t BlockBytes>
class PackedUInt4Quantization {
    using Layout = PackedUInt4Layout<BlockBytes>;
    static constexpr int center_code = 7;

  public:
    using Code = uint8_t;
    using BaseParams = int;
    using QueryParams = int;
    static constexpr bool supports_cache = true;
    static constexpr bool same_base_query_codes = false;
    static constexpr bool align_codes = true;
    static constexpr int twice_dot_shift = 4;
    static constexpr int query_multiplier = 1 << (twice_dot_shift - 1);
    static constexpr int max_delta = 127 + query_multiplier * 2 * center_code;

    static constexpr size_t base_stride(size_t d) { return Layout::stride(d); }
    static constexpr size_t query_stride(size_t stride) { return 2 * stride; }
    static constexpr size_t logical_bytes(size_t d) { return Layout::logical_bytes(d); }

    void train(const float *data, int n, int d) {
        // A bounded deterministic sample keeps training small and fits the
        // clipping range to reconstruction error rather than rare extrema.
        centers.resize(d);
        const int count = std::min(n, 4096);
        std::vector<float> values(static_cast<size_t>(count) * d);
        for (int i = 0; i < count; ++i) {
            const auto *source = data + (static_cast<size_t>(i) * n / count) * d;
            auto *sample = values.data() + static_cast<size_t>(i) * d;
            std::copy_n(source, d, sample);
        }
        for (int j = 0; j < d; ++j) {
            double sum = 0;
            for (int i = 0; i < count; ++i)
                sum += values[static_cast<size_t>(i) * d + j];
            centers[j] = static_cast<float>(sum / count);
        }
        double squared_sum = 0;
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < d; ++j) {
                auto &x = values[static_cast<size_t>(i) * d + j];
                x = std::abs(x - centers[j]);
                squared_sum += static_cast<double>(x) * x;
            }
        }
        if (squared_sum == 0) {
            Logger::info("UINT4 graph quantization step = %g, row bytes = %zu\n", step, base_stride(d));
            return;
        }
        const float rms = static_cast<float>(std::sqrt(squared_sum / values.size()));
        constexpr int trials = 25;
        std::array<double, trials> errors;
#pragma omp parallel for
        for (int k = 0; k < trials; ++k) {
            const float candidate_step = rms * (0.15f + k * 0.04f);
            const float inverse = 1.0f / candidate_step;
            double error = 0;
            for (float x : values) {
                const float code = std::min(std::floor(x * inverse + 0.5f), float(center_code));
                const float diff = x - code * candidate_step;
                error += static_cast<double>(diff) * diff;
            }
            errors[k] = error;
        }
        const auto best = std::min_element(errors.begin(), errors.end()) - errors.begin();
        step = rms * (0.15f + best * 0.04f);
        Logger::info("UINT4 graph quantization step = %g, row bytes = %zu\n", step, base_stride(d));
    }

    int encode_base(const float *source, uint8_t *row, size_t d) const {
        int norm = 0;
        const auto encode = [&](size_t j) {
            if (j >= d)
                return 0;
            const int code = static_cast<int>(std::clamp(std::round((source[j] - centers[j]) / step),
                                                        -float(center_code), float(center_code))) + center_code;
            norm += code * code;
            return code;
        };
        Layout::encode(d, row, [&](size_t a, size_t b) {
            return std::pair{encode(a), encode(b)};
        });
        return norm * query_multiplier * query_multiplier;
    }

    int encode_query(const float *source, uint8_t *row, size_t d) const {
        int norm = 0;
        // The base reconstructs to 8 * code in signed byte query units.
#pragma omp simd reduction(+ : norm)
        for (size_t j = 0; j < d; ++j) {
            const float position = (source[j] - centers[j]) / step + center_code;
            const int code = detail::round_signed_query_code(position * query_multiplier);
            row[j] = static_cast<uint8_t>(code);
            norm += code * code;
        }
        return norm;
    }

    static float squared_distance(int base, int query, int dot) {
        return static_cast<float>(query + base - 2 * query_multiplier * dot);
    }

    static float ranking_score(int base, int query, int dot) { return squared_distance(base, query, dot); }

    static float symmetric_distance(const uint8_t *x, const uint8_t *y, size_t d, int, int) {
        int sum = 0;
        for (size_t j = 0; j < d; ++j) {
            const int delta = Layout::code(x, j) - Layout::code(y, j);
            sum += delta * delta;
        }
        return static_cast<float>(sum * query_multiplier * query_multiplier);
    }

  private:
    std::vector<float> centers;
    float step = 1.0f;
};

// Per-row affine codes for the coarse shortlist.
template <size_t BlockBytes>
class RowwiseUInt4Quantization {
    using Layout = PackedUInt4Layout<BlockBytes>;

  public:
    using Code = uint8_t;
    struct BaseParams {
        float scale, offset, norm;
    };
    struct QueryParams {
        float scale;
        int sum;
        float norm;
    };
    static constexpr bool supports_cache = false;
    static constexpr bool same_base_query_codes = false;
    static constexpr bool align_codes = true;
    static constexpr int max_delta = 127;

    static constexpr size_t base_stride(size_t d) { return Layout::stride(d); }
    static constexpr size_t query_stride(size_t stride) { return 2 * stride; }

    BaseParams encode_base(const float *source, uint8_t *code, size_t dim) const {
        const auto centered = [&](size_t j) { return source[j] - center[j]; };
        float offset = centered(0), maximum = offset;
        for (size_t j = 1; j < dim; ++j) {
            const float value = centered(j);
            offset = std::min(offset, value);
            maximum = std::max(maximum, value);
        }
        const float scale = (maximum - offset) / 15;
        const float inverse = scale > 0 ? 1 / scale : 0;
        const auto encode = [&](size_t feature) {
            return feature < dim
                ? static_cast<int>(std::clamp((centered(feature) - offset) * inverse + 0.5f, 0.0f, 15.0f))
                : 0;
        };
        float norm = 0;
        Layout::encode(dim, code, [&](size_t a, size_t b) {
            const int low = encode(a), high = encode(b);
            const float x0 = a < dim ? offset + scale * low : 0;
            const float x1 = b < dim ? offset + scale * high : 0;
            norm += x0 * x0 + x1 * x1;
            return std::pair{low, high};
        });
        return {scale, offset, norm};
    }

    QueryParams encode_query(const float *source, uint8_t *code, size_t dim) const {
        float max_abs = 0;
#pragma omp simd reduction(max : max_abs)
        for (size_t j = 0; j < dim; ++j)
            max_abs = std::max(max_abs, std::abs(source[j] - center[j]));
        const float scale = max_abs / 127;
        const float inverse = max_abs > 0 ? 127 / max_abs : 0;
        int sum = 0, squared_sum = 0;
#pragma omp simd reduction(+ : sum, squared_sum)
        for (size_t j = 0; j < dim; ++j) {
            const int value = detail::round_signed_query_code((source[j] - center[j]) * inverse);
            code[j] = static_cast<uint8_t>(value);
            sum += value;
            squared_sum += value * value;
        }
        return {scale, sum, scale * scale * squared_sum};
    }

    static float ranking_score(const BaseParams &base, const QueryParams &query, int dot) {
        const float product = base.scale * dot + base.offset * query.sum;
        // The query norm is constant across a shortlist.
        return base.norm - 2 * query.scale * product;
    }

    static float squared_distance(const BaseParams &base, const QueryParams &query, int dot) {
        return query.norm + ranking_score(base, query, dot);
    }

    static float symmetric_distance(const uint8_t *a, const uint8_t *b, size_t d,
                                    const BaseParams &x, const BaseParams &y) {
        float sum = 0;
        for (size_t j = 0; j < d; ++j) {
            const float delta = (x.offset + x.scale * Layout::code(a, j)) - (y.offset + y.scale * Layout::code(b, j));
            sum += delta * delta;
        }
        return sum;
    }

    void train(const float *data, int n, int dim) {
        const int threads = omp_get_max_threads();
        std::vector<double> sums(static_cast<size_t>(threads) * dim);
        center.resize(dim);
#pragma omp parallel
        {
            auto *local = sums.data() + static_cast<size_t>(omp_get_thread_num()) * dim;
#pragma omp for
            for (int i = 0; i < n; ++i) {
                const auto *row = data + static_cast<size_t>(i) * dim;
                for (int j = 0; j < dim; ++j)
                    local[j] += row[j];
            }
        }
        for (int j = 0; j < dim; ++j) {
            double sum = 0;
            for (int t = 0; t < threads; ++t)
                sum += sums[static_cast<size_t>(t) * dim + j];
            center[j] = static_cast<float>(sum / n);
        }
    }

  private:
    std::vector<float> center;
};

} // namespace rnndescent
