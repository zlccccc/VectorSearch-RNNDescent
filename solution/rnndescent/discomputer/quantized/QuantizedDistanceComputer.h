#pragma once

#include "../../common/HugePages.h"
#include "../../common/Logger.h"
#include "../MyDistanceComputer.h"
#include "../utils.h"
#include "NeighborsStorageBase.h"
#include "QuantizedKernelBase.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rnndescent {

struct SignedInt8Quantization {
    using Code = int8_t;
    static constexpr int max_scale = 127;

    static Code encode(float value, float center, float scale) {
        const float rounded = std::round((value - center) / scale);
        return static_cast<Code>(std::clamp(rounded, -float(max_scale), float(max_scale)));
    }
};

struct UInt8Quantization {
    using Code = uint8_t;
    static constexpr int max_scale = 63;

    // The offset keeps codes in [0, 126], so either operand fits in a signed byte.
    static Code encode(float value, float center, float scale) {
        const float rounded = std::round((value - center) / scale);
        return static_cast<Code>(std::clamp(rounded, -float(max_scale), float(max_scale)) + max_scale);
    }
};

template <class Kernel, class = void>
inline constexpr bool has_neighbor_cache_packer = false;

template <class Kernel>
inline constexpr bool has_neighbor_cache_packer<Kernel, std::void_t<decltype(&Kernel::pack_neighbor_cache)>> = true;

template <class Format, size_t BlockBytes>
class ByteQuantization {
  public:
    using Code = typename Format::Code;
    using BaseParams = int;
    using QueryParams = int;
    static constexpr bool supports_cache = true;
    static constexpr bool same_base_query_codes = true;
    static constexpr bool align_codes = BlockBytes == alignSize;
    // Keep the same bound for the signed and unsigned byte encoders.
    static constexpr int max_delta = 2 * SignedInt8Quantization::max_scale;

    static constexpr size_t base_stride(size_t d) { return (d + BlockBytes - 1) / BlockBytes * BlockBytes; }
    static constexpr size_t query_stride(size_t stride) { return stride; }
    static constexpr size_t logical_bytes(size_t d) { return d; }

    void train(const float *data, int n, int d) {
        mean.resize(d);
        float max_range = 0.0f;
#pragma omp parallel for reduction(max : max_range)
        for (int k = 0; k < d; ++k) {
            float min_value = std::numeric_limits<float>::infinity();
            float max_value = -min_value;
            for (int i = 0; i < n; ++i) {
                const float value = data[static_cast<size_t>(i) * d + k];
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
            }
            mean[k] = (max_value + min_value) / 2;
            max_range = std::max(max_range, (max_value - min_value) / 2);
        }
        scale = max_range > 0.0f ? max_range / Format::max_scale : 1.0f;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * Format::max_scale);
    }

    int encode_base(const float *source, Code *code, size_t d) const {
        int norm = 0;
        for (size_t k = 0; k < d; ++k) {
            code[k] = Format::encode(source[k], mean[k], scale);
            norm += code[k] * code[k];
        }
        return norm;
    }

    int encode_query(const float *source, Code *code, size_t d) const { return encode_base(source, code, d); }

    // Scores use code units. The index-wide scale squared preserves ranking.
    static float squared_distance(int base, int query, int dot) {
        return static_cast<float>(query + base - 2 * dot);
    }
    static float ranking_score(int base, int query, int dot) { return squared_distance(base, query, dot); }

  private:
    std::vector<float> mean;
    float scale = 0.0f;
};

// The encoding owns training, format and score conversion. The kernel owns
// dot products and the cache layout. Every format uses the same row storage
// and DistanceComputer entry points.
template <class Kernel, class Encoding = ByteQuantization<typename Kernel::Quantization, Kernel::block_bytes>>
class QuantizedDistanceComputer final : public MyDistanceComputer {
    using Code = typename Encoding::Code;
    using BaseParams = typename Encoding::BaseParams;
    using QueryParams = typename Encoding::QueryParams;
    using CodeAllocator = std::conditional_t<Encoding::align_codes, AlignedAllocator<Code>, std::allocator<Code>>;
    using ParamsAllocator = std::conditional_t<Encoding::supports_cache, std::allocator<BaseParams>, AlignedAllocator<BaseParams>>;

  public:
    QuantizedDistanceComputer(const float *data, int n, int d, const int *row_order = nullptr)
        : n_(n), d_(d), stride_(Encoding::base_stride(d)), query_stride_(Encoding::query_stride(stride_)) {
        // Bound the squared distance and all integer dot-product intermediates.
        constexpr int max_dimension = std::numeric_limits<int32_t>::max() / (Encoding::max_delta * Encoding::max_delta);
        if (d > max_dimension)
            throw std::invalid_argument("quantized vectors support at most " + std::to_string(max_dimension) + " dimensions");
        RNNDESCENT_ASSERT_MSG(data != nullptr && n > 0 && d > 0, "quantization requires a nonempty base matrix");
        encoding_.train(data, n, d);
        matrix_.resize(static_cast<size_t>(n) * stride_);
        base_params_.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const auto *source = data + static_cast<size_t>(row_order ? row_order[i] : i) * d_;
            base_params_[i] = encoding_.encode_base(source, matrix_.data() + static_cast<size_t>(i) * stride_, d_);
        }
        detail::try_huge_pages(matrix_.data(), matrix_.size() * sizeof(Code));
        detail::try_huge_pages(base_params_.data(), base_params_.size() * sizeof(BaseParams));
    }

    int row_count() const override { return n_; }
    int dimension() const override { return d_; }
    uint32_t code_stride() const override { return static_cast<uint32_t>(stride_); }

    void set_query(const float *data, int count) override {
        queries_.resize(static_cast<size_t>(count) * query_stride_);
        query_params_.resize(count);
#pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            const auto *source = data + static_cast<size_t>(i) * d_;
            query_params_[i] = encoding_.encode_query(source, queries_.data() + static_cast<size_t>(i) * query_stride_, d_);
        }
    }

    float operator()(int q, int i) override {
        return Encoding::squared_distance(base_params_[i], query_params_[q], dot(query(q), row(i)));
    }

    float symmetric_dis(int i, int j) override {
        if constexpr (Encoding::same_base_query_codes) {
            return Encoding::squared_distance(base_params_[i], base_params_[j], dot(row(i), row(j)));
        } else {
            return Encoding::symmetric_distance(row(i), row(j), d_, base_params_[i], base_params_[j]);
        }
    }

    void distances_batch_4(int q, int a, int b, int c, int d, float &da, float &db, float &dc, float &dd) override {
        const auto products = dot4(query(q), a, b, c, d);
        const auto &params = query_params_[q];
        da = Encoding::squared_distance(base_params_[a], params, products[0]);
        db = Encoding::squared_distance(base_params_[b], params, products[1]);
        dc = Encoding::squared_distance(base_params_[c], params, products[2]);
        dd = Encoding::squared_distance(base_params_[d], params, products[3]);
    }

    void score_candidates(int q, std::vector<SingleNeighbor> &candidates) override {
        const auto *code = query(q);
        const auto &params = query_params_[q];
        size_t i = 0;
        for (; i + 4 <= candidates.size(); i += 4) {
            auto *group = candidates.data() + i;
            const auto products = dot4(code, group[0].id, group[1].id, group[2].id, group[3].id);
            for (int j = 0; j < 4; ++j)
                group[j].distance = Encoding::ranking_score(base_params_[group[j].id], params, products[j]);
        }
        for (; i < candidates.size(); ++i) {
            auto &candidate = candidates[i];
            const auto product = dot(code, row(candidate.id));
            candidate.distance = Encoding::ranking_score(base_params_[candidate.id], params, product);
        }
    }

    void *get_query_ptr(int q) override { return queries_.data() + static_cast<size_t>(q) * query_stride_; }

    QuantizedQuery get_quantized_query(int q) const override {
        if constexpr (Encoding::supports_cache) {
            return {query(q), query_params_[q]};
        } else {
            return MyDistanceComputer::get_quantized_query(q);
        }
    }

    void copy_index(int i, void *out, int &norm, bool include_padding = false) override {
        if constexpr (Encoding::supports_cache) {
            std::memcpy(out, row(i), (include_padding ? stride_ : Encoding::logical_bytes(d_)) * sizeof(Code));
            norm = base_params_[i];
        } else {
            MyDistanceComputer::copy_index(i, out, norm, include_padding);
        }
    }

    void pack_neighbor_cache(const int *ids, int count, void *codes, int *norms, uint8_t *scratch) override {
        if constexpr (!Encoding::supports_cache) {
            MyDistanceComputer::pack_neighbor_cache(ids, count, codes, norms, scratch);
        } else if constexpr (has_neighbor_cache_packer<Kernel>) {
            Kernel::pack_neighbor_cache(*this, ids, count, code_stride(), static_cast<Code *>(codes), norms, scratch);
        } else {
            QuantizedKernelBase<Encoding, Kernel::block_bytes, 4>::pack_tiles(
                *this, ids, count, code_stride(), static_cast<Code *>(codes), norms, scratch);
        }
    }

    void compute_cached_distances(int q, int count, const void *codes, const int *norms, float *result) const override {
        if constexpr (Encoding::supports_cache) {
            Kernel::cached_distances(query(q), query_params_[q], count, code_stride(),
                                     static_cast<const Code *>(codes), norms, result);
        } else {
            MyDistanceComputer::compute_cached_distances(q, count, codes, norms, result);
        }
    }

  private:
    const int n_, d_;
    const size_t stride_, query_stride_;
    Encoding encoding_;
    std::vector<Code, CodeAllocator> matrix_, queries_;
    std::vector<BaseParams, ParamsAllocator> base_params_;
    std::vector<QueryParams> query_params_;

    const Code *row(int i) const { return matrix_.data() + static_cast<size_t>(i) * stride_; }
    const Code *query(int i) const { return queries_.data() + static_cast<size_t>(i) * query_stride_; }

    int dot(const Code *q, const Code *x) const {
        return Kernel::dot(q, x, stride_);
    }

    std::array<int, 4> dot4(const Code *q, int a, int b, int c, int d) const {
        return Kernel::dot4(q, row(a), row(b), row(c), row(d), stride_);
    }
};

template <class Kernel>
struct QuantizedNeighbors : NeighborsStorageBase<QuantizedNeighbors<Kernel>, typename Kernel::Quantization::Code, int> {
    using Code = typename Kernel::Quantization::Code;
    using Base = NeighborsStorageBase<QuantizedNeighbors<Kernel>, Code, int>;
    using Base::Base;
    static constexpr size_t cache_lane_width = Kernel::block_bytes;

    void populate_cache(uint8_t *scratch) {
        if (this->pool)
            this->dis->pack_neighbor_cache(this->edges, this->size, this->pool, this->l2norms, scratch);
    }

    template <typename Allocator>
    void compute_distance(int idq, std::vector<float, Allocator> &result) {
        result.resize(this->size);
        compute_distance(idq, result.data());
    }

    void compute_distance(int idq, float *result) {
        if (this->pool != nullptr) {
            this->dis->compute_cached_distances(idq, this->size, this->pool, this->l2norms, result);
        } else {
            // Uncached neighborhoods fetch quantized rows by ID and score four at a time.
            // Time spent in this fallback limits the end-to-end gain from the cached kernel.
            const auto *ids = this->edges;
            for (int i = 0; i < this->size; i += 4) {
                this->dis->distances_batch_4(idq, ids[i], ids[i + 1], ids[i + 2], ids[i + 3], result[i], result[i + 1], result[i + 2], result[i + 3]);
            }
        }
    }
};

} // namespace rnndescent
