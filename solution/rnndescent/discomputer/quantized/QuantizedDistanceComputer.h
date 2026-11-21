#pragma once

#include "../../common/Logger.h"
#include "../MyDistanceComputer.h"
#include "../utils.h"
#include "NeighborsStorageBase.h"
#include "QuantizedKernelBase.h"

#include <algorithm>
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

// A kernel selects the byte format, SIMD block width and dot products. Both the
// distance computer and the neighbor cache use that same kernel and row stride.
template <class Kernel> class QuantizedDistanceComputer final : public MyDistanceComputer {
    using Quantization = typename Kernel::Quantization;
    using Code = typename Quantization::Code;
    // A 64-byte SIMD block can then be read from one aligned cache line.
    using CodeAllocator = std::conditional_t<Kernel::block_bytes == alignSize, AlignedAllocator<Code>, std::allocator<Code>>;

  public:
    QuantizedDistanceComputer(const float *data, int n, int d, const int *row_order = nullptr)
        : n(n), d(d), stride((this->d + Kernel::block_bytes - 1) / Kernel::block_bytes * Kernel::block_bytes) {
        // Bound the worst-case squared L2 and all dot-product intermediates for every kernel.
        constexpr int max_delta = 2 * SignedInt8Quantization::max_scale;
        constexpr int max_dimension = std::numeric_limits<int32_t>::max() / (max_delta * max_delta);
        if (d > max_dimension)
            throw std::invalid_argument("quantized vectors support at most " + std::to_string(max_dimension) + " dimensions");

        matrix.resize(n * stride);
        matrix_norms.resize(n);
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
        scale = max_range > 0.0f ? max_range / Quantization::max_scale : 1.0f;
        Logger::info("Int8 QT scale = %f; mean = %f; maxvalue = %f\n", scale, mean[0], scale * Quantization::max_scale);

        // Value initialization fills the padding with zero. Encode only the logical dimensions.
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const float *source = data + static_cast<size_t>(row_order ? row_order[i] : i) * d;
            matrix_norms[i] = encode_row(source, matrix.data() + i * stride);
        }
    }

    int row_count() const override { return n; }
    int dimension() const override { return static_cast<int>(d); }
    uint32_t code_stride() const override { return static_cast<uint32_t>(stride); }

    void set_query(const float *data, int count) override {
        query.resize(count * stride);
        query_norms.resize(count);
#pragma omp parallel for
        for (int i = 0; i < count; ++i)
            query_norms[i] = encode_row(data + i * d, query.data() + i * stride);
    }

    // Rank candidates by squared L2 in code units. The index-wide scale^2 preserves their order.
    float operator()(int idq, int i) override {
        const auto dot = Kernel::dot(query.data() + idq * stride, matrix.data() + i * stride, stride);
        return static_cast<float>(query_norms[idq] + matrix_norms[i] - 2 * dot);
    }

    float symmetric_dis(int i, int j) override {
        const auto *codes = matrix.data();
        const auto dot = Kernel::dot(codes + i * stride, codes + j * stride, stride);
        return static_cast<float>(matrix_norms[i] + matrix_norms[j] - 2 * dot);
    }

    void distances_batch_4(int idq, int i0, int i1, int i2, int i3, float &dis0, float &dis1, float &dis2, float &dis3) override {
        const auto *codes = matrix.data();
        const auto dots = Kernel::dot4(query.data() + idq * stride, codes + i0 * stride, codes + i1 * stride,
                                       codes + i2 * stride, codes + i3 * stride, stride);
        const int qnorm = query_norms[idq];
        dis0 = static_cast<float>(qnorm + matrix_norms[i0] - 2 * dots[0]);
        dis1 = static_cast<float>(qnorm + matrix_norms[i1] - 2 * dots[1]);
        dis2 = static_cast<float>(qnorm + matrix_norms[i2] - 2 * dots[2]);
        dis3 = static_cast<float>(qnorm + matrix_norms[i3] - 2 * dots[3]);
    }

    void copy_index(int i, void *out, int &norm, bool include_padding = false) override {
        std::memcpy(out, matrix.data() + i * stride, include_padding ? stride : d);
        norm = matrix_norms[i];
    }

    void *get_query_ptr(int idq) override { return query.data() + idq * stride; }

    QuantizedQuery get_quantized_query(int idq) const override {
        return {query.data() + idq * stride, query_norms[idq]};
    }

  private:
    const int n;
    const size_t d, stride;
    std::vector<Code, CodeAllocator> matrix, query;
    std::vector<int> matrix_norms, query_norms;
    std::vector<float> mean;
    float scale = 0.0f;

    int encode_row(const float *source, Code *code) const {
        int norm = 0;
        for (size_t k = 0; k < d; ++k) {
            code[k] = Quantization::encode(source[k], mean[k], scale);
            norm += code[k] * code[k];
        }
        return norm;
    }
};

template <class Kernel, class = void>
inline constexpr bool has_neighbor_cache_packer = false;

template <class Kernel>
inline constexpr bool has_neighbor_cache_packer<Kernel, std::void_t<decltype(&Kernel::pack_neighbor_cache)>> = true;

template <class Kernel>
struct QuantizedNeighbors : NeighborsStorageBase<QuantizedNeighbors<Kernel>, typename Kernel::Quantization::Code, int> {
    using Code = typename Kernel::Quantization::Code;
    using Base = NeighborsStorageBase<QuantizedNeighbors<Kernel>, Code, int>;
    using Base::Base;
    static constexpr size_t cache_lane_width = Kernel::block_bytes;

    void populate_cache(uint8_t *scratch) {
        if (!this->pool)
            return;

        if constexpr (has_neighbor_cache_packer<Kernel>) {
            Kernel::pack_neighbor_cache(*this->dis, this->edges, this->size, this->code_stride,
                                        this->pool, this->l2norms, scratch);
        } else {
            QuantizedKernelBase<typename Kernel::Quantization, Kernel::block_bytes, 4>::pack_tiles(
                *this->dis, this->edges, this->size, this->code_stride, this->pool, this->l2norms, scratch);
        }
    }

    template <typename Allocator>
    void compute_distance(int idq, std::vector<float, Allocator> &result) {
        result.resize(this->size);
        compute_distance(idq, result.data());
    }

    void compute_distance(int idq, float *result) {
        if (this->pool != nullptr) {
            // Score the prepacked neighborhood and combine the dots with the squared norms.
            const auto query = this->dis->get_quantized_query(idq);
            Kernel::cached_distances(static_cast<const Code *>(query.code), query.squared_norm, this->size, this->code_stride,
                                     this->pool, this->l2norms, result);
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
