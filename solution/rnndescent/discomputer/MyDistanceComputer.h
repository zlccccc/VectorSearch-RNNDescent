#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "utils.h"

namespace rnndescent {

struct QuantizedQuery {
    const void *code;
    int squared_norm;
};

struct MyDistanceComputer {

    virtual float operator()(int idq, int i) = 0;

    virtual float symmetric_dis(int i, int j) = 0;

    virtual void set_query(const float *x, int n) = 0; // 一次需要set多个query

    // Score an adjacency list padded to groups of four, in ID order.
    virtual void score_neighbors(int query_id, const int *ids, int count, float *scores) {
        for (int i = 0; i < count; ++i)
            scores[i] = (*this)(query_id, ids[i]);
    }

    // Score a contiguous range of candidate IDs in place.
    virtual void score_candidates(int query_id, SingleNeighbor *candidates, size_t count) {
        for (size_t i = 0; i < count; ++i)
            candidates[i].distance = (*this)(query_id, candidates[i].id);
    }

    virtual int row_count() const {
        throw std::runtime_error("row_count is not implemented for this distance computer");
    }

    virtual int dimension() const {
        throw std::runtime_error("dimension is not implemented for this distance computer");
    }

    // Number of bytes in a quantized row, including zero padding for SIMD loads.
    virtual uint32_t code_stride() const {
        throw std::runtime_error("code_stride is not implemented for this distance computer");
    }

    virtual QuantizedQuery get_quantized_query(int) const {
        throw std::runtime_error("get_quantized_query is not implemented for this distance computer");
    }

    virtual void pack_neighbor_cache(const int *, int, void *, int *, uint8_t *) {
        throw std::runtime_error("neighbor packing is not implemented for this distance computer");
    }

    virtual void compute_cached_distances(int, int, const void *, const int *, float *) const {
        throw std::runtime_error("cached distances are not implemented for this distance computer");
    }

    // Copy the logical code bytes by default, or the full stride for the neighbor cache.
    virtual void copy_index(int, void *, int &, bool = false) {
        throw std::runtime_error("copy_index is not implemented for this distance computer");
    }

    virtual ~MyDistanceComputer() = default;
};
} // namespace rnndescent
