#pragma once
#include <cassert>
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

    // compute four distances
    virtual void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) = 0;

    // Batch-scoring entry point for reranking an entire shortlist.
    virtual void score_candidates(int query_id, std::vector<SingleNeighbor> &candidates) {
        const size_t end4 = candidates.size() / 4 * 4;
        for (size_t i = 0; i < end4; i += 4) {
            auto *group = candidates.data() + i;
            distances_batch_4(query_id, group[0].id, group[1].id, group[2].id, group[3].id,
                              group[0].distance, group[1].distance, group[2].distance, group[3].distance);
        }
        for (size_t i = end4; i < candidates.size(); ++i)
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

    virtual void *get_query_ptr(int idx0) {
        (void)idx0;
        throw std::runtime_error("get_query_ptr is not implemented for this distance computer");
    }

    virtual int8_t *get_query_ptr_int8(int idx0) {
        return static_cast<int8_t *>(get_query_ptr(idx0));
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
    virtual void copy_index(int idx0, void *y, int &l2norm, bool = false) {
        (void)idx0;
        (void)y;
        (void)l2norm;
        throw std::runtime_error("copy_index<int> is not implemented for this distance computer");
    }

    virtual void copy_index(int idx0, void *y, float &l2norm, bool = false) {
        (void)idx0;
        (void)y;
        (void)l2norm;
        throw std::runtime_error("copy_index<float> is not implemented for this distance computer");
    }

    virtual ~MyDistanceComputer() {}
};
} // namespace rnndescent
