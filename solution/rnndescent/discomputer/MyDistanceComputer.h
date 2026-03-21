#pragma once
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/NNDescent.h>

namespace rnndescent {

struct MyDistanceComputer {

    virtual float operator()(int idq, int i) = 0;

    virtual float symmetric_dis(int i, int j) = 0;

    virtual void set_query(const float *x, int n) = 0; // 一次需要set多个query

    // compute four distances
    virtual void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) = 0;

    virtual int row_count() const {
        throw std::runtime_error("row_count is not implemented for this distance computer");
    }

    virtual void *get_query_ptr(int idx0) {
        (void)idx0;
        throw std::runtime_error("get_query_ptr is not implemented for this distance computer");
    }

    virtual int8_t *get_query_ptr_int8(int idx0) {
        return static_cast<int8_t *>(get_query_ptr(idx0));
    }

    virtual void copy_index(int idx0, void *y, int &l2norm) {
        (void)idx0;
        (void)y;
        (void)l2norm;
        throw std::runtime_error("copy_index<int> is not implemented for this distance computer");
    }

    virtual void copy_index(int idx0, void *y, float &l2norm) {
        (void)idx0;
        (void)y;
        (void)l2norm;
        throw std::runtime_error("copy_index<float> is not implemented for this distance computer");
    }

    virtual ~MyDistanceComputer() {}
};
} // namespace rnndescent