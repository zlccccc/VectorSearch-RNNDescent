#pragma once
#include <cassert>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/NNDescent.h>

namespace rnndescent {

struct MyDistanceComputer {

    virtual float operator()(int idq, int i) = 0;

    virtual float symmetric_dis(int i, int j) = 0;

    virtual void set_query(const float *x, int n) = 0; // 一次需要set多个query

    // compute four distances
    virtual void distances_batch_4(int idq, int idx0, int idx1, int idx2, int idx3, float &dis0, float &dis1, float &dis2, float &dis3) = 0;

    virtual void *get_query_ptr(int idx0) { assert(0); }

    virtual void copy_index(int idx0, void *y, int &l2norm) { assert(0); }

    virtual void copy_index(int idx0, void *y, float &l2norm) { assert(0); }

    virtual ~MyDistanceComputer() {}
};
} // namespace rnndescent