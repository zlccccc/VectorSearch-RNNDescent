#pragma once

#include "Assertions.h"

#include <vector>

namespace rnndescent {

struct FloatMatrixView {
  public:
    FloatMatrixView() = default;
    FloatMatrixView(const float *data, int rows, int dim) : data_(data), rows(rows), dim(dim) {}

    static FloatMatrixView from_buffer(const float *data, int rows, int dim) { return FloatMatrixView(data, rows, dim); }

    static FloatMatrixView from_vector(const std::vector<float> &data, int dim) {
        RNNDESCENT_ASSERT_MSG(dim > 0, "matrix view requires positive dimension");
        RNNDESCENT_ASSERT_MSG(data.size() % dim == 0, "matrix view vector size must be divisible by dimension");
        return FloatMatrixView(data.data(), static_cast<int>(data.size() / dim), dim);
    }

    void validate(const char *name) const {
        (void)name;
        RNNDESCENT_ASSERT_MSG(data_ != nullptr, "matrix view data pointer is null");
        RNNDESCENT_ASSERT_MSG(rows > 0, "matrix view requires positive row count");
        RNNDESCENT_ASSERT_MSG(dim > 0, "matrix view requires positive dimension");
    }

    const float *data_ptr() const { return data_; }
    int row_count() const { return rows; }
    int dimension() const { return dim; }

    const float *row_ptr(int row) const { return data_ + static_cast<size_t>(row) * dim; }

  private:
    const float *data_ = nullptr;
    int rows = 0;
    int dim = 0;
};

struct MutableFloatMatrixView {
  public:
    MutableFloatMatrixView() = default;
    MutableFloatMatrixView(float *data, int rows, int dim) : data_(data), rows(rows), dim(dim) {}

    static MutableFloatMatrixView from_buffer(float *data, int rows, int dim) { return MutableFloatMatrixView(data, rows, dim); }

    static MutableFloatMatrixView from_vector(std::vector<float> &data, int dim) {
        RNNDESCENT_ASSERT_MSG(dim > 0, "mutable matrix view requires positive dimension");
        RNNDESCENT_ASSERT_MSG(data.size() % dim == 0, "mutable matrix view vector size must be divisible by dimension");
        return MutableFloatMatrixView(data.data(), static_cast<int>(data.size() / dim), dim);
    }

    void validate(const char *name) const {
        (void)name;
        RNNDESCENT_ASSERT_MSG(data_ != nullptr, "mutable matrix view data pointer is null");
        RNNDESCENT_ASSERT_MSG(rows > 0, "mutable matrix view requires positive row count");
        RNNDESCENT_ASSERT_MSG(dim > 0, "mutable matrix view requires positive dimension");
    }

    float *data_ptr() const { return data_; }
    int row_count() const { return rows; }
    int dimension() const { return dim; }

    float *row_ptr(int row) const { return data_ + static_cast<size_t>(row) * dim; }

  private:
    float *data_ = nullptr;
    int rows = 0;
    int dim = 0;
};

struct SearchResultView {
  public:
    SearchResultView() = default;
    SearchResultView(int *indices, float *distances, int topk) : indices_(indices), distances_(distances), topk_(topk) {}

    static SearchResultView from_buffers(int *indices, float *distances, int topk) { return SearchResultView(indices, distances, topk); }

    static SearchResultView from_vectors(std::vector<int> &indices, std::vector<float> &distances, int topk) {
        RNNDESCENT_ASSERT_MSG(topk > 0, "search topk must be positive");
        RNNDESCENT_ASSERT_MSG(indices.size() == distances.size(), "search result vector sizes must match");
        return SearchResultView(indices.data(), distances.data(), topk);
    }

    void validate() const {
        RNNDESCENT_ASSERT_MSG(indices_ != nullptr, "search result indices buffer is null");
        RNNDESCENT_ASSERT_MSG(distances_ != nullptr, "search result distance buffer is null");
        RNNDESCENT_ASSERT_MSG(topk_ > 0, "search topk must be positive");
    }

    int *indices_ptr() const { return indices_; }
    float *distances_ptr() const { return distances_; }
    int topk() const { return topk_; }

    SearchResultView slice(int queryid) const {
        validate();
        return SearchResultView(indices_ + queryid * topk_, distances_ + queryid * topk_, topk_);
    }

  private:
    int *indices_ = nullptr;
    float *distances_ = nullptr;
    int topk_ = 0;
};

} // namespace rnndescent
