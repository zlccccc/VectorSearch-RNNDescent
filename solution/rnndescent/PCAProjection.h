#pragma once

#include "common/Views.h"

#include <vector>

namespace rnndescent {

class PCAProjection {
  public:
    struct Config {
        bool enabled = false;
        int out_dim = 0;

        bool applies_to(int input_dimension) const { return enabled && out_dim > 0 && out_dim < input_dimension; }
    };

    PCAProjection(const FloatMatrixView &base, int output_dimension);

    int output_dimension() const { return output_dimension_; }
    void apply(const FloatMatrixView &input, const MutableFloatMatrixView &output) const;
    // The returned view is backed by the caller's buffer.
    FloatMatrixView project(const FloatMatrixView &input, std::vector<float> &buffer) const;
    // Reuses internal storage. The view is valid until the next query projection or destruction.
    FloatMatrixView project_queries(const FloatMatrixView &queries);

  private:
    int input_dimension_;
    int output_dimension_;
    std::vector<float> components_;
    std::vector<float> bias_;
    std::vector<float> query_buffer_;
};

} // namespace rnndescent
