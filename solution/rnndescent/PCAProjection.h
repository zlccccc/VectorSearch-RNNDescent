#pragma once

#include "Views.h"

#include <vector>

namespace rnndescent {

class PCAProjection {
  public:
    PCAProjection(const FloatMatrixView &base, int output_dimension);

    int output_dimension() const { return output_dimension_; }
    void apply(const FloatMatrixView &input, const MutableFloatMatrixView &output) const;

  private:
    int input_dimension_;
    int output_dimension_;
    std::vector<float> components_;
    std::vector<float> bias_;
};

} // namespace rnndescent
