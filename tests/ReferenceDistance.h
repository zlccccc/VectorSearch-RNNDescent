#pragma once

namespace tests {

inline double scalar_l2(const float *a, const float *b, int dim) {
    double sum = 0;
    for (int k = 0; k < dim; ++k) {
        const double delta = static_cast<double>(a[k]) - b[k];
        sum += delta * delta;
    }
    return sum;
}

} // namespace tests
