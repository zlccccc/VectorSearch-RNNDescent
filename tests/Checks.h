#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace tests {

inline void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

inline void require_near(double actual, double expected, double tolerance, const std::string &message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template <class Function>
void require_error(Function operation, const std::string &expected_message) {
    try {
        operation();
    } catch (const std::runtime_error &error) {
        require(std::string(error.what()).find(expected_message) != std::string::npos,
                "unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected error: " + expected_message);
}

inline double scalar_l2(const float *a, const float *b, int dim) {
    double sum = 0;
    for (int k = 0; k < dim; ++k) {
        const double delta = static_cast<double>(a[k]) - b[k];
        sum += delta * delta;
    }
    return sum;
}

void test_simd();
void test_cache();
void test_benchmark();
void test_solution();
void test_recall();

} // namespace tests
