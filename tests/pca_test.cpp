#include "solution/rnndescent/PCAProjection.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace tests {
namespace {

TEST(PCAProjectionTest, ProjectsOntoTheLeadingCenteredAxis) {
    constexpr int rows = 4;
    for (const int dimension : {2, 5}) {
        SCOPED_TRACE(dimension);
        std::vector<float> base(rows * dimension, 7.0f);
        for (int i = 0; i < rows; ++i) {
            base[i * dimension] = i < 2 ? 6.0f : 14.0f;
            base[i * dimension + 1] = (i % 2 == 0) ? -4.0f : -2.0f;
        }
        rnndescent::PCAProjection pca(rnndescent::FloatMatrixView::from_vector(base, dimension), 1);

        std::vector<float> query(3 * dimension, 7.0f), projected(3);
        query[0] = 10.0f;
        query[dimension] = 12.0f;
        query[2 * dimension] = 8.0f;
        pca.apply(rnndescent::FloatMatrixView::from_vector(query, dimension),
                  rnndescent::MutableFloatMatrixView::from_vector(projected, 1));

        // Eigenvectors can have either sign.
        EXPECT_NEAR(projected[0], 0.0f, 1e-5f);
        EXPECT_NEAR(std::abs(projected[1]), 2.0f, 1e-5f);
        EXPECT_NEAR(projected[1], -projected[2], 1e-5f);
    }
}

} // namespace
} // namespace tests
