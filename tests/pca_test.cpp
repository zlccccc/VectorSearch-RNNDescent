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

        const auto query_view = rnndescent::FloatMatrixView::from_vector(query, dimension);
        const auto batch = pca.project_queries(query_view);
        ASSERT_EQ(batch.row_count(), 3);
        ASSERT_EQ(batch.dimension(), 1);
        for (int i = 0; i < batch.row_count(); ++i)
            EXPECT_FLOAT_EQ(batch.row_ptr(i)[0], projected[i]);

        const auto smaller_batch = pca.project_queries(rnndescent::FloatMatrixView::from_buffer(query.data() + dimension, 1, dimension));
        ASSERT_EQ(smaller_batch.row_count(), 1);
        EXPECT_FLOAT_EQ(smaller_batch.row_ptr(0)[0], projected[1]);
    }
}

TEST(PCAProjectionTest, SampledTrainingPreservesTheLeadingAxisAcrossBatches) {
    for (const int dimension : {2, 17}) {
        SCOPED_TRACE(dimension);
        const int rows = dimension * 1000 + 1;
        std::vector<float> base(static_cast<size_t>(rows) * dimension, 3.0f);
        for (int i = 0; i < rows; ++i)
            base[static_cast<size_t>(i) * dimension] = static_cast<float>(i % 31 - 15);
        rnndescent::PCAProjection pca(rnndescent::FloatMatrixView::from_vector(base, dimension), 1);
        std::vector<float> queries(2 * dimension, 3.0f), projected(2);
        queries[0] = 0.0f;
        queries[dimension] = 2.0f;
        pca.apply(rnndescent::FloatMatrixView::from_vector(queries, dimension),
                  rnndescent::MutableFloatMatrixView::from_vector(projected, 1));
        EXPECT_NEAR(std::abs(projected[1] - projected[0]), 2.0f, 1e-4f);
    }
}

} // namespace
} // namespace tests
