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

TEST(PCAProjectionTest, OrdersTwoProjectedQueries) {
    constexpr int input_dim = 5, output_dim = 4;
    const std::vector<float> base(input_dim * input_dim, 0.0f);
    const rnndescent::PCAProjection pca(rnndescent::FloatMatrixView::from_vector(base, input_dim), output_dim);
    const std::vector<float> projected{2, 2, 2, 2, -1, -1, -1, -1};
    EXPECT_EQ(pca.query_order(rnndescent::FloatMatrixView::from_vector(projected, output_dim)),
              (std::vector<int>{1, 0}));
    EXPECT_TRUE(pca.query_order(rnndescent::FloatMatrixView::from_buffer(projected.data(), 1, output_dim)).empty());
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

TEST(PCAProjectionTest, HadamardMixesPcaAxesAndPreservesProjectedDistances) {
    for (const int output_dim : {6, 64, 96}) {
        SCOPED_TRACE(output_dim);
        const int axes = output_dim + 1;
        const int rows = 2 * axes;
        // Exercise both the Gram and covariance training paths.
        const int input_dim = output_dim == 6 ? rows + 1 : axes + 1;
        constexpr float center = 0.25f;
        std::vector<float> base(static_cast<size_t>(rows) * input_dim, center);
        for (int axis = 0; axis < axes; ++axis) {
            const float offset = (axes - axis) * 0.25f;
            base[static_cast<size_t>(2 * axis) * input_dim + axis] += offset;
            base[static_cast<size_t>(2 * axis + 1) * input_dim + axis] -= offset;
        }
        rnndescent::PCAProjection pca(rnndescent::FloatMatrixView::from_vector(base, input_dim), output_dim);

        std::vector<float> queries(4 * input_dim, center);
        queries[input_dim] += 1;
        queries[2 * input_dim + output_dim - 1] += 1;
        queries[3 * input_dim + output_dim] += 1;
        const auto projected = pca.project_queries(rnndescent::FloatMatrixView::from_vector(queries, input_dim));
        int first_block = 1;
        while (first_block <= output_dim / 2)
            first_block *= 2;
        const float mixed_axis = 1.0f / std::sqrt(static_cast<float>(first_block));
        for (int i = 0; i < output_dim; ++i) {
            EXPECT_NEAR(projected.row_ptr(0)[i], 0.0f, 1e-5f);
            EXPECT_NEAR(std::abs(projected.row_ptr(1)[i]), i < first_block ? mixed_axis : 0.0f, 1e-5f);
            EXPECT_NEAR(projected.row_ptr(3)[i], 0.0f, 1e-5f);
        }
        const auto squared_distance = [&](int a, int b) {
            float sum = 0;
            for (int i = 0; i < output_dim; ++i) {
                const float diff = projected.row_ptr(a)[i] - projected.row_ptr(b)[i];
                sum += diff * diff;
            }
            return sum;
        };
        EXPECT_NEAR(squared_distance(0, 1), 1.0f, 1e-5f);
        EXPECT_NEAR(squared_distance(0, 2), 1.0f, 1e-5f);
        EXPECT_NEAR(squared_distance(1, 2), 2.0f, 1e-5f);
    }
}

} // namespace
} // namespace tests
