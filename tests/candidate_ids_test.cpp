#include "solution/rnndescent/search/CandidateBitmap.h"
#include "solution/rnndescent/search/NeighborScoreFilter.h"

#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace tests {
namespace {

TEST(CandidateBitmapTest, MarksIdsAcrossWordBoundaries) {
    rnndescent::CandidateBitmap ids;
    ids.initialize(129);
    for (int id : {0, 1, 63, 64, 65, 127, 128})
        ASSERT_TRUE(ids.insert(id)) << id;
    for (int id : {0, 1, 63, 64, 65, 127, 128})
        EXPECT_FALSE(ids.insert(id)) << id;
}

TEST(CandidateBitmapTest, ClearsMarksBetweenQueries) {
    rnndescent::CandidateBitmap ids;
    ids.initialize(70);
    for (int id : {0, 63, 64, 69})
        ASSERT_TRUE(ids.insert(id)) << id;

    ids.clear();
    for (int id : {0, 63, 64, 69}) {
        EXPECT_TRUE(ids.insert(id)) << id;
        EXPECT_FALSE(ids.insert(id)) << id;
    }
}

TEST(NeighborScoreFilterTest, MatchesScalarOrderAndMarksOnlyPassingIds) {
    constexpr int point_count = 129;
    constexpr int score_count = 37;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 8> pattern = {-infinity, 0.0f, 1.0f, nan, 0.5f, infinity, 2.0f, -0.0f};
    std::array<float, score_count> scores;
    std::array<int, score_count> ids;
    for (int i = 0; i < score_count; ++i) {
        scores[i] = pattern[i % pattern.size()];
        ids[i] = (i * 17) % 29 + 59;
    }

    for (float cutoff : {-infinity, 0.0f, 1.0f, infinity, nan}) {
        for (int count = 0; count <= score_count; ++count) {
            rnndescent::CandidateBitmap visited;
            visited.initialize(point_count);
            ASSERT_TRUE(visited.insert(59));
            ASSERT_TRUE(visited.insert(64));
            std::array<bool, point_count> seen{};
            seen[59] = seen[64] = true;
            std::array<rnndescent::SingleNeighbor, score_count + 1> actual;
            actual[0] = {-1, -1.0f};
            auto *const begin = actual.data() + 1;
            const auto *end = rnndescent::detail::append_neighbors_below_cutoff(
                scores.data(), ids.data(), count, cutoff, visited, begin);

            int expected_count = 0;
            for (int i = 0; i < count; ++i) {
                if (scores[i] < cutoff && !seen[ids[i]]) {
                    ASSERT_LT(begin + expected_count, end);
                    EXPECT_EQ(begin[expected_count].id, ids[i]);
                    EXPECT_EQ(begin[expected_count].distance, scores[i]);
                    seen[ids[i]] = true;
                    ++expected_count;
                }
            }
            EXPECT_EQ(end, begin + expected_count);
            EXPECT_EQ(actual[0].id, -1);
            for (int id = 0; id < point_count; ++id)
                EXPECT_EQ(visited.insert(id), !seen[id]) << id;
        }
    }
}

} // namespace
} // namespace tests
