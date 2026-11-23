#include "solution/rnndescent/search/CandidateBitmap.h"
#include "solution/rnndescent/search/NeighborScoreFilter.h"

#include <algorithm>
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

TEST(NeighborScoreFilterTest, CollectsPassingScoresInOrderAcrossSpans) {
    constexpr int score_count = 68;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 8> pattern = {-infinity, 0.0f, 1.0f, nan, 0.5f, infinity, 2.0f, -0.0f};
    std::array<float, score_count> scores;
    std::array<int, score_count + 2> ids;
    ids.fill(-1);
    const auto id_slot = [](std::size_t position) { return position + (position >= 4) + (position >= 28); };
    for (int i = 0; i < score_count; ++i) {
        scores[i] = pattern[i % pattern.size()];
        ids[id_slot(i)] = (i * 17) % 29 + 59;
    }

    for (float cutoff : {-infinity, 0.0f, 1.0f, infinity, nan}) {
        for (std::size_t count = 0; count <= scores.size(); count += 4) {
            const auto first_end = std::min<std::size_t>(4, count);
            const auto second_end = std::min<std::size_t>(28, count);
            const rnndescent::detail::NeighborIdSpan spans[] = {
                {nullptr, 0},
                {ids.data(), first_end},
                {nullptr, first_end},
                {ids.data() + id_slot(4), second_end},
                {ids.data() + id_slot(28), count},
                {nullptr, count},
            };
            std::array<uint32_t, rnndescent::detail::score_mask_count(score_count)> masks;
            masks.fill(~uint32_t{0});
            std::array<rnndescent::SingleNeighbor, score_count + 1> actual;
            actual[0] = {-1, -1.0f};
            auto *const begin = actual.data() + 1;
            rnndescent::detail::make_neighbor_score_masks(scores.data(), count, cutoff, masks.data());
            const auto *end = rnndescent::detail::collect_neighbors_from_masks(
                scores.data(), count, masks.data(), spans, begin);

            int expected_count = 0;
            for (std::size_t i = 0; i < count; ++i) {
                if (scores[i] < cutoff) {
                    ASSERT_LT(begin + expected_count, end);
                    EXPECT_EQ(begin[expected_count].id, ids[id_slot(i)]);
                    EXPECT_EQ(begin[expected_count].distance, scores[i]);
                    ++expected_count;
                }
            }
            EXPECT_EQ(end, begin + expected_count);
            EXPECT_EQ(actual[0].id, -1);
        }
    }
}

} // namespace
} // namespace tests
