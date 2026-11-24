#include "solution/rnndescent/build/GraphPruning.h"
#include "solution/rnndescent/build/GraphTypes.h"

#include <array>
#include <gtest/gtest.h>

namespace tests {
namespace {

TEST(GraphPruningTest, MergesEdgeTagsIndependentlyOfCandidateOrder) {
    using rnndescent::XNeighbor;
    const std::array<XNeighbor, 6> candidates{{
        {3, 1, true}, {2, 1, false}, {3, 1, false}, {7, 5, false}, {7, 5, false}, {9, 6, true},
    }};
    std::array<int, candidates.size()> order{0, 1, 2, 3, 4, 5};
    do {
        std::vector<XNeighbor> row;
        for (int i : order)
            row.push_back(candidates[i]);
        rnndescent::detail::sort_and_merge_neighbors(row);

        ASSERT_EQ(row.size(), 4u);
        EXPECT_EQ(row[0].id(), 2);
        EXPECT_EQ(row[1].id(), 3);
        EXPECT_EQ(row[2].id(), 7);
        EXPECT_EQ(row[3].id(), 9);
        EXPECT_FLOAT_EQ(row[0].distance, 1);
        EXPECT_FLOAT_EQ(row[1].distance, 1);
        EXPECT_FLOAT_EQ(row[2].distance, 5);
        EXPECT_FLOAT_EQ(row[3].distance, 6);
        EXPECT_FALSE(row[0].flag());
        EXPECT_TRUE(row[1].flag());
        EXPECT_FALSE(row[2].flag());
        EXPECT_TRUE(row[3].flag());
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST(GraphPruningTest, ReplacesThreeHopRedundancyAndRetainsTheRoute) {
    rnndescent::KNNGraph graph(128);
    const auto set_row = [&](int node, std::initializer_list<int> ids) {
        float distance = 0;
        for (int id : ids)
            graph[node].emplace_back(id, ++distance, false);
    };
    set_row(0, {1, 2, 3, 4, 5, 6});
    set_row(1, {3, 7});
    set_row(2, {5}); // A two-hop route alone does not cause replacement.
    set_row(3, {4});
    set_row(100, {1, 2, 3, 4, 5, 6});

    const int degree = 4;
    auto serial = graph;
    EXPECT_EQ(rnndescent::detail::prune_local_three_hop(serial, degree, 1), 2u);
    EXPECT_EQ(rnndescent::detail::prune_local_three_hop(graph, degree, 2), 2u);
    const auto retained = [&](const auto &row) {
        std::vector<int> ids;
        for (int i = 0; i < std::min<int>(degree, row.size()); ++i)
            ids.push_back(row[i].id());
        return ids;
    };
    EXPECT_EQ(retained(graph[0]), (std::vector<int>{1, 2, 3, 5}));
    EXPECT_EQ(retained(graph[1]), (std::vector<int>{3, 7}));
    EXPECT_EQ(retained(graph[2]), (std::vector<int>{5}));
    EXPECT_EQ(retained(graph[3]), (std::vector<int>{4}));
    EXPECT_EQ(retained(graph[100]), retained(graph[0]));
    for (size_t i = 0; i < graph.size(); ++i)
        EXPECT_EQ(retained(graph[i]), retained(serial[i]));
}

} // namespace
} // namespace tests
