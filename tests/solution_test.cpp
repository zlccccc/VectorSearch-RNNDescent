#include "ReferenceDistance.h"
#include "solution/solution.h"

#include <algorithm>
#include <array>
#include <gmock/gmock.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tests {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(SolutionTest, BuildsSearchesAndResetsWithPcaAndCache) {
    constexpr int dim = 65;
    constexpr int rows = 32;
    constexpr int topk = 5;
    constexpr int selected[] = {21, 2, 30, 5, 18};
    std::vector<float> base(rows * dim);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < dim; ++k)
            base[i * dim + k] = float((41 * i + 13 * k + 5 * i * k + i * i) % 101 - 50) / 8;

    std::vector<float> queries;
    for (int id : selected) {
        queries.insert(queries.end(), base.begin() + id * dim, base.begin() + (id + 1) * dim);
        queries.back() += 0.25f;
    }
    Solution solution;
    std::vector<int> ids;
    EXPECT_THAT([&] { solution.search(queries, ids, topk); },
                ThrowsMessage<std::runtime_error>(HasSubstr("search called before build")));

    Solution::WarmupConfig warmup;
    warmup.topk = 0;
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = 2;
    build.T2 = 2;
    build.S = 12;
    build.R = 16;
    build.K0 = 8;
    build.num_threads = 2;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_threads = 2;
    search.beam_size = 2;
    // Refine every point so the expected IDs are independent of graph construction order.
    search.num_initialize = rows;
    search.search_L = rows;
    search.refine_max = rows;

    for (const auto &[pca, cache] : {std::pair{false, false}, std::pair{false, true},
                                     std::pair{true, false}, std::pair{true, true}}) {
        SCOPED_TRACE(::testing::Message() << "PCA=" << pca << ", cache=" << cache);
        build.neighbor_pool_size_limit_bytes = cache ? 1 << 20 : 0;
        solution.build(dim, base, warmup, build, search, {pca, pca ? 17 : 0});
        EXPECT_THAT([&] { solution.search({1, 2}, ids, topk); },
                    ThrowsMessage<std::runtime_error>(HasSubstr("query size must be divisible")));
        solution.search(queries, ids, topk);
        const auto &distances = solution.distance_buffer();
        constexpr size_t queries_count = sizeof(selected) / sizeof(selected[0]);
        ASSERT_EQ(ids.size(), queries_count * topk);
        ASSERT_EQ(distances.size(), ids.size());
        for (size_t q = 0; q < queries_count; ++q) {
            std::array<std::pair<double, int>, rows> exact{};
            for (int i = 0; i < rows; ++i)
                exact[i] = {scalar_l2(base.data() + i * dim, queries.data() + q * dim, dim), i};
            std::sort(exact.begin(), exact.end());
            for (int i = 0; i < topk; ++i) {
                const size_t offset = q * topk + i;
                EXPECT_EQ(ids[offset], exact[i].second) << "query " << q << ", rank " << i;
                EXPECT_NEAR(distances[offset], exact[i].first, 1e-3) << "query " << q << ", rank " << i;
            }
        }
        solution.reset();
        EXPECT_THAT([&] { solution.search(queries, ids, topk); },
                    ThrowsMessage<std::runtime_error>(HasSubstr("index has not been built")));
    }
}

TEST(SolutionTest, SeparatelyBuiltIndexesKeepTheirOwnNeighbors) {
    constexpr int first_dim = 65;
    constexpr int second_dim = 33;
    constexpr int rows = 128;
    constexpr int topk = 5;
    std::mt19937 rng(19);
    auto make_base = [&](int dim) {
        std::vector<float> base(rows * dim);
        for (float &coordinate : base)
            coordinate = static_cast<float>(rng() % 4096) / 2048.0f - 1.0f;
        return base;
    };
    const auto first_base = make_base(first_dim);
    const auto second_base = make_base(second_dim);
    const std::vector<float> first_queries(first_base.begin(), first_base.begin() + 16 * first_dim);
    const std::vector<float> second_queries(second_base.end() - 16 * second_dim, second_base.end());

    Solution::WarmupConfig warmup;
    warmup.topk = 0;
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = 2;
    build.T2 = 2;
    build.S = 16;
    build.R = 32;
    build.K0 = 12;
    build.num_threads = 1;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = 16;
    search.search_L = 12;
    search.refine_max = 8;
    search.beam_size = 2;
    search.num_threads = 2;

    for (const bool first_cached : {false, true}) {
        SCOPED_TRACE(first_cached);
        Solution first, second;
        build.neighbor_pool_size_limit_bytes = first_cached ? 1 << 20 : 0;
        first.build(first_dim, first_base, warmup, build, search);
        std::vector<int> first_ids, second_ids, ids;
        first.search(first_queries, first_ids, topk);
        const auto first_distances = first.distance_buffer();

        build.neighbor_pool_size_limit_bytes = first_cached ? 0 : 1 << 20;
        second.build(second_dim, second_base, warmup, build, search);
        second.search(second_queries, second_ids, topk);
        const auto second_distances = second.distance_buffer();

        first.search(first_queries, ids, topk);
        EXPECT_EQ(ids, first_ids);
        EXPECT_EQ(first.distance_buffer(), first_distances);

        first.reset();
        second.search(second_queries, ids, topk);
        EXPECT_EQ(ids, second_ids);
        EXPECT_EQ(second.distance_buffer(), second_distances);
    }
}

TEST(SolutionTest, TwoStageRefinementReturnsOriginalFp32Distances) {
    constexpr int rows = 96, dim = 65;
    std::mt19937 rng(53);
    std::vector<float> base(rows * dim);
    for (auto &value : base)
        value = float(rng() % 4096) / 2048 - 1;
    const std::vector<float> queries(base.begin() + 17 * dim, base.begin() + 19 * dim);
    Solution::WarmupConfig warmup;
    warmup.topk = 0;
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = build.T2 = 1;
    build.S = build.K0 = 16;
    build.R = 32;
    build.num_threads = 1;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = search.search_L = search.refine_max = rows;
    search.num_threads = 2;
    Solution solution;
    std::vector<int> ids;
    for (bool pca : {false, true}) {
        for (bool cache : {false, true}) {
            for (int limit : {0, 32}) {
                SCOPED_TRACE(::testing::Message() << "PCA=" << pca << ", cache=" << cache << ", FP32 limit=" << limit);
                build.neighbor_pool_size_limit_bytes = cache ? 1 << 20 : 0;
                search.fp32_refine_max = limit;
                solution.build(dim, base, warmup, build, search, {pca, pca ? 64 : 0});
                // Top K above 32 must expand the FP32 shortlist instead of truncating the output.
                for (int topk : {5, 40}) {
                    solution.search(queries, ids, topk);
                    const auto &distances = solution.distance_buffer();
                    ASSERT_EQ(ids.size(), 2 * topk);
                    ASSERT_EQ(distances.size(), ids.size());
                    for (int q = 0; q < 2; ++q) {
                        EXPECT_EQ(ids[q * topk], 17 + q);
                        std::vector<int> seen;
                        for (int k = 0; k < topk; ++k) {
                            const int id = ids[q * topk + k];
                            ASSERT_GE(id, 0);
                            ASSERT_LT(id, rows);
                            EXPECT_EQ(std::count(seen.begin(), seen.end(), id), 0);
                            seen.push_back(id);
                            const double expected = scalar_l2(queries.data() + q * dim, base.data() + id * dim, dim);
                            EXPECT_NEAR(distances[q * topk + k], expected, 1e-5 * expected + 1e-5);
                            if (k > 0)
                                EXPECT_LE(distances[q * topk + k - 1], distances[q * topk + k]);
                        }
                    }
                }
                solution.reset();
            }
        }
    }
}

TEST(RecallTest, ImprovesWithSearchBudgetAndMatchesWithCache) {
    constexpr int dim = 65;
    constexpr int rows = 768;
    constexpr int query_count = 128;
    constexpr int topk = 10;
    std::mt19937 rng(42);
    auto make_vectors = [&](int count) {
        std::vector<float> vectors(count * dim);
        for (int i = 0; i < count; ++i) {
            for (int k = 0; k < dim; ++k) {
                const int sample = static_cast<int>(rng() % 4096) - 2048;
                // Give the first 17 dimensions most of the signal.
                vectors[i * dim + k] = sample * (k < 17 ? 1.0f / 2048 : 1.0f / 16384);
            }
        }
        return vectors;
    };
    const auto base = make_vectors(rows);
    const auto queries = make_vectors(query_count);
    std::array<std::array<int, topk>, query_count> truth{};
    for (int q = 0; q < query_count; ++q) {
        std::vector<std::pair<double, int>> exact;
        exact.reserve(rows);
        for (int i = 0; i < rows; ++i)
            exact.emplace_back(scalar_l2(base.data() + i * dim, queries.data() + q * dim, dim), i);
        std::partial_sort(exact.begin(), exact.begin() + topk, exact.end());
        for (int k = 0; k < topk; ++k)
            truth[q][k] = exact[k].second;
    }

    Solution::WarmupConfig warmup;
    warmup.topk = 0;
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = 2;
    build.T2 = 4;
    build.S = 24;
    build.R = 48;
    build.K0 = 16;
    // The same graph is built for both cache settings.
    build.num_threads = 1;

    auto run = [&](int entry_points, int search_l, int refine, bool cache, std::vector<int> &ids, double &recall) {
        SCOPED_TRACE(::testing::Message() << "search_L=" << search_l << ", cache=" << cache);
        rnndescent::RNNDescent::SearchConfig search;
        search.num_threads = 2;
        search.beam_size = 2;
        search.num_initialize = entry_points;
        search.search_L = search_l;
        search.refine_max = refine;
        build.neighbor_pool_size_limit_bytes = cache ? 1 << 20 : 0;

        Solution solution;
        solution.build(dim, base, warmup, build, search);
        solution.search(queries, ids, topk);
        const auto &distances = solution.distance_buffer();
        ASSERT_EQ(ids.size(), query_count * topk);
        ASSERT_EQ(distances.size(), ids.size());
        int hits = 0;
        for (int q = 0; q < query_count; ++q) {
            const auto first = ids.begin() + q * topk;
            for (int k = 0; k < topk; ++k) {
                const int id = first[k];
                const float distance = distances[q * topk + k];
                SCOPED_TRACE(::testing::Message() << "query=" << q << ", rank=" << k);
                ASSERT_GE(id, 0);
                ASSERT_LT(id, rows);
                EXPECT_TRUE(std::find(first, first + k, id) == first + k) << "duplicate ID " << id;
                if (k > 0)
                    EXPECT_LE(distances[q * topk + k - 1], distance);
                const double exact = scalar_l2(base.data() + id * dim, queries.data() + q * dim, dim);
                EXPECT_NEAR(distance, exact, 1e-5 * exact + 1e-6);
                hits += std::find(truth[q].begin(), truth[q].end(), id) != truth[q].end();
            }
        }
        recall = static_cast<double>(hits) / (query_count * topk);
    };

    std::vector<int> low_ids, cached_ids, high_ids;
    double low = 0, cached = 0, high = 0;
    ASSERT_NO_FATAL_FAILURE(run(16, 12, 10, false, low_ids, low));
    ASSERT_NO_FATAL_FAILURE(run(16, 12, 10, true, cached_ids, cached));
    ASSERT_NO_FATAL_FAILURE(run(96, 64, 48, false, high_ids, high));
    RecordProperty("recall_at_10_low", std::to_string(low));
    RecordProperty("recall_at_10_cached", std::to_string(cached));
    RecordProperty("recall_at_10_high", std::to_string(high));
    EXPECT_GE(low, 0.60);
    EXPECT_LE(low, 0.93) << "low-budget search should stay below saturation";
    EXPECT_EQ(low_ids, cached_ids);
    EXPECT_GE(high, 0.95);
    EXPECT_GE(high - low, 0.08);
}

} // namespace
} // namespace tests
