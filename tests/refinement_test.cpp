#include "solution/rnndescent/search/CandidateReranker.h"
#include "solution/rnndescent/RNNDescent.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace tests {
namespace {

struct TrackedDistances final : rnndescent::MyDistanceComputer {
    std::vector<float> scores;
    std::vector<int> evaluated;
    int dim;

    TrackedDistances(std::vector<float> scores, int dim) : scores(std::move(scores)), dim(dim) {}
    int row_count() const override { return static_cast<int>(scores.size()); }
    int dimension() const override { return dim; }
    uint32_t code_stride() const override { return 0; }
    void set_query(const float *, int) override {}
    float symmetric_dis(int, int) override { return 0; }
    float operator()(int, int id) override {
        evaluated.push_back(id);
        return scores.at(id);
    }
    void distances_batch_4(int q, int a, int b, int c, int d, float &da, float &db, float &dc, float &dd) override {
        da = (*this)(q, a);
        db = (*this)(q, b);
        dc = (*this)(q, c);
        dd = (*this)(q, d);
    }
};

TEST(RefinementTest, SelectsInOriginalSpaceAndLimitsFp32Work) {
    constexpr int rows = 95, graph_dim = 17, original_dim = 65;
    std::vector<float> coarse_scores(rows), exact_scores(rows);
    for (int i = 0; i < rows; ++i) {
        coarse_scores[i] = i * 37 % 96;
        exact_scores[i] = 100 + i;
    }
    // This true nearest neighbor is 32nd in the quantized ranking.
    exact_scores[19] = 0;
    ASSERT_EQ(coarse_scores[19], 31);
    auto coarse_distance = std::make_unique<TrackedDistances>(coarse_scores, original_dim);
    auto exact_distance = std::make_unique<TrackedDistances>(exact_scores, original_dim);
    auto &coarse = *coarse_distance;
    auto &exact = *exact_distance;
    rnndescent::CandidateReranker reranker(std::move(exact_distance), graph_dim, std::move(coarse_distance));

    for (const auto [limit, topk] : {std::pair{32, 10}, std::pair{7, 10}, std::pair{32, 40}, std::pair{0, 10}}) {
        SCOPED_TRACE(::testing::Message() << "FP32 limit=" << limit << ", TopK=" << topk);
        const int expected_count = limit == 0 ? rows : std::max(limit, topk);
        std::vector<int> expected(rows);
        std::iota(expected.begin(), expected.end(), 0);
        if (expected_count < rows) {
            std::sort(expected.begin(), expected.end(), [&](int a, int b) { return coarse_scores[a] < coarse_scores[b]; });
            expected.resize(expected_count);
        }
        std::vector<int> expected_evaluated = expected;
        std::sort(expected_evaluated.begin(), expected_evaluated.end());
        std::sort(expected.begin(), expected.end(), [&](int a, int b) { return exact_scores[a] < exact_scores[b]; });
        expected.resize(topk);
        for (int repeat = 0; repeat < 2; ++repeat) {
            coarse.evaluated.clear();
            exact.evaluated.clear();
            std::vector<rnndescent::SingleNeighbor> candidates;
            for (int id = rows - 1; id >= 0; --id)
                candidates.emplace_back(id, static_cast<float>(rows - 1 - id));
            std::vector<int> ids(topk);
            std::vector<float> distances(topk);
            reranker.rerank(0, candidates, limit, rnndescent::SearchResultView::from_vectors(ids, distances, topk));
            EXPECT_EQ(ids, expected);
            for (int k = 0; k < topk; ++k)
                EXPECT_EQ(distances[k], exact_scores[ids[k]]);
            std::sort(exact.evaluated.begin(), exact.evaluated.end());
            EXPECT_EQ(exact.evaluated, expected_evaluated);
            EXPECT_EQ(coarse.evaluated.size(), expected_count < rows ? rows : 0);
        }
    }
}

TEST(RefinementTest, ReusesOriginalDimensionalGraphScores) {
    constexpr int rows = 96, dim = 65, keep = 32, topk = 10;
    std::vector<float> base(rows * dim), query(dim);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < dim; ++k)
            base[i * dim + k] = float((i * 41 + k * 13 + i * k * 5) % 1009) / 1009;
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = build.T2 = 1;
    build.S = build.K0 = 16;
    build.R = 32;
    build.num_threads = 1;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = search.search_L = search.refine_max = rows;
    search.num_threads = 1;
    rnndescent::RNNDescent graph(rnndescent::FloatMatrixView::from_vector(base, dim), false, build, search);
    graph.graph_distance_computer->set_query(query.data(), 1);
    auto coarse = rnndescent::SelectedDistanceComputerFactory::create_quantized_refine(base.data(), rows, dim);
    coarse->set_query(query.data(), 1);
    std::vector<int> expected(rows);
    std::vector<float> scores(rows);
    std::iota(expected.begin(), expected.end(), 0);
    for (int i = 0; i < rows; ++i)
        scores[i] = (*coarse)(0, i);
    std::sort(expected.begin(), expected.end(), [&](int a, int b) { return scores[a] < scores[b]; });
    ASSERT_LT(scores[expected[keep - 1]], scores[expected[keep]]) << "shortlist boundary must not tie";
    expected.resize(keep);
    std::sort(expected.begin(), expected.end());
    std::vector<float> exact_scores(rows);
    std::iota(exact_scores.begin(), exact_scores.end(), 0.0f);
    auto exact_distance = std::make_unique<TrackedDistances>(exact_scores, dim);
    auto &exact = *exact_distance;
    rnndescent::CandidateReranker reranker(std::move(exact_distance), dim);
    std::vector<int> ids(topk);
    std::vector<float> distances(topk);
    graph.searchSingle(0, 0, reranker, search, build.K0, rnndescent::SearchResultView::from_vectors(ids, distances, topk));
    std::sort(exact.evaluated.begin(), exact.evaluated.end());
    EXPECT_EQ(exact.evaluated, expected);
    expected.resize(topk);
    EXPECT_EQ(ids, expected);
}

TEST(RefinementTest, BfsOrderedCodesReturnTheSameOriginalIdsAndDistances) {
    constexpr int rows = 96, dim = 65, graph_dim = 17;
    std::vector<float> base(rows * dim), projection(rows * graph_dim);
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < dim; ++k)
            base[i * dim + k] = float((i * 41 + k * 13 + i * k * 5) % 1009) / 1009;
        std::copy_n(base.data() + i * dim, graph_dim, projection.data() + i * graph_dim);
    }
    rnndescent::RNNDescent::BuildConfig build;
    build.T1 = build.T2 = 1;
    build.S = build.K0 = 16;
    build.R = 32;
    build.num_threads = 1;
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = rows;
    search.search_L = 64;
    search.refine_max = 61;
    search.fp32_refine_max = 32;
    search.num_threads = 1;
    rnndescent::RNNDescent graph(rnndescent::FloatMatrixView::from_vector(projection, graph_dim), false, build, search);
    ASSERT_FALSE(std::is_sorted(graph.search_from_ids.begin(), graph.search_from_ids.end()));
    const auto base_view = rnndescent::FloatMatrixView::from_vector(base, dim);
    rnndescent::CandidateReranker original(base_view, graph_dim, search.refine_max, search);
    rnndescent::CandidateReranker bfs(base_view, graph_dim, search.refine_max, search, graph.search_from_ids.data());

    constexpr int query_id = 19;
    const auto query = rnndescent::FloatMatrixView::from_buffer(base.data() + query_id * dim, 1, dim);
    graph.graph_distance_computer->set_query(projection.data() + query_id * graph_dim, 1);
    original.set_queries(query);
    bfs.set_queries(query);
    for (const auto [limit, topk] : {std::pair{32, 10}, std::pair{32, 40}, std::pair{0, 10}}) {
        SCOPED_TRACE(::testing::Message() << "FP32 limit=" << limit << ", TopK=" << topk);
        search.fp32_refine_max = limit;
        const auto collect = [&](const rnndescent::CandidateReranker &reranker) {
            std::vector<int> ids(topk);
            std::vector<float> distances(topk);
            graph.searchSingle(0, 0, reranker, search, build.K0,
                               rnndescent::SearchResultView::from_vectors(ids, distances, topk));
            return std::make_pair(std::move(ids), std::move(distances));
        };
        const auto expected = collect(original);
        EXPECT_EQ(expected.first.front(), query_id);
        EXPECT_EQ(collect(bfs), expected);
    }
}

TEST(RefinementTest, DirectFp32UsesEveryGraphCandidate) {
    const std::vector<float> base{0, 0, 1, 0, 2, 0, 3, 0};
    const std::vector<float> query{0, 0};
    const int original_ids[]{3, 2, 1, 0};
    rnndescent::RNNDescent::SearchConfig search;
    search.fp32_refine_max = 1;
    search.rerank_quantization.reset();
    rnndescent::CandidateReranker reranker(rnndescent::FloatMatrixView::from_vector(base, 2), 1, 4, search, original_ids);
    reranker.set_queries(rnndescent::FloatMatrixView::from_vector(query, 2));
    std::vector<rnndescent::SingleNeighbor> candidates{{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    int id = -1;
    float distance = -1;
    reranker.rerank(0, candidates, search.fp32_refine_max, {&id, &distance, 1}, original_ids);
    EXPECT_EQ(candidates.size(), 4);
    EXPECT_EQ(id, 0);
    EXPECT_EQ(distance, 0);
}

TEST(RefinementTest, RetainsDiscoveredCandidatesBeyondExpansionWidth) {
    auto tracked = std::make_unique<TrackedDistances>(std::vector<float>{0, 1, 2, 3, 10, 11, 12, 13, 4, 0.5f}, 1);
    auto &distances = *tracked;
    std::unique_ptr<rnndescent::MyDistanceComputer> computer = std::move(tracked);
    using Neighbors = rnndescent::SelectedNeighborsContainerType;
    Neighbors::Pool pool;
    pool.initialize(Neighbors::cache_lane_width, {0, 4, 8});
    std::vector<Neighbors> neighbors(computer->row_count());
    int edges_from_entry[]{8, 8, 8, 8};
    int edges_from_tail[]{9, 9, 9, 9};
    const std::vector<int> original_to_bfs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    neighbors[0] = Neighbors(pool, edges_from_entry, 4, original_to_bfs);
    neighbors[8] = Neighbors(pool, edges_from_tail, 4, original_to_bfs);

    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = search.search_L = 4;
    search.refine_max = 6;
    search.num_threads = 1;
    rnndescent::search::PerfStats perf;
    rnndescent::search::GraphSearcher searcher({neighbors, computer}, search, 4, perf);
    auto &candidates = searcher.search_candidates(0, 0, search, 4, 1);
    std::vector<int> ids;
    for (const auto &candidate : candidates)
        ids.push_back(candidate.id);
    EXPECT_EQ(ids, (std::vector<int>{0, 1, 2, 3, 8, 4}));
    EXPECT_EQ(std::find(distances.evaluated.begin(), distances.evaluated.end(), 9), distances.evaluated.end());
}

#if defined(RNNDESCENT_HAS_UINT4_KERNEL)
class PackedUInt4RefinementTest : public ::testing::TestWithParam<int> {};

TEST_P(PackedUInt4RefinementTest, ScoresBfsRowsAndCandidateTail) {
    using Factory = rnndescent::SelectedDistanceComputerFactory;
    constexpr int rows = 46, keep = 16;
    const int dim = GetParam();
    std::vector<float> base(rows * dim), query(dim);
    std::vector<int> row_order(rows);
    for (int i = 0; i < rows / 2; ++i) {
        for (int j = 0; j < dim; ++j) {
            const float value = (i + 1) * ((j + j / 32 + i) % 16 - 5);
            base[i * dim + j] = value;
            base[(rows - 1 - i) * dim + j] = -value;
        }
    }
    for (int j = 0; j < dim; ++j)
        query[j] = (j * 13) % 255 - 127;
    for (int i = 0; i < rows; ++i)
        row_order[i] = (i * 7 + 1) % rows;

    // Every paired row cancels in the mean. Row values lie exactly on their
    // 4-bit grid and query values fit exactly in a signed byte.
    const auto base_view = rnndescent::FloatMatrixView::from_vector(base, dim);
    const auto query_view = rnndescent::FloatMatrixView::from_vector(query, dim);
    auto packed = Factory::create_quantized_refine(base.data(), rows, dim, row_order.data(), rnndescent::QuantizedFormat::UInt4);
    packed->set_query(query.data(), 1);
    std::vector<rnndescent::SingleNeighbor> candidates;
    for (int i = 0; i < rows; ++i)
        candidates.emplace_back(i, 0);
    packed->score_candidates(0, candidates);

    int nearest = -1;
    float nearest_distance = std::numeric_limits<float>::max();
    for (const auto &candidate : candidates) {
        const int original = row_order[candidate.id];
        const auto *row = base_view.row_ptr(original);
        float score = 0, exact_distance = 0;
        for (int j = 0; j < dim; ++j) {
            score += row[j] * row[j] - 2 * query[j] * row[j];
            const float diff = row[j] - query[j];
            exact_distance += diff * diff;
        }
        EXPECT_FLOAT_EQ(candidate.distance, score);
        if (exact_distance < nearest_distance) {
            nearest = original;
            nearest_distance = exact_distance;
        }
    }

    rnndescent::RNNDescent::SearchConfig search;
    search.fp32_refine_max = keep;
    search.rerank_quantization = rnndescent::QuantizedFormat::UInt4;
    rnndescent::CandidateReranker reranker(base_view, 64, rows, search, row_order.data());
    reranker.set_queries(query_view);
    int id = -1;
    float distance = 0;
    reranker.rerank(0, candidates, keep, {&id, &distance, 1}, row_order.data());
    EXPECT_TRUE(reranker.uses_graph_ordered_codes());
    EXPECT_EQ(candidates.size(), keep);
    EXPECT_EQ(id, nearest);
    EXPECT_FLOAT_EQ(distance, nearest_distance);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, PackedUInt4RefinementTest, ::testing::Values(65, 960));
#endif

} // namespace
} // namespace tests
