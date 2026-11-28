#include "solution/rnndescent/search/CandidateReranker.h"
#include "solution/rnndescent/RNNDescent.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <omp.h>
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
            reranker.rerank(0, candidates, limit, topk);
            ASSERT_EQ(candidates.size(), static_cast<size_t>(topk));
            rnndescent::CandidateReranker::write_topk(candidates, {ids.data(), distances.data(), topk});
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
    search.single_query_threads = 3;
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
    graph.search_single_thread(0, 0, reranker, search,
                               rnndescent::SearchResultView::from_vectors(ids, distances, topk));
    std::sort(exact.evaluated.begin(), exact.evaluated.end());
    EXPECT_EQ(exact.evaluated, expected);
    expected.resize(topk);
    EXPECT_EQ(ids, expected);

    rnndescent::CandidateReranker fp32(rnndescent::FloatMatrixView::from_vector(base, dim), dim, rows, search);
    fp32.set_queries(rnndescent::FloatMatrixView::from_vector(query, dim));
    std::vector<int> parallel_ids(topk);
    std::vector<float> parallel_distances(topk);
    graph.search_single_thread(0, 0, fp32, search, {ids.data(), distances.data(), topk});
    graph.search_single_parallel(0, fp32, search, {parallel_ids.data(), parallel_distances.data(), topk});
    EXPECT_EQ(parallel_ids, ids);
    EXPECT_EQ(parallel_distances, distances);
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
    const rnndescent::FloatMatrixView query(base.data() + query_id * dim, 1, dim);
    graph.graph_distance_computer->set_query(projection.data() + query_id * graph_dim, 1);
    original.set_queries(query);
    bfs.set_queries(query);
    for (const auto [limit, topk] : {std::pair{32, 10}, std::pair{32, 40}, std::pair{0, 10}}) {
        SCOPED_TRACE(::testing::Message() << "FP32 limit=" << limit << ", TopK=" << topk);
        search.fp32_refine_max = limit;
        const auto collect = [&](const rnndescent::CandidateReranker &reranker) {
            std::vector<int> ids(topk);
            std::vector<float> distances(topk);
            graph.search_single_thread(0, 0, reranker, search,
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
    reranker.rerank(0, candidates, search.fp32_refine_max, 1, original_ids);
    ASSERT_EQ(candidates.size(), 1u);
    rnndescent::CandidateReranker::write_topk(candidates, {&id, &distance, 1});
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
    search.fp32_refine_max = 0;
    search.num_threads = 1;
    rnndescent::search::PerfStats perf;
    rnndescent::search::GraphSearcher searcher({neighbors, computer}, search, 4, perf);
    auto exact = std::make_unique<TrackedDistances>(distances.scores, 1);
    const auto &refined_ids = exact->evaluated;
    rnndescent::CandidateReranker reranker(std::move(exact), 1);
    int id;
    float distance;
    searcher.search_single_thread(0, 0, search, reranker, {&id, &distance, 1});
    EXPECT_EQ(refined_ids, (std::vector<int>{0, 1, 2, 3, 8, 4}));
    EXPECT_EQ(std::find(distances.evaluated.begin(), distances.evaluated.end(), 9), distances.evaluated.end());
}

TEST(RefinementTest, ParallelSingleQueryCapsWorkersAndScoresDisjointEntryRanges) {
    constexpr int entry_count = 160;
    constexpr int requested_workers = 16;
    constexpr int fp32_limit = 13;
    struct EntryRecorder final : rnndescent::MyDistanceComputer {
        std::array<std::vector<int>, requested_workers> scored_ids;

        int dimension() const override { return 1; }
        uint32_t code_stride() const override { return 0; }
        void set_query(const float *, int) override {}
        float symmetric_dis(int, int) override { return 0; }
        float operator()(int, int id) override {
            scored_ids[omp_get_thread_num()].push_back(id);
            return static_cast<float>(id);
        }
    };

    auto recorder = std::make_unique<EntryRecorder>();
    auto &scored_ids = recorder->scored_ids;
    std::unique_ptr<rnndescent::MyDistanceComputer> computer = std::move(recorder);
    std::vector<rnndescent::SelectedNeighborsContainerType> neighbors(entry_count);
    rnndescent::RNNDescent::SearchConfig search;
    search.num_initialize = search.search_L = search.refine_max = entry_count;
    search.fp32_refine_max = fp32_limit;
    search.num_threads = 1;
    search.single_query_threads = requested_workers;
    // EntryRecorder indexes its buffers by OpenMP thread ID.
    search.persistent_query_workers = false;
    rnndescent::search::PerfStats perf;
    std::vector<int> original_ids(entry_count);
    std::iota(original_ids.rbegin(), original_ids.rend(), 0);
    rnndescent::search::GraphSearcher searcher({neighbors, computer, &original_ids}, search, 4, perf);
    auto fp32 = std::make_unique<EntryRecorder>();
    auto &fp32_ids = fp32->scored_ids;
    rnndescent::CandidateReranker exact(std::move(fp32), 1);
    int id;
    float distance;
    searcher.search_parallel(0, search, exact, {&id, &distance, 1});

    std::vector<int> scored;
    int active_workers = 0;
    for (const auto &ids : scored_ids) {
        active_workers += !ids.empty();
        EXPECT_EQ(ids.size() % 4, 0u);
        scored.insert(scored.end(), ids.begin(), ids.end());
    }
    EXPECT_EQ(active_workers, std::min(8, omp_get_thread_limit()));
    std::vector<int> expected(entry_count);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(scored, expected);

    const int local_limit = active_workers == 1 ? fp32_limit : 16;
    int best = entry_count;
    for (int worker = 0; worker < active_workers; ++worker) {
        const auto &entries = scored_ids[worker];
        std::vector<int> expected_fp32;
        for (int i = 0; i < std::min(local_limit, static_cast<int>(entries.size())); ++i)
            expected_fp32.push_back(original_ids[entries[i]]);
        std::sort(fp32_ids[worker].begin(), fp32_ids[worker].end());
        std::sort(expected_fp32.begin(), expected_fp32.end());
        EXPECT_EQ(fp32_ids[worker], expected_fp32);
        best = std::min(best, expected_fp32.front());
    }
    EXPECT_EQ(id, best);
    EXPECT_FLOAT_EQ(distance, best);

    if (active_workers > 1) {
        auto local_fp32 = std::make_unique<EntryRecorder>();
        const auto &local_fp32_ids = local_fp32->scored_ids;
        auto coarse = std::make_unique<EntryRecorder>();
        const auto &coarse_ids = coarse->scored_ids;
        rnndescent::CandidateReranker two_stage(std::move(local_fp32), 1, std::move(coarse));
        for (auto &ids : scored_ids)
            ids.clear();
        search.single_query_limits.candidates_per_worker = 1;
        searcher.search_parallel(0, search, two_stage, {&id, &distance, 1});
        for (int worker = 0; worker < active_workers; ++worker) {
            const size_t local_width = (entry_count - 1) / active_workers + 1;
            EXPECT_EQ(coarse_ids[worker].size(), std::min(local_width, scored_ids[worker].size()));
            // Coarse ranking prefers the opposite end of the graph shortlist after mapping IDs.
            ASSERT_FALSE(coarse_ids[worker].empty());
            const int best_coarse = *std::min_element(coarse_ids[worker].begin(), coarse_ids[worker].end());
            EXPECT_EQ(local_fp32_ids[worker], (std::vector<int>{best_coarse}));
        }
        EXPECT_EQ(id, 0);
        EXPECT_FLOAT_EQ(distance, 0);

        using Neighbors = rnndescent::SelectedNeighborsContainerType;
        Neighbors::Pool pool;
        pool.initialize(Neighbors::cache_lane_width, {0, 4, 8});
        int first_edges[]{entry_count - 1, entry_count - 1, entry_count - 1, entry_count - 1};
        int second_edges[]{entry_count - 2, entry_count - 2, entry_count - 2, entry_count - 2};
        expected.resize(entry_count);
        std::iota(expected.begin(), expected.end(), 0);
        neighbors[0] = Neighbors(pool, first_edges, 4, expected);
        neighbors[1] = Neighbors(pool, second_edges, 4, expected);

        search.single_query_limits.candidates_per_worker = 1;
        search.fp32_refine_max = 0;
        for (const int max_expansions : {1, 0}) {
            for (auto &ids : scored_ids)
                ids.clear();
            for (auto &ids : fp32_ids)
                ids.clear();
            search.single_query_limits.expansions_per_worker = max_expansions;
            searcher.search_parallel(0, search, exact, {&id, &distance, 1});

            const auto &first_worker = scored_ids[0];
            EXPECT_EQ(std::count(first_worker.begin(), first_worker.end(), entry_count - 1), 4);
            EXPECT_EQ(std::count(first_worker.begin(), first_worker.end(), entry_count - 2), max_expansions ? 0 : 4);
            const auto reranked = std::accumulate(fp32_ids.begin(), fp32_ids.end(), size_t{0},
                                                  [](size_t total, const auto &ids) { return total + ids.size(); });
            EXPECT_EQ(reranked, active_workers);
        }
    }
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
    packed->score_candidates(0, candidates.data(), candidates.size());

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
    reranker.rerank(0, candidates, keep, 1, row_order.data());
    ASSERT_EQ(candidates.size(), 1u);
    rnndescent::CandidateReranker::write_topk(candidates, {&id, &distance, 1});
    EXPECT_TRUE(reranker.uses_graph_ordered_codes());
    EXPECT_EQ(id, nearest);
    EXPECT_FLOAT_EQ(distance, nearest_distance);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, PackedUInt4RefinementTest, ::testing::Values(65, 960));
#endif

} // namespace
} // namespace tests
