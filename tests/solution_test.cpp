#include "Checks.h"
#include "solution/solution.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace tests {

void test_solution() {
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
    require_error([&] { solution.search(queries, ids, topk); }, "search called before build");

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
        build.neighbor_pool_size_limit_bytes = cache ? 1 << 20 : 0;
        solution.build(dim, base, warmup, build, search, {pca, pca ? 17 : 0});
        require_error([&] { solution.search({1, 2}, ids, topk); }, "query size must be divisible");
        solution.search(queries, ids, topk);
        const auto &distances = solution.distance_buffer();
        constexpr size_t queries_count = sizeof(selected) / sizeof(selected[0]);
        require(ids.size() == queries_count * topk && distances.size() == ids.size(), "result buffer shape");
        for (size_t q = 0; q < queries_count; ++q) {
            std::array<std::pair<double, int>, rows> exact{};
            for (int i = 0; i < rows; ++i)
                exact[i] = {scalar_l2(base.data() + i * dim, queries.data() + q * dim, dim), i};
            std::sort(exact.begin(), exact.end());
            for (int i = 0; i < topk; ++i) {
                const size_t offset = q * topk + i;
                require(ids[offset] == exact[i].second, "top-k original row IDs after graph reorder");
                require_near(distances[offset], exact[i].first, 1e-3, "rerank uses the original vectors");
            }
        }
        solution.reset();
        require_error([&] { solution.search(queries, ids, topk); }, "index has not been built");
    }
}

void test_recall() {
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

    auto run = [&](int entry_points, int search_l, int refine, bool cache, std::vector<int> &ids) {
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
        require(ids.size() == query_count * topk && distances.size() == ids.size(), "limited search result shape");
        int hits = 0;
        for (int q = 0; q < query_count; ++q) {
            const auto first = ids.begin() + q * topk;
            for (int k = 0; k < topk; ++k) {
                const int id = first[k];
                const float distance = distances[q * topk + k];
                require(id >= 0 && id < rows, "limited search returned an invalid row ID");
                require(std::find(first, first + k, id) == first + k, "limited search returned a duplicate row ID");
                require(k == 0 || distances[q * topk + k - 1] <= distance, "limited search results are not sorted");
                const double exact = scalar_l2(base.data() + id * dim, queries.data() + q * dim, dim);
                require_near(distance, exact, 1e-5 * exact + 1e-6, "limited search rerank distance");
                hits += std::find(truth[q].begin(), truth[q].end(), id) != truth[q].end();
            }
        }
        return static_cast<double>(hits) / (query_count * topk);
    };

    std::vector<int> low_ids, cached_ids, high_ids;
    const double low = run(16, 12, 10, false, low_ids);
    const double cached = run(16, 12, 10, true, cached_ids);
    const double high = run(96, 64, 48, false, high_ids);
    std::cout << "Recall@10: low=" << low << " cached=" << cached << " high=" << high << '\n';
    require(low >= 0.60 && low <= 0.93, "low-budget recall must remain useful and below saturation");
    require(low_ids == cached_ids, "changing neighbor caching changed low-budget query results");
    require(high >= 0.95 && high - low >= 0.08, "increasing search budget must recover more true neighbors");
}

} // namespace tests
