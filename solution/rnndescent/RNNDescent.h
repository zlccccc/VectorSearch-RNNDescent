#pragma once

#include "build/GraphBuilder.h"
#include "build/GraphConfig.h"
#include "GraphPruning.h"
#include "build/GraphReorder.h"
#include "search/GraphSearcher.h"
#include "build/GraphTypes.h"
#include "Logger.h"

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace rnndescent {

struct RNNDescent {
    using BuildConfig = config::BuildConfig;
    using SearchConfig = config::SearchConfig;
    using KNNGraph = ::rnndescent::KNNGraph;
    using PerfStats = search::PerfStats;

    static BuildConfig sanitize_build_config(BuildConfig config) { return config::sanitize_build_config(config); }
    static SearchConfig sanitize_search_config(SearchConfig config, int topk = 1, int max_points = 0) {
        return config::sanitize_search_config(config, topk, max_points);
    }

    static void gen_random(std::mt19937 &rng, std::vector<int> &addr, int n) {
        const int size = static_cast<int>(addr.size());
        if (size == 0)
            return;
        std::unordered_map<int, int> swapped;
        swapped.reserve(size * 2);
        for (int i = 0; i < size; ++i) {
            std::uniform_int_distribution<int> distribution(i, n - 1);
            const int j = distribution(rng);
            const auto it_i = swapped.find(i);
            const int value_i = it_i == swapped.end() ? i : it_i->second;
            const auto it_j = swapped.find(j);
            const int value_j = it_j == swapped.end() ? j : it_j->second;
            swapped[j] = value_i;
            addr[i] = value_j;
        }
    }

    RNNDescent(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config,
               const SearchConfig &search_config) {
        build(data_view, verbose, build_config, search_config);
    }

    ~RNNDescent() { reset(); }

    void build(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config,
               const SearchConfig &search_config) {
        data_view.validate("build data");
        const BuildConfig safe_build_config = sanitize_build_config(build_config);
        const SearchConfig safe_search_config = sanitize_search_config(search_config, 1, data_view.row_count());
        reset();

        if (verbose)
            Logger::info(verbose, "Parameters: S=%d, R=%d, T1=%d, T2=%d; Point=%d; numThreadsMax=%d\n",
                         safe_build_config.S, safe_build_config.R, safe_build_config.T1, safe_build_config.T2,
                         data_view.row_count(), safe_search_config.num_threads);

        GraphBuilder builder(data_view, verbose, safe_build_config, node_locks_);
        auto graph_result = builder.build();
#if defined(__GLIBC__)
        malloc_trim(0);
#endif

        GraphIndexStorage reorder_storage{search_from_ids, final_graph_neighbors, graph_distance_computer, neighbor_pool_};
        GraphReorder::materialize(data_view, verbose, safe_build_config, safe_search_config, std::move(graph_result),
                                  reorder_storage);

        search::Storage search_storage{search_from_ids, final_graph_neighbors, graph_distance_computer};
        searcher_ = std::make_unique<search::GraphSearcher>(search_storage, safe_search_config, safe_build_config.K0,
                                                              perf_stats);
    }

    void searchSingle(int threadid, int queryid, const CandidateReranker &reranker,
                      const SearchConfig &search_config, int max_degree, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(searcher_ != nullptr, "The index is not build yet.");
        searcher_->search_single(threadid, queryid, reranker, search_config, max_degree, result);
    }

    void flush_perf_stats() {
        if (searcher_)
            searcher_->flush_perf_stats();
    }

    void reset() {
        flush_perf_stats();
        searcher_.reset();
        Logger::line(true, "Resetting RNNDescent index.");
        std::vector<SelectedNeighborsContainerType>().swap(final_graph_neighbors);
        neighbor_pool_.reset();
        search_from_ids.clear();
        graph_distance_computer.reset();
        node_locks_.clear();
        perf_stats.reset();
    }

    std::vector<int> search_from_ids;
    std::vector<SelectedNeighborsContainerType> final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> graph_distance_computer;
    SelectedNeighborsContainerType::Pool neighbor_pool_;
    PerfStats perf_stats;

  private:
    std::unique_ptr<search::GraphSearcher> searcher_;
    std::vector<MutexWarpper> node_locks_;
};

} // namespace rnndescent
