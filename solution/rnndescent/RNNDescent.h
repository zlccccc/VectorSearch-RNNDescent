#pragma once

#include "build/GraphBuilder.h"
#include "build/GraphConfig.h"
#include "build/GraphReorder.h"
#include "search/CandidateReranker.h"
#include "search/GraphSearcher.h"
#include "build/GraphTypes.h"
#include "common/Logger.h"

#include <memory>
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

    RNNDescent(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config,
               const SearchConfig &search_config) {
        build(data_view, verbose, build_config, search_config);
    }

    ~RNNDescent() { reset(); }

    void build(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config,
               const SearchConfig &search_config) {
        data_view.validate("build data");
        SelectedDistanceComputerFactory::validate_quantization(search_config.graph_quantization);
        const BuildConfig safe_build_config = config::sanitize_build_config(build_config);
        const SearchConfig safe_search_config = config::sanitize_search_config(search_config, 1, data_view.row_count());
        reset();

        Logger::info(verbose, "Parameters: S=%d, R=%d, T1=%d, T2=%d; Point=%d; numThreadsMax=%d\n",
                     safe_build_config.S, safe_build_config.R, safe_build_config.T1, safe_build_config.T2,
                     data_view.row_count(), safe_search_config.num_threads);

        GraphBuilder builder(data_view, verbose, safe_build_config);
        auto graph_result = builder.build();
#if defined(__GLIBC__)
        malloc_trim(0);
#endif

        GraphIndexStorage reorder_storage{search_from_ids, final_graph_neighbors, graph_distance_computer, neighbor_pool_};
        GraphReorder::materialize(data_view, verbose, safe_build_config, safe_search_config, std::move(graph_result),
                                  reorder_storage);

        search::Storage search_storage{final_graph_neighbors, graph_distance_computer, &search_from_ids};
        searcher_ = std::make_unique<search::GraphSearcher>(search_storage, safe_search_config, safe_build_config.K0,
                                                              perf_stats);
    }

    void search_batch(int query_count, const int *query_order, const CandidateReranker &reranker,
                      const SearchConfig &search_config, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(searcher_ != nullptr, "The index is not build yet.");
        searcher_->search_batch(query_count, query_order, search_config, reranker, result);
    }

    void search_single_thread(int threadid, int queryid, const CandidateReranker &reranker,
                              const SearchConfig &search_config, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(searcher_ != nullptr, "The index is not build yet.");
        searcher_->search_single_thread(threadid, queryid, search_config, reranker, result);
    }

    void search_single_parallel(int queryid, const CandidateReranker &reranker,
                                const SearchConfig &search_config, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(searcher_ != nullptr, "The index is not build yet.");
        searcher_->search_parallel(queryid, search_config, reranker, result);
    }

    void flush_perf_stats() {
        if (searcher_)
            searcher_->flush_perf_stats();
    }

    void reset() {
        flush_perf_stats();
        searcher_.reset();
        std::vector<SelectedNeighborsContainerType>().swap(final_graph_neighbors);
        neighbor_pool_.reset();
        search_from_ids.clear();
        graph_distance_computer.reset();
        perf_stats.reset();
    }

    std::vector<int> search_from_ids;
    std::vector<SelectedNeighborsContainerType> final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> graph_distance_computer;
    SelectedNeighborsContainerType::Pool neighbor_pool_;
    PerfStats perf_stats;

  private:
    std::unique_ptr<search::GraphSearcher> searcher_;
};

} // namespace rnndescent
