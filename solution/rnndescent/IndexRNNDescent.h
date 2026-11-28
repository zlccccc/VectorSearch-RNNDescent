#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "common/Assertions.h"
#include "search/CandidateReranker.h"
#include "common/Logger.h"
#include "PCAProjection.h"
#include "RNNDescent.h"
#include "common/Views.h"

namespace rnndescent {

struct IndexRNNDescent {
    using PCAConfig = PCAProjection::Config;

    IndexRNNDescent(int d, bool verbose, const RNNDescent::BuildConfig &build_config,
                    const RNNDescent::SearchConfig &search_config, const PCAConfig &pca_config)
        : verbose(verbose), input_dimension_(d), build_config(build_config),
          search_config(config::sanitize_search_config(search_config)), pca_config(pca_config) {}

    ~IndexRNNDescent() { reset(); }

    int dimension() const { return input_dimension_; }

  private:
    bool verbose;
    int input_dimension_;
    std::unique_ptr<CandidateReranker> reranker;
    std::unique_ptr<RNNDescent> rnndescent;
    RNNDescent::BuildConfig build_config;
    RNNDescent::SearchConfig search_config;
    PCAConfig pca_config;
    std::unique_ptr<PCAProjection> pca;
    search::QueryPhaseProfile query_profile_;

  public:

    void set_single_query_limits(int candidates_per_worker, int expansions_per_worker) {
        if (candidates_per_worker < 0 || expansions_per_worker < 0)
            throw std::runtime_error("single-query limits must be non-negative");
        search_config.single_query_limits = {candidates_per_worker, expansions_per_worker};
    }

    void build(const FloatMatrixView &base) {
        RNNDESCENT_ASSERT_MSG(reranker == nullptr, "incremental add is not supported");
        base.validate("add data");
        RNNDESCENT_ASSERT_MSG(base.dimension() == dimension(), "add data dimension does not match index dimension");
        const int n = base.row_count();

        const int candidate_count = config::sanitize_search_config(search_config, 1, n).refine_max;
        SelectedDistanceComputerFactory::validate_quantization(search_config.graph_quantization);
        if (search_config.uses_coarse_shortlist(candidate_count))
            SelectedDistanceComputerFactory::validate_quantization(*search_config.rerank_quantization);

        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = std::chrono::high_resolution_clock::now();
        float process_time;
        {
            std::vector<float> projected;
            auto graph_base = base;
            if (pca_config.applies_to(base.dimension())) {
                Logger::line(verbose, "Train PCA Matrix");
                pca = std::make_unique<PCAProjection>(base, pca_config.out_dim);
                graph_base = pca->project(base, projected);
                Logger::line(verbose, "End Calc PCA Matrix");
                nowtime = std::chrono::high_resolution_clock::now();
                process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
                Logger::info(verbose, "PCA Process done in %f ms\n", process_time);
            }
            rnndescent = std::make_unique<RNNDescent>(graph_base, verbose, build_config, search_config);
        }

        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        Logger::info(verbose, "RNN Descent build done in %f ms\n", process_time);

        const int graph_dimension = pca ? pca->output_dimension() : base.dimension();
        reranker = std::make_unique<CandidateReranker>(base, graph_dimension, candidate_count, search_config,
                                                       rnndescent->search_from_ids.data());
    }

    void search(const FloatMatrixView &queries, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(reranker != nullptr, "index has not been built");
        RNNDESCENT_ASSERT_MSG(rnndescent != nullptr, "search graph has not been initialized");
        RNNDESCENT_ASSERT_MSG(rnndescent->graph_distance_computer != nullptr, "search graph has not been initialized");
        queries.validate("query batch");
        result.validate();
        RNNDESCENT_ASSERT_MSG(queries.dimension() == dimension(), "query dimension does not match index dimension");
        const int n = queries.row_count();
        const int workers = search_config.num_threads;
        auto sample = query_profile_.measure();

        reranker->set_queries(queries);
        sample.stage(search::QueryStage::RerankPrepare);
        const auto graph_queries = pca ? pca->project_queries(queries, workers) : queries;
        sample.stage(search::QueryStage::Projection);
        rnndescent->graph_distance_computer->set_query(graph_queries.data_ptr(), n);
        sample.stage(search::QueryStage::GraphEncode);

        const auto ordered_queries = n > 1 && search_config.reorder_query_batch && pca
                                         ? pca->query_order(graph_queries)
                                         : std::vector<int>{};
        const int *const query_order = ordered_queries.empty() ? nullptr : ordered_queries.data();
        sample.stage(search::QueryStage::Ordering);
        if (n == 1 && search_config.single_query_threads > 1) {
            rnndescent->search_single_parallel(0, *reranker, search_config, result);
        } else {
            rnndescent->search_batch(n, query_order, *reranker, search_config, result);
        }

        sample.finish(n);
    }

    void reset() {
        query_profile_.flush();
        rnndescent.reset();
        reranker.reset();
        pca.reset();
    }

    void flush_perf_stats() {
        query_profile_.flush();
        if (rnndescent)
            rnndescent->flush_perf_stats();
    }
};

} // namespace rnndescent
