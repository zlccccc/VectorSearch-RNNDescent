#pragma once

#include <omp.h>

#include <algorithm>
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

    explicit IndexRNNDescent(int d = 0) : verbose(false), input_dimension_(d) {}

    IndexRNNDescent(int d, bool verbose, const RNNDescent::BuildConfig &build_config, const RNNDescent::SearchConfig &search_config, const PCAConfig &pca_config)
        : verbose(verbose), input_dimension_(d), build_config(build_config), search_config(search_config), pca_config(pca_config) {}

    ~IndexRNNDescent() { reset(); }

    int dimension() const { return input_dimension_; }

  private:
    bool verbose;
    int input_dimension_ = 0;
    std::unique_ptr<CandidateReranker> reranker;
    std::unique_ptr<RNNDescent> rnndescent;
    RNNDescent::BuildConfig build_config;
    RNNDescent::SearchConfig search_config;
    PCAConfig pca_config;
    std::unique_ptr<PCAProjection> pca;

  public:

    void build(const FloatMatrixView &base) {
        RNNDESCENT_ASSERT_MSG(reranker == nullptr, "incremental add is not supported");
        base.validate("add data");
        RNNDESCENT_ASSERT_MSG(base.dimension() == dimension(), "add data dimension does not match index dimension");
        const int n = base.row_count();

        build_config = RNNDescent::sanitize_build_config(build_config);
        search_config = RNNDescent::sanitize_search_config(search_config);
        const auto graph_search_config = RNNDescent::sanitize_search_config(search_config, 1, n);
        const int candidate_count = graph_search_config.refine_max;
        SelectedDistanceComputerFactory::validate_quantization(search_config.graph_quantization);
        if (search_config.rerank_quantization && search_config.fp32_refine_max > 0 &&
            search_config.fp32_refine_max < candidate_count)
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
        prevtime = nowtime;
        Logger::info(verbose, "RNN Descent build done in %f ms\n", process_time);

        reranker = std::make_unique<CandidateReranker>(base, rnndescent->graph_distance_computer->dimension(), candidate_count,
                                                       search_config, rnndescent->search_from_ids.data());
    }

    void search(const FloatMatrixView &queries, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(reranker != nullptr, "index has not been built");
        RNNDESCENT_ASSERT_MSG(rnndescent != nullptr, "search graph has not been initialized");
        RNNDESCENT_ASSERT_MSG(rnndescent->graph_distance_computer != nullptr, "search graph has not been initialized");
        queries.validate("query batch");
        result.validate();
        RNNDESCENT_ASSERT_MSG(queries.dimension() == dimension(), "query dimension does not match index dimension");
        const int n = queries.row_count();
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now();
        auto nowtime = prevtime, starttime = prevtime;
        float process_time;
#endif

        reranker->set_queries(queries);
        const auto graph_queries = pca ? pca->project_queries(queries) : queries;
        rnndescent->graph_distance_computer->set_query(graph_queries.data_ptr(), n);

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        Logger::info(verbose, "search: PCA Process done in %f ms/item\n", process_time / n);
#endif

#pragma omp parallel for num_threads(search_config.num_threads) schedule(dynamic, 4)
        for (int queryid = 0; queryid < n; queryid++) {
            int threadid = omp_get_thread_num();
            rnndescent->searchSingle(threadid, queryid, *reranker, search_config, build_config.K0, result.slice(queryid));
        }

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        auto all_time = std::chrono::duration<float, std::milli>(nowtime - starttime).count();
        Logger::info(verbose, "search: RNNDescent Search Process done in %f ms/item; all = %f ms/item; internal point %f\n", process_time / n,
                     all_time / n, 1000. * n / all_time);
#endif
    }

    void reset() {
        rnndescent.reset();
        reranker.reset();
        pca.reset();
    }

    void flush_perf_stats() {
        if (rnndescent)
            rnndescent->flush_perf_stats();
    }
};

} // namespace rnndescent
