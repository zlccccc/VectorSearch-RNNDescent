#pragma once

#include <omp.h>

#include <chrono>
#include <memory>
#include <vector>

#include "Assertions.h"
#include "Logger.h"
#include "PCAProjection.h"
#include "RNNDescent.h"
#include "Views.h"

namespace rnndescent {

struct IndexRNNDescent {
    // Current implementation only supports L2 distance.
    struct PCAConfig {
        bool enabled = false;
        int out_dim = 0;
    };

    explicit IndexRNNDescent(int d = 0) : verbose(false), input_dimension_(d) {}

    IndexRNNDescent(int d, bool verbose, const RNNDescent::BuildConfig &build_config, const RNNDescent::SearchConfig &search_config, const PCAConfig &pca_config)
        : verbose(verbose), build_config(build_config), search_config(search_config), pca_config(pca_config), input_dimension_(d) {}

    ~IndexRNNDescent() { reset(); }

    int dimension() const { return refine_distance_computer ? refine_distance_computer->dimension() : input_dimension_; }

  private:
    bool verbose;
    int input_dimension_ = 0;
    std::unique_ptr<MyDistanceComputer> refine_distance_computer;
    std::unique_ptr<RNNDescent> rnndescent;
    RNNDescent::BuildConfig build_config;
    RNNDescent::SearchConfig search_config;
    PCAConfig pca_config;
    std::unique_ptr<PCAProjection> pca;
    std::vector<float> pca_query_buffer;
    void rebuild_graph_index(const FloatMatrixView &base_view) {
        rnndescent = std::make_unique<RNNDescent>(base_view, verbose, build_config, search_config);
    }

  public:

    void build(const FloatMatrixView &base) {
        RNNDESCENT_ASSERT_MSG(refine_distance_computer == nullptr, "incremental add is not supported");
        base.validate("add data");
        RNNDESCENT_ASSERT_MSG(base.dimension() == dimension(), "add data dimension does not match index dimension");
        const int n = base.row_count();
        const float *x = base.data_ptr();

        build_config = RNNDescent::sanitize_build_config(build_config);
        search_config = RNNDescent::sanitize_search_config(search_config);

        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = std::chrono::high_resolution_clock::now();
        float process_time;
        const bool use_pca = pca_config.enabled && pca_config.out_dim > 0 && pca_config.out_dim < dimension();
        if (use_pca) {
            Logger::line(verbose, "Train PCA Matrix");
            const int projected_dim = pca_config.out_dim;
            pca = std::make_unique<PCAProjection>(base, projected_dim);
            std::vector<float> projected(static_cast<size_t>(n) * projected_dim);
            pca->apply(base, MutableFloatMatrixView::from_vector(projected, projected_dim));
            Logger::line(verbose, "End Calc PCA Matrix");
            nowtime = std::chrono::high_resolution_clock::now();
            process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
            Logger::info(verbose, "PCA Process done in %f ms\n", process_time);
            rebuild_graph_index(FloatMatrixView::from_vector(projected, projected_dim));
        } else {
            pca.reset();
            rebuild_graph_index(FloatMatrixView::from_buffer(x, n, input_dimension_));
        }

        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        Logger::info(verbose, "RNN Descent build done in %f ms\n", process_time);

        RNNDESCENT_ASSERT_MSG(refine_distance_computer == nullptr, "refine distance computer has already been initialized");
        refine_distance_computer = SelectedDistanceComputerFactory::create_refine_search(x, n, dimension());
    }

    void search(const FloatMatrixView &queries, const SearchResultView &result) {
        RNNDESCENT_ASSERT_MSG(refine_distance_computer != nullptr, "index has not been built");
        RNNDESCENT_ASSERT_MSG(rnndescent != nullptr, "search graph has not been initialized");
        RNNDESCENT_ASSERT_MSG(rnndescent->graph_distance_computer != nullptr, "search graph has not been initialized");
        queries.validate("query batch");
        result.validate();
        RNNDESCENT_ASSERT_MSG(queries.dimension() == dimension(), "query dimension does not match index dimension");
        const int n = queries.row_count();
        const float *x = queries.data_ptr();
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now();
        auto nowtime = prevtime, starttime = prevtime;
        float process_time;
#endif

        refine_distance_computer->set_query(x, n);
        if (pca) {
            const int projected_dim = pca->output_dimension();
            pca_query_buffer.resize(static_cast<size_t>(n) * projected_dim);
            pca->apply(queries, MutableFloatMatrixView::from_vector(pca_query_buffer, projected_dim));
            rnndescent->graph_distance_computer->set_query(pca_query_buffer.data(), n);
        } else {
            rnndescent->graph_distance_computer->set_query(x, n);
        }

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        Logger::info(verbose, "search: PCA Process done in %f ms/item\n", process_time / n);
#endif

#pragma omp parallel for num_threads(search_config.num_threads) schedule(dynamic, 4)
        for (int queryid = 0; queryid < n; queryid++) {
            int threadid = omp_get_thread_num();
            rnndescent->searchSingle(threadid, queryid, *refine_distance_computer, search_config, build_config.K0, result.slice(queryid));
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
        refine_distance_computer.reset();
        pca.reset();
        pca_query_buffer.clear();
    }

    void flush_perf_stats() {
        if (rnndescent)
            rnndescent->flush_perf_stats();
    }
};

} // namespace rnndescent
