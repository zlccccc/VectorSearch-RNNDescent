#include "solution.h"
#include "rnndescent/common/Logger.h"
#include "rnndescent/common/Views.h"

#include <algorithm>
#include <chrono>
#include <random>

void Solution::build(int d, const std::vector<float> &base, const WarmupConfig &warmup_config, const rnndescent::RNNDescent::BuildConfig &build_config,
                     const rnndescent::RNNDescent::SearchConfig &search_config, const rnndescent::IndexRNNDescent::PCAConfig &pca_config) {
    if (d <= 0)
        throw std::runtime_error("build dimension must be positive");
    if (base.empty())
        throw std::runtime_error("base vectors must not be empty");
    if (base.size() % d != 0)
        throw std::runtime_error("base size must be divisible by dimension");
    index_ = std::make_unique<rnndescent::IndexRNNDescent>(d, true, build_config, search_config, pca_config);
    const auto base_view = rnndescent::FloatMatrixView::from_vector(base, d);
    index_->build(base_view);
    warmup(base_view, warmup_config);
}

void Solution::warmup(const rnndescent::FloatMatrixView &base, const WarmupConfig &warmup_config) {
    if (warmup_config.topk <= 0)
        return;
    if (warmup_config.sample_count < 0)
        throw std::runtime_error("warmup sample_count must be non-negative");
    if (warmup_config.repeat_count < 0)
        throw std::runtime_error("warmup repeat_count must be non-negative");

    const int warmup_count = std::min(base.row_count(), warmup_config.sample_count);
    if (warmup_count == 0 || warmup_config.repeat_count == 0)
        return;

    // Warm up the search path with a reproducible sample of base vectors.
    const auto start = std::chrono::steady_clock::now();
    const int dimension = base.dimension();
    std::vector<float> query(static_cast<size_t>(warmup_count) * dimension);
    std::mt19937 rng(warmup_config.random_seed);
    std::uniform_int_distribution<int> dist(0, base.row_count() - 1);
    for (int i = 0; i < warmup_count; i++) {
        const auto *sample = base.row_ptr(dist(rng));
        std::copy_n(sample, dimension, query.data() + static_cast<size_t>(i) * dimension);
    }
    std::vector<int> warmup_result;
    for (int i = 0; i < warmup_config.repeat_count; i++)
        search(query, warmup_result, warmup_config.topk);
    index_->flush_perf_stats();
    const auto warmup_time = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    rnndescent::Logger::info("Warmup done in %f ms\n", warmup_time);
}

void Solution::search(const std::vector<float> &queries, std::vector<int> &res, int topk) {
    if (!index_)
        throw std::runtime_error("search called before build");
    if (topk <= 0)
        throw std::runtime_error("search topk must be positive");
    if (queries.empty())
        return;
    const int dimension = index_->dimension();
    if (queries.size() % dimension != 0)
        throw std::runtime_error("query size must be divisible by index dimension");
    const size_t rows = queries.size() / dimension;
    const size_t result_count = rows * topk;
    res.resize(result_count);
    distances_.resize(result_count);
    const rnndescent::FloatMatrixView query_view{queries.data(), static_cast<int>(rows), dimension};
    const auto result_view = rnndescent::SearchResultView::from_vectors(res, distances_, topk);
    index_->search(query_view, result_view);
}

void Solution::search(const rnndescent::FloatMatrixView &queries, const rnndescent::SearchResultView &result) {
    if (!index_)
        throw std::runtime_error("search called before build");
    index_->search(queries, result);
}

void Solution::set_single_query_limits(int candidates_per_worker, int expansions_per_worker) {
    if (!index_)
        throw std::runtime_error("set_single_query_limits called before build");
    index_->set_single_query_limits(candidates_per_worker, expansions_per_worker);
}

void Solution::search_batch(const std::vector<float> &queries, std::vector<int> &res, int topk) {
    search(queries, res, topk);
}

void Solution::reset() {
    if (index_) {
        index_->reset();
    }
}
