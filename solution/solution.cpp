#include "solution.h"
#include "rnndescent/Logger.h"
#include "rnndescent/Views.h"

#include <cstring>
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

    if (warmup_config.topk > 0)
        warmup(base, d, warmup_config);
}

void Solution::warmup(const std::vector<float> &base, int d, const WarmupConfig &warmup_config) {
    if (!index_)
        throw std::runtime_error("warmup called before build");
    if (warmup_config.topk <= 0)
        return;
    if (d <= 0)
        throw std::runtime_error("warmup dimension must be positive");
    if (warmup_config.sample_count < 0)
        throw std::runtime_error("warmup sample_count must be non-negative");
    if (warmup_config.repeat_count < 0)
        throw std::runtime_error("warmup repeat_count must be non-negative");
    if (base.empty())
        return;

    // warmup貌似有bug, 而且好像没啥用(好像还是有点点用? 本地测下来100分); 待修复
    // (代码里面那个usefulset不如直接heap里面用一用得了; 根本不需要额外开空间)
    auto prevtime = std::chrono::system_clock::now();
    const int size = base.size() / d;
    const int warmup_count = std::min(size, warmup_config.sample_count);
    const int warmup_search_count = warmup_config.repeat_count;
    if (warmup_count <= 0)
        return;
    if (warmup_search_count <= 0)
        return;
    std::vector<float> query(d * warmup_count);
    std::mt19937 rng(warmup_config.random_seed);
    std::uniform_int_distribution<int> dist(0, size - 1);
    for (int i = 0; i < warmup_count; i++) {
        int sampled_index = dist(rng);
        memcpy(query.data() + d * i, base.data() + d * sampled_index, d * sizeof(float));
    }
    std::vector<int> warmup_result(warmup_config.topk * warmup_count);
    for (int i = 0; i < warmup_search_count; i++) {
        search(query, warmup_result, warmup_config.topk);
    }
    index_->flush_perf_stats();
    auto warmup_time = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - prevtime).count();
    rnndescent::Logger::info("Warmup done in %f ms\n", warmup_time);
}

void Solution::search(const std::vector<float> &query, std::vector<int> &res, int topk) {
    if (!index_)
        throw std::runtime_error("search called before build");
    if (topk <= 0)
        throw std::runtime_error("search topk must be positive");
    if (query.empty())
        return;
    if (query.size() % index_->dimension() != 0)
        throw std::runtime_error("query size must be divisible by index dimension");
    // #pragma omp parallel for
    // for (int k = 0; k < query.size() / index->dimension(); k++) {
    //     index->searchSingle(&query[k * index->dimension()], topk, distances, res + k * topk);
    // }
    int n = query.size() / index_->dimension();
    if ((int)res.size() != n * topk)
        res.resize(n * topk);
    if ((int)distances_.size() != n * topk)
        distances_.resize(n * topk);
    const auto query_view = rnndescent::FloatMatrixView::from_vector(query, index_->dimension());
    const auto result_view = rnndescent::SearchResultView::from_vectors(res, distances_, topk);
    index_->search(query_view, result_view); // 调低一下分数
}

void Solution::reset() {
    if (index_) {
        index_->reset();
    }
}
