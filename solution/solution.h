#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include "rnndescent/IndexRNNDescent.h"
#include <memory>
#include <vector>

class Solution {
  public:
    struct WarmupConfig {
        int topk = 10;
        int sample_count = 10000;
        int repeat_count = 1;
        int random_seed = 0;
    };

    void build(int d, const std::vector<float> &base, const WarmupConfig &warmup_config,
               const rnndescent::RNNDescent::BuildConfig &build_config,
               const rnndescent::RNNDescent::SearchConfig &search_config,
               const rnndescent::IndexRNNDescent::PCAConfig &pca_config = {});
    void search(const std::vector<float> &queries, std::vector<int> &res, int topk);
    void search(const rnndescent::FloatMatrixView &queries, const rnndescent::SearchResultView &result);
    void search_batch(const std::vector<float> &queries, std::vector<int> &res, int topk);
    // Apply to subsequent parallel single-query searches on the built index.
    void set_single_query_limits(int candidates_per_worker, int expansions_per_worker = 0);
    void reset();
    void flush_perf_stats() {
        if (index_)
            index_->flush_perf_stats();
    }
    const std::vector<float> &distance_buffer() const { return distances_; }
    ~Solution() = default;

  private:
    void warmup(const std::vector<float> &base, int d, const WarmupConfig &warmup_config);
    std::unique_ptr<rnndescent::IndexRNNDescent> index_;
    std::vector<float> distances_;
};

#endif
