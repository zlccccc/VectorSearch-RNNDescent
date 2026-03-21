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
    void search(const std::vector<float> &query, std::vector<int> &res, int topk);
    void reset();
    const std::vector<float> &distance_buffer() const { return distances_; }
    ~Solution() = default;

  private:
    void warmup(const std::vector<float> &base, int d, const WarmupConfig &warmup_config);
    std::unique_ptr<rnndescent::IndexRNNDescent> index_;
    std::vector<float> distances_;
};

#endif
