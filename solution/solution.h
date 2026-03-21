#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include "rnndescent/IndexRNNDescent.h"
#include <memory>
#include <vector>

class Solution {
  public:
    void build(int d, const std::vector<float> &base, int warmup_topk,
               const rnndescent::RNNDescent::BuildConfig &build_config,
               const rnndescent::RNNDescent::SearchConfig &search_config,
               const rnndescent::IndexRNNDescent::PCAConfig &pca_config = {});
    void warmup(const std::vector<float> &base, int d, int warmup_topk);
    void search(const std::vector<float> &query, std::vector<int> &res, int topk);
    void reset();
    const std::vector<float> &distance_buffer() const { return distances_; }
    ~Solution() = default;

  private:
    std::unique_ptr<rnndescent::IndexRNNDescent> index_;
    std::vector<float> distances_;
};

#endif
