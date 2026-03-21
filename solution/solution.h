#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include "rnndescent/IndexRNNDescent.h"
#include <algorithm>
#include <array>
#include <faiss/Index.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class Solution {
  public:
    void build(int d, const vector<float> &base, int warmup_topk,
               const rnndescent::RNNDescent::BuildConfig &build_config,
               const rnndescent::RNNDescent::SearchConfig &search_config,
               const rnndescent::IndexRNNDescent::PCAConfig &pca_config = {});
    void warmup(const vector<float> &base, int d, int warmup_topk);
    void search(const vector<float> &query, vector<int> &res, int topk);
    void test(int d, const vector<float> &base, int topk);
    void reset();
    std::unique_ptr<rnndescent::IndexRNNDescent> index;
    std::vector<float> distances;
    ~Solution() = default;
};

#endif