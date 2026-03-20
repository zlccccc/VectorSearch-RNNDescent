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

const int querysize = 10000;
const int topk = 10;
class Solution {
  public:
    void build(int d, const vector<float> &base);
    void search(const vector<float> &query, vector<int> &res);
    void test(int d, const vector<float> &base);
    void reset();
    std::unique_ptr<rnndescent::IndexRNNDescent> index;
    std::array<float, topk * querysize> distances{};
    ~Solution() = default;
};

#endif