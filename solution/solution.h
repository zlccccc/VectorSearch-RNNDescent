#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include "rnndescent/IndexRNNDescent.h"
#include <algorithm>
#include <faiss/Index.h>
#include <iostream>
#include <vector>

using namespace std;

const int querysize = 10000;
const int topk = 10;
class Solution {
  public:
    void build(int d, const vector<float> &base);
    void search(const vector<float> &query, int *res);
    void test(int d, const vector<float> &base);
    rnndescent::IndexRNNDescent *index = nullptr;
    float distances[topk * querysize];
    ~Solution() {
        if (index != nullptr) {
            delete index;
            index = nullptr;
        }
    }
};

#endif