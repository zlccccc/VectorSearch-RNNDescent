#include "solution/solution.h"
#include <bits/stdc++.h>
using namespace std;

// int d = 512;
// int data_size = 1000000;

// int d = 1024;
// int data_size = 1000000;
// // int data_size = 100000;

// int d = 1536;
// int data_size = 1000000;
// int data_size = 300000;

int output_iter = 10;
int test_iter = 10000;
// int test_iter = 10;
// int test_iter = 1000;

void randvector(float *data, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        data[i] = rand() * 1. / RAND_MAX;
        // data[i] -= 0.5;
        sum += data[i] * data[i];
    }
    sum = sqrt(sum);
    for (int k = 0; k < d; k++) {
        data[k] = data[k] / sum;
    }
}

void solve(int data_size, int d) {
    Solution solution;
    // vector<uint8_t> test((long long)d * 24 * 100'0000);
    // // vector<uint8_t> test(20ll * 1024 * 1024 * 1024);  // 20G
    // fill(test.begin(), test.end(), 0x3f);
    puts("start to solve the problem");
    vector<float> dataset(data_size * d);
    srand(0);
    for (int i = 0; i < data_size; i++)
        randvector(dataset.data() + i * d, d);
    vector<float> query(test_iter * d);
    // #pragma omp parallel for
    for (int i = 0; i < test_iter; i++)
        randvector(query.data() + i * d, d);
    puts("start solve");
    vector<int> result(topk * test_iter);
    auto before_build = std::chrono::high_resolution_clock::now();
    solution.build(d, dataset);
    auto after_build = std::chrono::high_resolution_clock::now();
    std::cout << "build time: " << std::chrono::duration<double>(after_build - before_build).count() << std::endl;
    // solution.test(d, dataset);
    // for (int i = 0; i < test_iter; i++) {
    //     vector<float> current(query.begin() + i * d, query.begin() + (i + 1) * d);
    //     solution.search(current, result.data());
    // }
    for (int i = 0; i < 10; i++) {
        auto before_test = std::chrono::high_resolution_clock::now();
        solution.search(query, result.data());
        auto after_test = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<double, std::milli>(after_test - before_test).count() / test_iter;
        std::cout << "[" << i << "] average test time: " << t << " ms; " << 1000. / t << " offline score" << std::endl;
    }

    for (int i = 0; i < output_iter; i++) {
        for (int k = 0; k < topk; k++)
            std::cout << result[i * topk + k] << " ";
        puts(" <- result id");
        for (int k = 0; k < topk; k++)
            std::cout << solution.distances[i * topk + k] << " ";
        puts(" <- solution distance");
        for (int k = 0; k < topk; k++) {
            float real = 0;
            for (int j = 0; j < d; j++)
                real += (dataset[result[i * topk + k] * d + j] - query[i * d + j]) * (dataset[result[i * topk + k] * d + j] - query[i * d + j]);
            std::cout << real << " ";
        }
        puts(" <- real distance");
    }

    solution.index->reset(); // 概率输出不出来; 直接手动reset吧
}

int main(int argc, char **argv) {
    solve(1000000, 512);
    solve(1000000, 1024);
    solve(1000000, 1536);
    return 0;
}