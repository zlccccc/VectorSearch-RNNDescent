#include "solution.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexNNDescent.h>

void Solution::build(int d, const vector<float> &base, int warmup_topk) {
    if (d <= 0)
        throw std::runtime_error("build dimension must be positive");
    if (base.empty())
        throw std::runtime_error("base vectors must not be empty");
    if (base.size() % d != 0)
        throw std::runtime_error("base size must be divisible by dimension");
    if (warmup_topk <= 0)
        throw std::runtime_error("warmup topk must be positive");
    index.reset();
    int size = base.size() / d;
    auto rnndescentindex = std::make_unique<rnndescent::IndexRNNDescent>(d, faiss::METRIC_L2);
    // auto rnndescentindex = new rnndescent::IndexRNNDescent(d, faiss::METRIC_INNER_PRODUCT);
    auto &built_index = *rnndescentindex;
    built_index.build_config.num_threads = omp_get_max_threads();
    built_index.search_config.num_threads = omp_get_max_threads();
    built_index.search_config.beam_size = 8;
    built_index.build_config.S = 196;
    built_index.build_config.R = 2048;
    built_index.build_config.T1 = 4;
    built_index.build_config.T2 = 15;
    built_index.verbose = true;

    if (d == 256) {
        built_index.search_config.num_initialize = 1024;
        built_index.search_config.search_L = 512;
        built_index.search_config.refine_max = 384;
        built_index.build_config.K0 = 96;
    } else if (d == 512) {
        built_index.search_config.num_initialize = 1024;
        built_index.search_config.search_L = 244; // topk
        built_index.search_config.refine_max = 64;
        built_index.build_config.K0 = 64;
        // rnndescentindex->rnndescent.search_L = 256;  // topk
        // rnndescentindex->rnndescent.K0 = 48;

        // rnndescentindex->rnndescent.search_L = 320;  // topk
        // rnndescentindex->rnndescent.K0 = 48;
    } else if (d == 1024) {
        built_index.search_config.num_initialize = 160;
        built_index.search_config.search_L = 116; // topk
        built_index.search_config.refine_max = 128;
        built_index.build_config.K0 = 32;
    } else if (d == 1536) {
        built_index.search_config.num_initialize = 512;
        built_index.search_config.search_L = 184; // topk
        built_index.search_config.refine_max = 128;
        built_index.build_config.K0 = 48;
    } else {
        throw std::runtime_error("Unsupported dimension");
    }

#ifdef INTERNAL_CLOCK_TEST
    built_index.build_config.S = 32;
    built_index.build_config.R = 256;
#endif

    built_index.train({base.data(), size, d});
    built_index.add({base.data(), size, d});
    index = std::move(rnndescentindex);

    // warmup貌似有bug, 而且好像没啥用(好像还是有点点用? 本地测下来100分); 待修复
    // (代码里面那个usefulset不如直接heap里面用一用得了; 根本不需要额外开空间)

    auto prevtime = std::chrono::system_clock::now();
    int warmup_count = 10000, warmup_search_count = 1;
    std::vector<float> query(d * warmup_count);
    for (int i = 0; i < warmup_count; i++) {
        int index = random() % size;
        memcpy(query.data() + d * i, base.data() + d * index, d * sizeof(float));
    }
    std::vector<int> warmup_result(warmup_topk * warmup_count);
    for (int i = 0; i < warmup_search_count; i++) {
        search(query, warmup_result, warmup_topk);
    }
    built_index.rnndescent.reset_time();
    auto warmup_time = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - prevtime).count();
    printf("Warmup done in %f ms\n", warmup_time);
}

void Solution::search(const vector<float> &query, vector<int> &res, int topk) {
    if (!index)
        throw std::runtime_error("search called before build");
    if (topk <= 0)
        throw std::runtime_error("search topk must be positive");
    if (query.empty())
        return;
    if (query.size() % index->d != 0)
        throw std::runtime_error("query size must be divisible by index dimension");
    // #pragma omp parallel for
    // for (int k = 0; k < query.size() / index->d; k++) {
    //     index->searchSingle(&query[k * index->d], topk, distances, res + k * topk);
    // }
    int n = query.size() / index->d;
    if ((int)res.size() != n * topk)
        res.resize(n * topk);
    if ((int)distances.size() != n * topk)
        distances.resize(n * topk);
    index->search({query.data(), n, index->d}, {res.data(), distances.data(), topk}); // 调低一下分数
}

void Solution::test(int d, const vector<float> &base, int topk) {
    int size = base.size() / d;
    if (topk <= 0)
        throw std::runtime_error("test topk must be positive");

    if (!index) {
        cout << "Test: Index not built" << endl;
        return;
    }
    int correct = 0, cnt = 0;
    mutex mtx;
    auto before_test = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    float distances[size];
    int labels[size];
    index->search({base.data(), size, d}, {labels, distances, 1});
    for (int i = 0; i < size; i++) {
        if (labels[i] == i)
            correct++;
        cnt++;
        auto current_time = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration<double, std::milli>(current_time - before_test).count() / cnt;
        if (cnt % 1000 == 0) {
            printf("Search progress [%d / %d] top1 acc %f; avg duration = %fms (Parallel)\n", cnt, size, float(correct) / cnt, duration);
        }
    }
    float recall = float(correct) / cnt;
    std::cout << "Recall: " << recall << std::endl;
}
void Solution::reset() {
    if (index) {
        index->reset();
    }
}
