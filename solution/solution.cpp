#include "solution.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexNNDescent.h>

void Solution::build(int d, const vector<float> &base) {
    int size = base.size() / d;
    auto rnndescentindex = new rnndescent::IndexRNNDescent(d, faiss::METRIC_L2);
    // auto rnndescentindex = new rnndescent::IndexRNNDescent(d, faiss::METRIC_INNER_PRODUCT);
    rnndescentindex->rnndescent.numThreadsMax = omp_get_max_threads();
    rnndescentindex->rnndescent.beamSizeMax = 8;
    rnndescentindex->rnndescent.S = 196;
    rnndescentindex->rnndescent.R = 2048;

    rnndescentindex->rnndescent.T1 = 4;
    rnndescentindex->rnndescent.T2 = 15;
    rnndescentindex->verbose = true;
    index = rnndescentindex;

    if (d == 256) {
        rnndescentindex->rnndescent.numSearchInitializeItem = 1024;
        rnndescentindex->rnndescent.search_L = 512;
        rnndescentindex->rnndescent.K0 = 96;
    } else if (d == 512) {
        rnndescentindex->rnndescent.numSearchInitializeItem = 1024;
        rnndescentindex->rnndescent.search_L = 244; // topk
        rnndescentindex->rnndescent.K0 = 64;
        // rnndescentindex->rnndescent.search_L = 256;  // topk
        // rnndescentindex->rnndescent.K0 = 48;

        // rnndescentindex->rnndescent.search_L = 320;  // topk
        // rnndescentindex->rnndescent.K0 = 48;
    } else if (d == 1024) {
        rnndescentindex->rnndescent.numSearchInitializeItem = 160;
        // rnndescentindex->rnndescent.search_L = 160;  // topk
        // rnndescentindex->rnndescent.search_L = 144;  // topk
        // rnndescentindex->rnndescent.search_L = 120;  // topk
        rnndescentindex->rnndescent.search_L = 116; // topk
        rnndescentindex->rnndescent.K0 = 32;
    } else if (d == 1536) {
        rnndescentindex->rnndescent.numSearchInitializeItem = 512;
        // rnndescentindex->rnndescent.search_L = 240;  // topk
        // rnndescentindex->rnndescent.search_L = 224;  // topk
        // rnndescentindex->rnndescent.search_L = 208;  // topk
        // rnndescentindex->rnndescent.search_L = 192;  // topk
        // rnndescentindex->rnndescent.search_L = 184;  // topk
        // rnndescentindex->rnndescent.search_L = 180;  // topk
        // rnndescentindex->rnndescent.K0 = 64;
        rnndescentindex->rnndescent.search_L = 184; // topk
        rnndescentindex->rnndescent.K0 = 48;
    } else {
        throw std::runtime_error("Unsupported dimension");
    }

#ifdef INTERNAL_CLOCK_TEST
    rnndescentindex->rnndescent.S = 32;
    rnndescentindex->rnndescent.R = 256;
#endif

    index->train(size, &base[0]);
    index->add(size, &base[0]);

    // warmup貌似有bug, 而且好像没啥用(好像还是有点点用? 本地测下来100分); 待修复
    // (代码里面那个usefulset不如直接heap里面用一用得了; 根本不需要额外开空间)

    auto prevtime = std::chrono::system_clock::now();
    int warmup_count = 10000, warmup_search_count = 1;
    std::vector<float> query(d * warmup_count);
    int res[topk * warmup_count];
    for (int i = 0; i < warmup_count; i++) {
        int index = random() % size;
        memcpy(query.data() + d * i, base.data() + d * index, d * sizeof(float));
    }
    for (int i = 0; i < warmup_search_count; i++)
        search(query, res);
    index->rnndescent.reset_time();
    auto warmup_time = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - prevtime).count();
    printf("Warmup done in %f ms\n", warmup_time);
}

void Solution::search(const vector<float> &query, int *res) {
    // #pragma omp parallel for
    // for (int k = 0; k < query.size() / index->d; k++) {
    //     index->searchSingle(&query[k * index->d], topk, distances, res + k * topk);
    // }
    int n = query.size() / index->d;
    assert(n <= querysize);
    // omp_set_num_threads(1);
    index->search(n, query.data(), topk, distances, res); // 调低一下分数
}

void Solution::test(int d, const vector<float> &base) {
    int size = base.size() / d;

    if (index == nullptr) {
        cout << "Test: Index not built" << endl;
        return;
    }
    int correct = 0, cnt = 0;
    mutex mtx;
    auto before_test = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    float distances[size];
    int labels[size];
    index->search(size, &base[0], 1, distances, labels);
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