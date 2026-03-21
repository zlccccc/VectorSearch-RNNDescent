#pragma once

#include "BenchmarkCLI.h"
#include "BenchmarkDataIO.h"
#include "BenchmarkEvaluation.h"
#include "BenchmarkReport.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace benchmark {

namespace chrono = std::chrono;

using std::cerr;
using std::cout;
using std::exception;
using std::vector;

inline void run_random_benchmark(BenchmarkConfig config) {
    Solution solution;
    BenchmarkStats stats;
    rnndescent::Logger::line("start random benchmark");
    std::mt19937 rng(0);
    vector<float> dataset(1ULL * config.data_size * config.dim);
    for (int i = 0; i < config.data_size; i++)
        randvector(rng, dataset, i, config.dim);

    vector<float> query(1ULL * config.test_iter * config.dim);
    for (int i = 0; i < config.test_iter; i++)
        randvector(rng, query, i, config.dim);

    vector<int> result((size_t)config.topk * config.test_iter);
    apply_solution_preset(config);
    config.dataset_name = "random";
    print_solution_config(config);
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(config.dim, dataset, make_warmup_config(config), config.build_config, config.search_config, config.pca_config);
    const auto after_build = chrono::high_resolution_clock::now();
    stats.build_seconds = chrono::duration<double>(after_build - before_build).count();
    cout << "build time: " << stats.build_seconds << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, std::milli>(after_test - before_test).count() / config.test_iter;
        stats.latency_ms.push_back(t);
        cout << "[" << i << "] average test time: " << t << " ms; " << 1000. / t << " offline score" << endl;
    }

    stats.accuracy = evaluate_accuracy_exact(dataset, query, result, solution.distance_buffer(), config.data_size, config.dim, config.topk,
                                             config.exact_check_queries);
    print_sample_results(dataset, query, result, solution.distance_buffer(), config.dim, config.topk, config.output_iter);
    write_benchmark_report(config, stats);
    solution.reset();
}

inline void run_dataset_benchmark(BenchmarkConfig config) {
    BenchmarkStats stats;
    resolve_dataset_paths(config);
    cout << "dataset mode: " << (config.dataset_name.empty() ? string("custom") : config.dataset_name) << endl;
    cout << "base: " << config.base_path << endl;
    cout << "query: " << config.query_path << endl;
    if (!config.gt_path.empty())
        cout << "groundtruth: " << config.gt_path << endl;

    const DenseVectors dataset = load_dense_vectors(config.base_path);
    const DenseVectors query = load_dense_vectors(config.query_path);
    if (dataset.dim != query.dim) {
        throw std::runtime_error("dataset/query dimension mismatch");
    }

    GroundTruth gt;
    if (!config.gt_path.empty()) {
        gt = load_ground_truth(config.gt_path);
        if (gt.rows != query.rows) {
            throw std::runtime_error("ground truth row count does not match query count");
        }
    }

    cout << "loaded base vectors: " << dataset.rows << " x " << dataset.dim << endl;
    cout << "loaded query vectors: " << query.rows << " x " << query.dim << endl;
    if (gt.rows > 0)
        cout << "loaded ground truth: " << gt.rows << " x " << gt.width << endl;

    Solution solution;
    vector<int> result((size_t)config.topk * query.rows);
    apply_solution_preset(config);
    print_solution_config(config);
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(dataset.dim, dataset.values, make_warmup_config(config), config.build_config, config.search_config, config.pca_config);
    const auto after_build = chrono::high_resolution_clock::now();
    stats.build_seconds = chrono::duration<double>(after_build - before_build).count();
    cout << "build time: " << stats.build_seconds << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query.values, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, std::milli>(after_test - before_test).count() / query.rows;
        stats.latency_ms.push_back(t);
        cout << "[" << i << "] average test time: " << t << " ms/query; " << 1000. / t << " qps(k)" << endl;
    }

    if (gt.rows > 0) {
        stats.accuracy = evaluate_accuracy_with_gt(dataset.values, query.values, gt, result, solution.distance_buffer(), dataset.dim, config.topk);
    }
    print_sample_results(dataset.values, query.values, result, solution.distance_buffer(), dataset.dim, config.topk, config.output_iter);
    write_benchmark_report(config, stats);
    solution.reset();
}

inline int run_application(int argc, char **argv) {
    try {
        BenchmarkConfig config = parse_args(argc, argv);
        apply_default_dataset_mode(config);
        if (config.mode == "dataset") {
            run_dataset_benchmark(config);
        } else if (config.mode == "random") {
            if (config.dim_explicit) {
                run_random_benchmark(config);
            } else {
                for (int dim : {256, 512}) {
                    BenchmarkConfig current = config;
                    current.dim = dim;
                    cout << "\n=== random benchmark dim=" << dim << " ===" << endl;
                    run_random_benchmark(current);
                }
            }
        } else {
            throw std::runtime_error("unsupported mode: " + config.mode);
        }
        return 0;
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        print_usage(argv[0]);
        return 1;
    }
}

} // namespace benchmark
