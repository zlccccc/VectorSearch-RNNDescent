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

inline vector<float> measure_queries(Solution &solution, const vector<float> &query, vector<int> &result,
                                     const BenchmarkConfig &config, BenchmarkStats &stats) {
    const auto queries = rnndescent::FloatMatrixView::from_vector(query, config.dim);
    const int rows = queries.row_count();
    const int batch_size = config.query_batch_size > 0 ? config.query_batch_size : rows;
    vector<float> distances(result.size());
    const auto output = rnndescent::SearchResultView::from_vectors(result, distances, config.topk);

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::steady_clock::now();
        for (int offset = 0; offset < rows;) {
            const int count = std::min(batch_size, rows - offset);
            solution.search({queries.row_ptr(offset), count, config.dim}, output.slice(offset));
            offset += count;
        }
        const auto after_test = chrono::steady_clock::now();
        const float t = chrono::duration<double, std::milli>(after_test - before_test).count() / rows;
        stats.latency_ms.push_back(t);
        cout << "[" << i << "] average test time: " << t << " ms/query; " << 1000. / t << " QPS" << endl;
    }
    return distances;
}

inline void run_index_benchmark(const BenchmarkConfig &config, const DenseVectors &dataset, const DenseVectors &query,
                                const GroundTruth *ground_truth) {
    Solution solution;
    BenchmarkStats stats;
    vector<int> result(static_cast<size_t>(config.topk) * query.rows);
    print_solution_config(config);
    const auto before_build = chrono::steady_clock::now();
    solution.build(dataset.dim, dataset.values, make_warmup_config(config), config.build_config, config.search_config, config.pca_config);
    const auto after_build = chrono::steady_clock::now();
    stats.build_seconds = chrono::duration<double>(after_build - before_build).count();
    cout << "build time: " << stats.build_seconds << endl;

    const auto distances = measure_queries(solution, query.values, result, config, stats);
    if (!ground_truth) {
        stats.accuracy = evaluate_accuracy_exact(dataset.values, query.values, result, distances, dataset.rows,
                                                 dataset.dim, config.topk, config.exact_check_queries);
    } else if (ground_truth->rows > 0) {
        stats.accuracy = evaluate_accuracy_with_gt(dataset.values, query.values, *ground_truth, result, distances,
                                                   dataset.dim, config.topk);
    }
    print_sample_results(dataset.values, query.values, result, distances, dataset.dim, config.topk, config.output_iter);
    write_benchmark_report(config, stats);
}

inline void run_random_benchmark(BenchmarkConfig config) {
    if (config.dim <= 0 || config.data_size <= 0 || config.test_iter <= 0)
        throw std::runtime_error("random dimension, data size and query count must be positive");

    rnndescent::Logger::line("start random benchmark");
    std::mt19937 rng(0);
    DenseVectors dataset{vector<float>(static_cast<size_t>(config.data_size) * config.dim), config.data_size, config.dim};
    for (int i = 0; i < dataset.rows; ++i)
        randvector(rng, dataset.values, i, dataset.dim);

    DenseVectors query{vector<float>(static_cast<size_t>(config.test_iter) * config.dim), config.test_iter, config.dim};
    for (int i = 0; i < query.rows; ++i)
        randvector(rng, query.values, i, query.dim);

    apply_solution_preset(config);
    config.dataset_name = "random";
    run_index_benchmark(config, dataset, query, nullptr);
}

inline void run_dataset_benchmark(BenchmarkConfig config) {
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

    config.dim = dataset.dim;
    apply_solution_preset(config);
    run_index_benchmark(config, dataset, query, &gt);
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
