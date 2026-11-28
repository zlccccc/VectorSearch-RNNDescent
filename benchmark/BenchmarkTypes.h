#pragma once

#include "../solution/solution.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace benchmark {

using std::cout;
using std::endl;
using std::runtime_error;
using std::string;
using std::transform;
using std::vector;

struct DenseVectors {
    vector<float> values;
    int rows = 0;
    int dim = 0;
};

struct GroundTruth {
    vector<int> ids;
    int rows = 0;
    int width = 0;
};

struct BenchmarkConfig {
    string mode = "random";
    bool mode_explicit = false;
    string dataset_name;
    string dataset_dir;
    string base_path;
    string query_path;
    string gt_path;
    int data_size = 100000;
    int dim = 256;
    int test_iter = 10000;
    int exact_check_queries = 20;
    int output_iter = 5;
    int topk = 10;
    int repeat = 10;
    int query_batch_size = 0;
    int warmup_topk = 10;
    int override_s = -1;
    int override_r = -1;
    int override_k0 = -1;
    int override_num_initialize = -1;
    int override_search_l = -1;
    int override_refine_max = -1;
    int override_fp32_refine_max = -1;
    int override_beam_size = -1;
    long long override_neighbor_pool_gb = -1;
    bool dim_explicit = false;
    rnndescent::RNNDescent::BuildConfig build_config;
    rnndescent::RNNDescent::SearchConfig search_config;
    rnndescent::IndexRNNDescent::PCAConfig pca_config;
};

struct AccuracyStats {
    int checked_queries = 0;
    int recall_k = 0;
    double recall1 = 0.0;
    double recallk = 0.0;
    double avg_distance_error = 0.0;
    double max_distance_error = 0.0;
    double avg_top1_gap = 0.0;
    string top1_gap_label = "exact";
};

struct BenchmarkStats {
    double build_seconds = 0.0;
    vector<float> latency_ms;
    AccuracyStats accuracy;
};

inline string normalize_dataset_name(string dataset_name) {
    transform(dataset_name.begin(), dataset_name.end(), dataset_name.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return dataset_name;
}

inline void apply_default_dataset_mode(BenchmarkConfig &config) {
    if (config.mode_explicit || config.mode != "random" || !config.dataset_dir.empty() || !config.base_path.empty() || !config.query_path.empty()) {
        return;
    }

    const char *dataset_env = std::getenv("DATASET");
    const string dataset_name = dataset_env != nullptr ? normalize_dataset_name(dataset_env) : "sift";
    const std::filesystem::path dataset_dir = std::filesystem::path("benches") / "datasets" / dataset_name;
    if (!std::filesystem::exists(dataset_dir)) {
        return;
    }

    config.mode = "dataset";
    config.dataset_name = dataset_name;
    config.dataset_dir = dataset_dir.string();
}

inline void apply_common_config(BenchmarkConfig &config) {
    const int thread_limit = std::min(16, omp_get_max_threads());
    config.build_config.num_threads = thread_limit;
    config.search_config.num_threads = thread_limit;
    config.search_config.beam_size = 8;
    config.build_config.S = 196;
    config.build_config.R = 2048;
    config.build_config.T1 = 4;
    config.build_config.T2 = 15;
}

inline void apply_dataset_preset(BenchmarkConfig &config) {
    const string key = normalize_dataset_name(config.dataset_name);
    if (key == "sift100k") {
        config.build_config.S = 64;
        config.build_config.R = 96;
        config.build_config.K0 = 24;
        config.search_config.num_initialize = 80;
        config.search_config.search_L = 72;
        config.search_config.refine_max = 24;
        config.search_config.beam_size = 4;
    } else if (key == "siftsmall" || key == "sift") {
        config.search_config.num_initialize = 192;
        config.search_config.search_L = 128;
        config.search_config.refine_max = 64;
        config.build_config.K0 = 32;
    } else if (key == "gist") {
        config.build_config.S = 320;
        config.build_config.R = 256;
        config.build_config.K0 = 64;
        config.search_config.num_initialize = 3072;
        config.search_config.search_L = 1152;
        config.search_config.refine_max = 1024;
    }
}

inline void apply_config_overrides(BenchmarkConfig &config) {
    if (config.override_s > 0)
        config.build_config.S = config.override_s;
    if (config.override_r > 0)
        config.build_config.R = config.override_r;
    if (config.override_k0 > 0)
        config.build_config.K0 = config.override_k0;
    if (config.override_num_initialize > 0)
        config.search_config.num_initialize = config.override_num_initialize;
    if (config.override_search_l > 0)
        config.search_config.search_L = config.override_search_l;
    if (config.override_refine_max > 0)
        config.search_config.refine_max = config.override_refine_max;
    if (config.override_fp32_refine_max >= 0)
        config.search_config.fp32_refine_max = config.override_fp32_refine_max;
    if (config.override_beam_size > 0)
        config.search_config.beam_size = config.override_beam_size;
    if (config.override_neighbor_pool_gb >= 0)
        config.build_config.neighbor_pool_size_limit_bytes = config.override_neighbor_pool_gb * 1024ll * 1024 * 1024;
}

inline void apply_solution_preset(BenchmarkConfig &config) {
    apply_common_config(config);
    if (config.mode == "dataset")
        apply_dataset_preset(config);
    apply_config_overrides(config);
}

inline const char *quantization_name(std::optional<rnndescent::QuantizedFormat> format) {
    if (!format)
        return "fp32";
    return *format == rnndescent::QuantizedFormat::UInt4 ? "uint4" : "8bit";
}

inline void print_solution_config(const BenchmarkConfig &config) {
    cout << "build config:"
         << " T1=" << config.build_config.T1 << " T2=" << config.build_config.T2 << " S=" << config.build_config.S << " R=" << config.build_config.R
         << " K0=" << config.build_config.K0
         << " local_three_hop_pruning=" << config.build_config.local_three_hop_pruning
         << " num_threads=" << config.build_config.num_threads
         << " neighbor_pool_limit_gb=" << (double)config.build_config.neighbor_pool_size_limit_bytes / 1024 / 1024 / 1024 << endl;
    cout << "search config:"
         << " num_initialize=" << config.search_config.num_initialize << " search_L=" << config.search_config.search_L
         << " beam_size=" << config.search_config.beam_size << " refine_max=" << config.search_config.refine_max
         << " fp32_refine_max=" << config.search_config.fp32_refine_max
         << " graph_quantization=" << quantization_name(config.search_config.graph_quantization)
         << " rerank_quantization=" << quantization_name(config.search_config.rerank_quantization)
         << " reorder_query_batch=" << config.search_config.reorder_query_batch
         << " single_query_threads=" << config.search_config.single_query_threads
         << " persistent_query_workers=" << config.search_config.persistent_query_workers
         << " single_query_refine_max=" << config.search_config.single_query_limits.candidates_per_worker
         << " single_query_max_expansions=" << config.search_config.single_query_limits.expansions_per_worker
         << " query_batch_size=" << config.query_batch_size
         << " num_threads=" << config.search_config.num_threads << endl;
    cout << "pca config:"
         << " enabled=" << (config.pca_config.enabled ? 1 : 0) << " out_dim=" << config.pca_config.out_dim << endl;
    cout << "warmup config:"
         << " topk=" << config.warmup_topk << endl;
}

inline Solution::WarmupConfig make_warmup_config(const BenchmarkConfig &config) {
    Solution::WarmupConfig warmup_config;
    warmup_config.topk = config.warmup_topk;
    return warmup_config;
}

} // namespace benchmark
