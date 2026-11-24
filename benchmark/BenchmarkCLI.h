#pragma once

#include "BenchmarkTypes.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace benchmark {

using std::cout;
using std::stoi;
using std::stoll;
using std::string;

inline void print_usage(const char *prog) {
    cout << "Usage:\n";
    cout << "  " << prog << "                      # random benchmark\n";
    cout << "  " << prog << " --mode dataset --dataset-dir <dir> [--topk 10] [--repeat 10]\n";
    cout << "  " << prog << " --mode dataset --base <base.fvecs|bvecs> --query <query.fvecs|bvecs> [--gt <groundtruth.ivecs>]\n";
    cout << "Optional args:\n";
    cout << "  --data-size <n> --dim <d> --test-iter <n> --topk <k> --repeat <n>\n";
    cout << "  --output-iter <n> --exact-check-queries <n> --warmup-topk <k>\n";
    cout << "  --S <n> --R <n> --K0 <n>\n";
    cout << "  --local-three-hop-prune  Use three-hop alternatives when selecting final graph edges\n";
    cout << "  --num-initialize <n> --search-L <n> --refine-max <n> [--beam-size <n>]\n";
    cout << "  --reorder-query-batch  Order queries within a PCA batch for cache locality (off by default)\n";
    cout << "  --fp32-refine-max <n>  FP32 shortlist size (default 32; 0 refines all candidates)\n";
    cout << "  --graph-quantization <8bit|uint4>  Graph navigation encoding\n";
    cout << "  --rerank-quantization <8bit|uint4|fp32>  Original-dimensional shortlist encoding or direct FP32\n";
    cout << "  --neighbor-pool-gb <n>  Neighbor pool budget in GiB (default 0 disables vector caching)\n";
    cout << "  --enable-pca --disable-pca --pca-out <n>\n";
}

inline BenchmarkConfig parse_args(int argc, char **argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; i++) {
        const string arg = argv[i];
        auto require_value = [&](const string &name) -> string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for argument: " + name);
            return argv[++i];
        };
        auto require_positive_int = [&](const string &name) -> int {
            const string input = require_value(name);
            size_t parsed = 0;
            const int value = stoi(input, &parsed);
            if (parsed != input.size() || value <= 0)
                throw std::runtime_error(name + " requires a positive integer");
            return value;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            config.mode = require_value(arg);
            config.mode_explicit = true;
        } else if (arg == "--dataset-name") {
            config.dataset_name = require_value(arg);
        } else if (arg == "--dataset-dir") {
            config.dataset_dir = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--base") {
            config.base_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--query") {
            config.query_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--gt") {
            config.gt_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--data-size") {
            config.data_size = stoi(require_value(arg));
        } else if (arg == "--dim") {
            config.dim = stoi(require_value(arg));
            config.dim_explicit = true;
        } else if (arg == "--test-iter") {
            config.test_iter = stoi(require_value(arg));
        } else if (arg == "--topk") {
            config.topk = stoi(require_value(arg));
        } else if (arg == "--repeat") {
            config.repeat = stoi(require_value(arg));
        } else if (arg == "--output-iter") {
            config.output_iter = stoi(require_value(arg));
        } else if (arg == "--exact-check-queries") {
            config.exact_check_queries = stoi(require_value(arg));
        } else if (arg == "--warmup-topk") {
            config.warmup_topk = stoi(require_value(arg));
        } else if (arg == "--S") {
            config.override_s = require_positive_int(arg);
        } else if (arg == "--R") {
            config.override_r = require_positive_int(arg);
        } else if (arg == "--K0") {
            config.override_k0 = stoi(require_value(arg));
        } else if (arg == "--local-three-hop-prune") {
            config.build_config.local_three_hop_pruning = true;
        } else if (arg == "--num-initialize") {
            config.override_num_initialize = stoi(require_value(arg));
        } else if (arg == "--search-L") {
            config.override_search_l = stoi(require_value(arg));
        } else if (arg == "--refine-max") {
            config.override_refine_max = stoi(require_value(arg));
        } else if (arg == "--fp32-refine-max") {
            const string input = require_value(arg);
            size_t parsed = 0;
            const int value = stoi(input, &parsed);
            if (parsed != input.size() || value < 0)
                throw std::runtime_error("--fp32-refine-max requires a non-negative integer");
            config.override_fp32_refine_max = value;
        } else if (arg == "--graph-quantization" || arg == "--rerank-quantization") {
            const auto value = require_value(arg);
            if (arg == "--rerank-quantization" && value == "fp32") {
                config.search_config.rerank_quantization.reset();
            } else {
                if (value != "8bit" && value != "uint4")
                    throw std::runtime_error(arg + " requires 8bit or uint4" +
                                             (arg == "--rerank-quantization" ? " or fp32" : ""));
                const auto format = value == "uint4" ? rnndescent::QuantizedFormat::UInt4 : rnndescent::QuantizedFormat::Byte;
                if (arg == "--graph-quantization")
                    config.search_config.graph_quantization = format;
                else
                    config.search_config.rerank_quantization = format;
            }
        } else if (arg == "--beam-size") {
            config.override_beam_size = stoi(require_value(arg));
        } else if (arg == "--reorder-query-batch") {
            config.search_config.reorder_query_batch = true;
        } else if (arg == "--neighbor-pool-gb") {
            const string input = require_value(arg);
            size_t parsed = 0;
            const long long budget_gib = stoll(input, &parsed);
            constexpr long long bytes_per_gib = 1024ll * 1024 * 1024;
            if (parsed != input.size() || budget_gib < 0 || budget_gib > std::numeric_limits<long long>::max() / bytes_per_gib)
                throw std::runtime_error("--neighbor-pool-gb requires a non-negative integer budget that fits in bytes");
            config.override_neighbor_pool_gb = budget_gib;
        } else if (arg == "--enable-pca") {
            config.override_pca_enabled = true;
            config.pca_enabled = true;
        } else if (arg == "--disable-pca") {
            config.override_pca_enabled = true;
            config.pca_enabled = false;
            config.pca_out_dim = 0;
        } else if (arg == "--pca-out") {
            config.override_pca_enabled = true;
            config.pca_enabled = true;
            config.pca_out_dim = stoi(require_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (config.topk <= 0 || config.repeat <= 0 || config.warmup_topk < 0) {
        throw std::runtime_error("topk/repeat must be positive and warmup-topk must be non-negative");
    }
    return config;
}

} // namespace benchmark
