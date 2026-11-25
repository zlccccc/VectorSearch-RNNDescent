#pragma once

#include "BenchmarkTypes.h"

#include <algorithm>
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
    cout << "  " << prog << " --dataset-dir <dir> [--dataset-name <preset>]\n";
    cout << "  " << prog << " --base <fvecs|bvecs> --query <fvecs|bvecs> [--gt <ivecs>]\n";
    cout << "  " << prog << " --mode random [--data-size <n>] [--dim <d>] [--test-iter <n>]\n";
    cout << "\nSearch quality:\n";
    cout << "  --search-L <n>            Expand candidates ranked in the current closest L\n";
    cout << "  --rerank-candidates <n>   Graph candidates sent to original-space reranking\n";
    cout << "  --fp32-candidates <n>     Candidates scored in FP32 after quantized reranking (default 32)\n";
    cout << "  --rerank-quantization <8bit|uint4|fp32>  Use fp32 to score all rerank candidates directly\n";
    cout << "  --topk <k>                Results per query (default 10)\n";
    cout << "\nIndex and advanced graph search:\n";
    cout << "  --S <n> --R <n> --K0 <n> [--local-three-hop-prune]\n";
    cout << "  --pca-out <d>             Graph projection dimension (default 0 keeps original dimensions)\n";
    cout << "  --graph-quantization <8bit|uint4>  Graph search encoding\n";
    cout << "  --neighbor-pool-gb <n>    Neighbor pool budget in GiB (default 0 disables vector caching)\n";
    cout << "  --num-initialize <n>      Entry points used for BFS ordering and search\n";
    cout << "  --beam-size <n>           Nodes expanded between candidate pool updates\n";
    cout << "\nQuery execution and measurement:\n";
    cout << "  --query-batch-size <n>    Queries per request (default 0 sends all queries at once)\n";
    cout << "  --single-query-threads <n>  Workers for a single-query search (default 1, capped at "
         << rnndescent::config::SearchConfig::max_single_query_threads << ")\n";
    cout << "  --worker-candidates <n>   FP32 candidates per worker (0 keeps "
         << rnndescent::config::SearchConfig::SingleQueryLimits::default_quantized_candidates
         << " after coarse ranking, or all for direct FP32)\n";
    cout << "  --worker-expansions <n>   Expanded nodes per worker for one parallel query (default 0 is unlimited)\n";
    cout << "  --no-reorder-query-batch  Execute PCA query batches in input order\n";
    cout << "  --repeat <n> --warmup-topk <k> --output-iter <n> --exact-check-queries <n>\n";
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
        auto require_int = [&](const string &name, int minimum) -> int {
            const string input = require_value(name);
            size_t parsed = 0;
            const int value = stoi(input, &parsed);
            if (parsed != input.size() || value < minimum)
                throw std::runtime_error(name + (minimum == 0 ? " requires a non-negative integer" : " requires a positive integer"));
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
        } else if (arg == "--query-batch-size") {
            config.query_batch_size = require_int(arg, 0);
        } else if (arg == "--output-iter") {
            config.output_iter = stoi(require_value(arg));
        } else if (arg == "--exact-check-queries") {
            config.exact_check_queries = stoi(require_value(arg));
        } else if (arg == "--warmup-topk") {
            config.warmup_topk = stoi(require_value(arg));
        } else if (arg == "--S") {
            config.override_s = require_int(arg, 1);
        } else if (arg == "--R") {
            config.override_r = require_int(arg, 1);
        } else if (arg == "--K0") {
            config.override_k0 = stoi(require_value(arg));
        } else if (arg == "--local-three-hop-prune") {
            config.build_config.local_three_hop_pruning = true;
        } else if (arg == "--num-initialize") {
            config.override_num_initialize = stoi(require_value(arg));
        } else if (arg == "--search-L") {
            config.override_search_l = stoi(require_value(arg));
        } else if (arg == "--rerank-candidates" || arg == "--refine-max") {
            config.override_refine_max = arg == "--refine-max" ? stoi(require_value(arg)) : require_int(arg, 1);
        } else if (arg == "--fp32-candidates" || arg == "--fp32-refine-max") {
            config.override_fp32_refine_max = require_int(arg, 0);
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
        } else if (arg == "--no-reorder-query-batch") {
            config.search_config.reorder_query_batch = false;
        } else if (arg == "--single-query-threads") {
            config.search_config.single_query_threads =
                std::min(require_int(arg, 1), rnndescent::config::SearchConfig::max_single_query_threads);
        } else if (arg == "--worker-candidates" || arg == "--single-query-refine-max") {
            config.search_config.single_query_limits.candidates_per_worker = require_int(arg, 0);
        } else if (arg == "--worker-expansions" || arg == "--single-query-max-expansions") {
            config.search_config.single_query_limits.expansions_per_worker = require_int(arg, 0);
        } else if (arg == "--neighbor-pool-gb") {
            const string input = require_value(arg);
            size_t parsed = 0;
            const long long budget_gib = stoll(input, &parsed);
            constexpr long long bytes_per_gib = 1024ll * 1024 * 1024;
            if (parsed != input.size() || budget_gib < 0 || budget_gib > std::numeric_limits<long long>::max() / bytes_per_gib)
                throw std::runtime_error("--neighbor-pool-gb requires a non-negative integer budget that fits in bytes");
            config.override_neighbor_pool_gb = budget_gib;
        } else if (arg == "--enable-pca") {
            config.pca_config.enabled = true;
        } else if (arg == "--disable-pca") {
            config.pca_config = {};
        } else if (arg == "--pca-out") {
            const int out_dim = require_int(arg, 0);
            config.pca_config = {out_dim > 0, out_dim};
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
