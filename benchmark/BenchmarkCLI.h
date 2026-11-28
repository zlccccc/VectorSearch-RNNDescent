#pragma once

#include "BenchmarkTypes.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace benchmark {

using std::cout;
using std::string;

inline void print_usage(const char *prog) {
    cout << "Usage:\n";
    cout << "  " << prog << " --dataset-dir <dir> [--dataset-name <preset>]\n";
    cout << "  " << prog << " --base <fvecs|bvecs> --query <fvecs|bvecs> [--gt <ivecs>]\n";
    cout << "  " << prog << " --mode random [--data-size <n>] [--dim <d>] [--test-iter <n>]\n";
    cout << "\nSearch quality:\n";
    cout << "  --search-L <n>            Rank cutoff for graph expansion (split across workers for one parallel query)\n";
    cout << "  --beam-size <n>           Nodes expanded per worker before each candidate-pool update\n";
    cout << "  --rerank-candidates <n>   Graph candidates per query sent to original-space reranking\n";
    cout << "  --fp32-candidates <n>     FP32 shortlist size after coarse ranking (default 32 for serial and batch search)\n";
    cout << "  --rerank-quantization <8bit|uint4|fp32>  Use fp32 to score all rerank candidates directly\n";
    cout << "  --topk <k>                Results per query (default 10)\n";
    cout << "\nIndex and advanced graph search:\n";
    cout << "  --S <n> --R <n> --K0 <n> [--local-three-hop-prune]\n";
    cout << "  --pca-out <d>             Graph projection dimension (default 0 keeps original dimensions)\n";
    cout << "  --graph-quantization <8bit|uint4>  Graph search encoding\n";
    cout << "  --neighbor-pool-gb <n>    Neighbor pool budget in GiB (default 0 disables vector caching)\n";
    cout << "  --num-initialize <n>      Entry points used for BFS ordering and search\n";
    cout << "\nQuery execution and measurement:\n";
    cout << "  --query-batch-size <n>    Queries per request (default 0 sends all queries at once)\n";
    cout << "  --single-query-threads <n>  Workers for a single-query search (default 1, capped at "
         << rnndescent::config::SearchConfig::max_single_query_threads << ")\n";
    cout << "                           Use with --query-batch-size 1 to measure single-query latency\n";
    cout << "  --no-persistent-query-workers  Use OpenMP for query dispatch instead of the default worker pool\n";
    cout << "  --no-reorder-query-batch  Execute PCA query batches in input order\n";
    cout << "  --repeat <n> --warmup-topk <k> --output-iter <n> --exact-check-queries <n>\n";
    cout << "\nAdvanced single-query overrides:\n";
    cout << "  --worker-candidates <n>   Per-worker FP32 shortlist size (0 uses Top"
         << rnndescent::config::SearchConfig::SingleQueryLimits::default_quantized_candidates
         << " after coarse ranking, or all local candidates for direct FP32)\n";
    cout << "  --worker-expansions <n>   Per-worker total graph expansion cap (0 is unlimited)\n";
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
        auto require_integer = [&](const string &name, long long minimum, long long maximum, const char *requirement) {
            const string input = require_value(name);
            char *end = nullptr;
            errno = 0;
            const long long value = std::strtoll(input.c_str(), &end, 10);
            if (errno == ERANGE || end == input.c_str() || end != input.c_str() + input.size() ||
                value < minimum || value > maximum)
                throw std::runtime_error(name + " requires " + requirement + ": " + input);
            return value;
        };
        auto require_int = [&](const string &name, int minimum = std::numeric_limits<int>::min()) {
            const char *requirement = "an integer";
            if (minimum == 0)
                requirement = "a non-negative integer";
            else if (minimum > 0)
                requirement = "a positive integer";
            return static_cast<int>(require_integer(name, minimum, std::numeric_limits<int>::max(), requirement));
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            config.mode = require_value(arg);
            config.mode_explicit = true;
        } else if (arg == "--dataset-name") {
            config.dataset_name = require_value(arg);
        } else if (arg == "--dataset-dir" || arg == "--base" || arg == "--query" || arg == "--gt") {
            const string path = require_value(arg);
            if (arg == "--dataset-dir")
                config.dataset_dir = path;
            else if (arg == "--base")
                config.base_path = path;
            else if (arg == "--query")
                config.query_path = path;
            else
                config.gt_path = path;
            if (!config.mode_explicit)
                config.mode = "dataset";
        } else if (arg == "--data-size") {
            config.data_size = require_int(arg);
        } else if (arg == "--dim") {
            config.dim = require_int(arg);
            config.dim_explicit = true;
        } else if (arg == "--test-iter") {
            config.test_iter = require_int(arg);
        } else if (arg == "--topk") {
            config.topk = require_int(arg, 1);
        } else if (arg == "--repeat") {
            config.repeat = require_int(arg, 1);
        } else if (arg == "--query-batch-size") {
            config.query_batch_size = require_int(arg, 0);
        } else if (arg == "--output-iter") {
            config.output_iter = require_int(arg);
        } else if (arg == "--exact-check-queries") {
            config.exact_check_queries = require_int(arg);
        } else if (arg == "--warmup-topk") {
            config.warmup_topk = require_int(arg, 0);
        } else if (arg == "--S") {
            config.override_s = require_int(arg, 1);
        } else if (arg == "--R") {
            config.override_r = require_int(arg, 1);
        } else if (arg == "--K0") {
            config.override_k0 = require_int(arg);
        } else if (arg == "--local-three-hop-prune") {
            config.build_config.local_three_hop_pruning = true;
        } else if (arg == "--num-initialize") {
            config.override_num_initialize = require_int(arg);
        } else if (arg == "--search-L") {
            config.override_search_l = require_int(arg);
        } else if (arg == "--rerank-candidates" || arg == "--refine-max") {
            config.override_refine_max = arg == "--refine-max" ? require_int(arg) : require_int(arg, 1);
        } else if (arg == "--fp32-candidates" || arg == "--fp32-refine-max") {
            config.search_config.fp32_refine_max = require_int(arg, 0);
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
            config.override_beam_size = require_int(arg);
        } else if (arg == "--reorder-query-batch") {
            config.search_config.reorder_query_batch = true;
        } else if (arg == "--no-reorder-query-batch") {
            config.search_config.reorder_query_batch = false;
        } else if (arg == "--single-query-threads") {
            config.search_config.single_query_threads =
                std::min(require_int(arg, 1), rnndescent::config::SearchConfig::max_single_query_threads);
        } else if (arg == "--persistent-query-workers" || arg == "--no-persistent-query-workers") {
            config.search_config.persistent_query_workers = arg == "--persistent-query-workers";
        } else if (arg == "--worker-candidates" || arg == "--single-query-refine-max") {
            config.search_config.single_query_limits.candidates_per_worker = require_int(arg, 0);
        } else if (arg == "--worker-expansions" || arg == "--single-query-max-expansions") {
            config.search_config.single_query_limits.expansions_per_worker = require_int(arg, 0);
        } else if (arg == "--neighbor-pool-gb") {
            constexpr long long bytes_per_gib = 1024ll * 1024 * 1024;
            const auto budget_gib = require_integer(arg, 0, std::numeric_limits<long long>::max() / bytes_per_gib,
                                                    "a non-negative integer budget that fits in bytes");
            config.build_config.neighbor_pool_size_limit_bytes = budget_gib * bytes_per_gib;
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

    return config;
}

} // namespace benchmark
