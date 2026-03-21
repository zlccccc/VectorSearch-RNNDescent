#pragma once

#include "BenchmarkTypes.h"

#include <cstdlib>
#include <iostream>
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
    cout << "  --K0 <n> --num-initialize <n> --search-L <n> --refine-max <n> [--beam-size <n>]\n";
    cout << "  --neighbor-pool-gb <n>\n";
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

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            config.mode = require_value(arg);
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
        } else if (arg == "--K0") {
            config.override_k0 = stoi(require_value(arg));
        } else if (arg == "--num-initialize") {
            config.override_num_initialize = stoi(require_value(arg));
        } else if (arg == "--search-L") {
            config.override_search_l = stoi(require_value(arg));
        } else if (arg == "--refine-max") {
            config.override_refine_max = stoi(require_value(arg));
        } else if (arg == "--beam-size") {
            config.override_beam_size = stoi(require_value(arg));
        } else if (arg == "--neighbor-pool-gb") {
            config.override_neighbor_pool_gb = stoll(require_value(arg));
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
