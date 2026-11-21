#pragma once

#include <algorithm>

namespace rnndescent {
namespace config {

struct BuildConfig {
    int T1 = 4;
    int T2 = 15;
    int S = 16;
    int R = 96;
    int K0 = 48;
    int random_seed = 2021;
    int num_threads = 16;
    long long neighbor_pool_size_limit_bytes = 0;
    bool save_neighbor = true;
    bool random_init = true;
    bool local_three_hop_pruning = false;
};

struct SearchConfig {
    int beam_size = 8;
    int num_threads = 16;
    int search_L = 96;
    int num_initialize = 64;
    int refine_max = 64;
    int fp32_refine_max = 32;
};

inline BuildConfig sanitize_build_config(BuildConfig config) {
    config.T1 = std::max(config.T1, 1);
    config.T2 = std::max(config.T2, 1);
    config.S = std::max(config.S, 1);
    config.R = std::max(config.R, 1);
    config.K0 = std::max(config.K0, 1);
    config.num_threads = std::max(config.num_threads, 1);
    config.neighbor_pool_size_limit_bytes = std::max(config.neighbor_pool_size_limit_bytes, 0LL);
    config.save_neighbor = config.save_neighbor && config.neighbor_pool_size_limit_bytes > 0;
    return config;
}

inline SearchConfig sanitize_search_config(SearchConfig config, int topk = 1, int max_points = 0) {
    config.beam_size = std::max(config.beam_size, 1);
    config.num_threads = std::max(config.num_threads, 1);
    config.search_L = std::max(config.search_L, std::max(1, topk));
    config.num_initialize = std::max(config.num_initialize, config.search_L);
    config.refine_max = std::max(config.refine_max, std::max(topk, 1));
    config.fp32_refine_max = std::max(config.fp32_refine_max, 0);
    if (max_points > 0) {
        config.search_L = std::min(config.search_L, max_points);
        config.refine_max = std::min(config.refine_max, max_points);
        if (max_points >= 4) {
            const int max_aligned_points = (max_points / 4) * 4;
            config.num_initialize = std::min(config.num_initialize, max_aligned_points);
            config.num_initialize = std::max(config.num_initialize, std::max(config.search_L, topk));
            config.num_initialize = std::min(config.num_initialize, max_aligned_points);
            config.num_initialize = (config.num_initialize / 4) * 4;
            config.search_L = std::min(config.search_L, config.num_initialize);
        } else {
            config.num_initialize = max_points;
            config.search_L = std::min(config.search_L, config.num_initialize);
        }
    } else if (config.num_initialize % 4 != 0) {
        config.num_initialize = (config.num_initialize + 3) / 4 * 4;
    }
    return config;
}

} // namespace config
} // namespace rnndescent
