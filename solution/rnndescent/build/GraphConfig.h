#pragma once

#include "../common/QuantizedFormat.h"

#include <algorithm>
#include <limits>
#include <optional>

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
    static constexpr int max_single_query_threads = 8;

    // Optional per-worker overrides for a query split across several threads.
    struct SingleQueryLimits {
        static constexpr int default_quantized_candidates = 16;

        int candidates_per_worker = 0; // FP32 candidates. 0 uses Top16 after coarse ranking or all for direct FP32.
        int expansions_per_worker = 0; // Total graph nodes expanded by one worker. 0 leaves it uncapped.
    };

    int beam_size = 8; // Nodes expanded by one worker before each candidate-pool update.
    int num_threads = 16;
    int search_L = 96; // Eligible rank range for expansion. Split across workers for one parallel query.
    int num_initialize = 64;
    int refine_max = 64; // Graph candidates sent to reranking.
    int fp32_refine_max = 32; // Candidates reranked in FP32, 0 refines every graph candidate.
    QuantizedFormat graph_quantization = QuantizedFormat::Byte;
    // A disengaged format sends the graph candidates straight to FP32.
    std::optional<QuantizedFormat> rerank_quantization = QuantizedFormat::Byte;
    bool reorder_query_batch = true;
    int single_query_threads = 1;
    SingleQueryLimits single_query_limits;
    bool persistent_query_workers = true;

    bool uses_coarse_shortlist(int candidate_count) const {
        return rerank_quantization && fp32_refine_max > 0 && fp32_refine_max < candidate_count;
    }
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
    const int min_candidates = std::max(1, topk);
    config.single_query_threads = std::clamp(config.single_query_threads, 1, SearchConfig::max_single_query_threads);
    config.search_L = std::max(config.search_L, min_candidates);
    config.refine_max = std::max(config.refine_max, min_candidates);
    config.num_initialize = std::max({config.num_initialize, config.search_L, config.refine_max});
    config.fp32_refine_max = std::max(config.fp32_refine_max, 0);
    if (max_points > 0 && max_points < 4) {
        config.num_initialize = max_points;
    } else {
        const int available = max_points > 0 ? max_points : std::numeric_limits<int>::max();
        const int entries = std::min(config.num_initialize, available / 4 * 4);
        config.num_initialize = (entries + 3) / 4 * 4;
    }
    config.search_L = std::min(config.search_L, config.num_initialize);
    config.refine_max = std::min(config.refine_max, config.num_initialize);
    return config;
}

} // namespace config
} // namespace rnndescent
