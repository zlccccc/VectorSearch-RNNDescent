#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <memory_resource>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <omp.h>

#include "Assertions.h"
#include "CandidateBitmap.h"
#include "CandidateReranker.h"
#include "GraphPruning.h"
#include "Logger.h"
#include "Views.h"
#include "discomputer/PlatformSimdDistanceComputer.h"

namespace rnndescent {

struct MutexWarpper : std::mutex {
    MutexWarpper() = default;
    MutexWarpper(MutexWarpper const &) noexcept : std::mutex() {}
    bool operator==(MutexWarpper const &other) noexcept { return this == &other; }
};

struct XNeighbor {
    int id_;
    float distance;
    XNeighbor() = default;
    XNeighbor(int id, float distance, bool flag) : id_(id), distance(distance) {
        if (flag) {
            id_ |= 0x80000000;
        }
    }
    int id() const { return id_ & 0x7fffffff; }
    bool flag() const { return (id_ & 0x80000000) != 0; }
    void setflag(bool flag) {
        if (flag)
            id_ |= 0x80000000;
        else
            id_ &= 0x7fffffff;
    }
    inline bool operator<(const XNeighbor &other) const {
        if (distance == other.distance) {
            return (id_ & 0x7fffffff) < (other.id_ & 0x7fffffff);
        }
        return distance < other.distance;
    }
};

namespace detail {

inline void sort_and_merge_neighbors(std::vector<XNeighbor> &neighbors) {
    std::sort(neighbors.begin(), neighbors.end());
    // Copies of an edge carry the same distance, so sorting groups them.
    const auto same_edge = [](const XNeighbor &a, const XNeighbor &b) { return a.id() == b.id(); };
    auto kept = std::adjacent_find(neighbors.begin(), neighbors.end(), same_edge);
    if (kept == neighbors.end())
        return;
    for (auto next = kept + 1; next != neighbors.end(); ++next) {
        if (same_edge(*kept, *next)) {
            // Equal IDs differ only in the high bit. Keep the edge fresh if either copy is fresh.
            kept->id_ |= next->id_;
        } else {
            *++kept = *next;
        }
    }
    neighbors.erase(kept + 1, neighbors.end());
}

} // namespace detail

struct RNNDescent {
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
        // Rank refinement candidates in 8-bit space, then recompute this many in FP32.
        // Zero keeps full FP32 refinement. Queries returning more neighbors raise this limit to Top K.
        int fp32_refine_max = 32;
    };

    static BuildConfig sanitize_build_config(BuildConfig config) {
        config.T1 = std::max(config.T1, 1);
        config.T2 = std::max(config.T2, 1);
        config.S = std::max(config.S, 1);
        config.R = std::max(config.R, 1);
        config.K0 = std::max(config.K0, 1);
        config.num_threads = std::max(config.num_threads, 1);
        config.neighbor_pool_size_limit_bytes = std::max(config.neighbor_pool_size_limit_bytes, 0ll);
        config.save_neighbor = config.save_neighbor && config.neighbor_pool_size_limit_bytes > 0;
        return config;
    }

    static SearchConfig sanitize_search_config(SearchConfig config, int topk = 1, int max_points = 0) {
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
    };

    static void gen_random(std::mt19937 &rng, std::vector<int> &addr, const int N) {
        const int size = addr.size();
        RNNDESCENT_ASSERT_MSG(0 <= size && size <= N, "gen_random requires 0 <= size <= N");
        if (size == 0)
            return;

        // Partial Fisher-Yates shuffle with sparse swaps: sample `size` unique ids from [0, N)
        // without materializing the full range.
        std::unordered_map<int, int> swapped;
        swapped.reserve(size * 2);
        for (int i = 0; i < size; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);

            const auto it_i = swapped.find(i);
            const int value_i = (it_i == swapped.end()) ? i : it_i->second;
            const auto it_j = swapped.find(j);
            const int value_j = (it_j == swapped.end()) ? j : it_j->second;

            swapped[j] = value_i;
            addr[i] = value_j;
        }
    }

    using KNNGraph = std::vector<std::vector<XNeighbor>>;

    RNNDescent(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config, const SearchConfig &search_config) {
        build(data_view, verbose, build_config, search_config);
    }

#ifdef INTERNAL_CLOCK_TEST
    struct PerfStats {
        float init_time{0};
        float getneighbor_time{0};
        float calculate_time{0};
        float update_time{0};
        float recalculate_time{0};
        long long query_count{0};
        long long calculate_distance_count{0};
        long long for_count{0};
        long long bfs_length{0};
        long long useful_count{0};
        long long bfs_items{0};

        void reset() {
            init_time = 0;
            getneighbor_time = 0;
            calculate_time = 0;
            update_time = 0;
            recalculate_time = 0;
            query_count = 0;
            calculate_distance_count = 0;
            for_count = 0;
            bfs_length = 0;
            useful_count = 0;
            bfs_items = 0;
        }
    } perf_stats;
#endif
    ~RNNDescent() { reset(); }

    void generate_graph(KNNGraph &graph, const FloatMatrixView &data_view, int n, bool verbose, const BuildConfig &build_config) {
        Logger::info(verbose, "generte graph ntotal = %d\n", n);
        auto qdis = SelectedDistanceComputerFactory::create_build_graph(data_view.data_ptr(), data_view.row_count(), data_view.dimension());
        init_graph(graph, n, *qdis, build_config);
        for (int t1 = 0; t1 < build_config.T1; ++t1) {
            Logger::info(verbose, "Iter %d : ", t1);
            for (int t2 = 0; t2 < build_config.T2; ++t2) {
                update_neighbors(graph, n, *qdis, build_config);
                Logger::info(verbose, "#");
            }
            Logger::line(verbose, "");

            if (t1 != build_config.T1 - 1)
                add_reverse_edges(graph, n, build_config);
        }

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u)
            detail::sort_and_merge_neighbors(graph[u]);

        if (build_config.local_three_hop_pruning) {
            const auto replaced = detail::prune_local_three_hop(graph, build_config.K0, build_config.num_threads);
            Logger::info(verbose, "Local three-hop pruning replaced %zu edges\n", replaced);
        }

        size_t all_edges_size = 0;
        for (int u = 0; u < n; ++u) {
            if (graph[u].size() > build_config.K0)
                graph[u].resize(build_config.K0);
            all_edges_size += graph[u].size();
        }
        Logger::info(verbose, "graph edges size = %zu\n", all_edges_size);
    }

    void build(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config, const SearchConfig &search_config) {
        data_view.validate("build data");
        const BuildConfig safe_build_config = sanitize_build_config(build_config);
        const int n = data_view.row_count();
        const int dim = data_view.dimension();
        const SearchConfig safe_search_config = sanitize_search_config(search_config, 1, n);

        reset();

        if (verbose)
            Logger::info(verbose, "Parameters: S=%d, R=%d, T1=%d, T2=%d; Point=%d; numThreadsMax=%d\n", safe_build_config.S, safe_build_config.R,
                         safe_build_config.T1, safe_build_config.T2, n, safe_search_config.num_threads);

        std::vector<int> edges;
        std::vector<size_t> edge_offsets(static_cast<size_t>(n) + 1);
        {
            KNNGraph graph;
            generate_graph(graph, data_view, n, verbose, safe_build_config);
            std::vector<MutexWarpper>().swap(node_locks_);

            size_t padded_edge_count = 0;
            for (const auto &neighbors : graph)
                padded_edge_count += (neighbors.size() + 3) / 4 * 4;
            edges.reserve(padded_edge_count);
            std::mt19937 padding_rng(safe_build_config.random_seed + 1);
            std::uniform_int_distribution<int> padding_dist(0, n - 1);
            std::unordered_set<int> selected_ids;
            for (int i = 0; i < n; i++) {
                auto &pool = graph[i];
                sort(pool.begin(), pool.end());
                // Keep the neighbor IDs unique and pad the row for four-way scoring.
                selected_ids.clear();
                selected_ids.reserve(pool.size() * 2 + 4);
                for (auto &edge : pool) {
                    if (!selected_ids.count(edge.id())) { // 这个地方感觉不太应该写成这样; 如果写得好的话, 理论上刚开始的edge id不会重复
                        selected_ids.insert(edge.id());
                        edges.push_back(edge.id());
                    }
                }
                while ((edges.size() - edge_offsets[i]) % 4 != 0) {
                    int id = padding_dist(padding_rng);
                    while (selected_ids.count(id))
                        id = padding_dist(padding_rng);
                    selected_ids.insert(id);
                    edges.push_back(id);
                }
                edge_offsets[i + 1] = edges.size();
                std::vector<XNeighbor>().swap(pool);
            }
        }
#if defined(__GLIBC__)
        // Return pages freed by graph construction before allocating the search index.
        malloc_trim(0);
#endif
        {
            const auto original_to_bfs = reorder_by_bfs(edges, edge_offsets, safe_search_config.num_initialize, safe_build_config, verbose);

            // Write quantized rows in BFS order directly from the original matrix.
            graph_distance_computer =
                SelectedDistanceComputerFactory::create_cached_graph(data_view.data_ptr(), n, dim, search_from_ids.data());

            const auto code_stride = graph_distance_computer->code_stride();
            neighbor_pool_.initialize(
                code_stride, edge_offsets, safe_build_config.neighbor_pool_size_limit_bytes, safe_build_config.save_neighbor);
            final_graph_neighbors.reserve(n);
            for (int bfs_id = 0; bfs_id < n; ++bfs_id) {
                const int original_id = search_from_ids[bfs_id];
                const size_t begin = edge_offsets[original_id];
                const int count = static_cast<int>(edge_offsets[original_id + 1] - begin);
                const int *neighbors = count > 0 ? edges.data() + begin : nullptr;
                final_graph_neighbors.emplace_back(neighbor_pool_, neighbors, count, graph_distance_computer.get(), original_to_bfs,
                                                   safe_build_config.save_neighbor);
            }
            if (safe_build_config.save_neighbor) {
#pragma omp parallel num_threads(safe_build_config.num_threads)
                {
                    std::vector<uint8_t> reorder_scratch(4 * static_cast<size_t>(code_stride));
#pragma omp for
                    for (int i = 0; i < n; ++i)
                        final_graph_neighbors[i].populate_cache(reorder_scratch.data());
                }
            }
        }
        thread_candidate_ids_.resize(safe_search_config.num_threads);
        threadRetset.resize(safe_search_config.num_threads);
        neighborDistance.resize(safe_search_config.num_threads);
        threadUsefulset.resize(safe_search_config.num_threads);
        const size_t batch_capacity = padded_degree(safe_build_config.K0) * safe_search_config.beam_size;
        for (int i = 0; i < safe_search_config.num_threads; i++) {
            thread_candidate_ids_[i].initialize(n);
            threadRetset[i].reserve(std::max(safe_search_config.num_initialize, safe_search_config.search_L));
            neighborDistance[i].reserve(safe_search_config.search_L);
            threadUsefulset[i].reserve(batch_capacity);
        }
    }

    // Maps BFS IDs to original rows. The prefix holds the query entry points.
    std::vector<int> search_from_ids;

    std::vector<std::vector<SingleNeighbor>> threadRetset; // threadRetset和之前的retset起到的价值差不多
    std::vector<std::vector<SingleNeighbor>> threadUsefulset;
    std::vector<std::vector<float>> neighborDistance;

    void searchSingle(int threadid, int queryid, const CandidateReranker &reranker, const SearchConfig &search_config, int max_degree,
                      const SearchResultView &result) {
        result.validate();
        RNNDESCENT_ASSERT_MSG(max_degree > 0, "search max_degree must be positive");
        const int topk = result.topk();

        const int available_points = static_cast<int>(search_from_ids.size());
        if (available_points <= 0)
            throw std::runtime_error("search entry points are not initialized");
        if (topk > available_points)
            throw std::runtime_error("search topk exceeds indexed point count");
        const SearchConfig safe_search_config = sanitize_search_config(search_config, topk, available_points);
        const int num_threads = safe_search_config.num_threads;
        const int beam_size = safe_search_config.beam_size;
        const int search_L = safe_search_config.search_L;
        const int num_initialize = safe_search_config.num_initialize;
        const int refine_max = safe_search_config.refine_max;
        if (!(0 <= threadid && threadid < num_threads))
            throw std::runtime_error("thread id out of range");
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
#endif
        if (num_initialize < topk)
            throw std::runtime_error("num_initialize must be >= topk");
        if (num_initialize < search_L)
            throw std::runtime_error("num_initialize must be >= search_L");
        RNNDESCENT_ASSERT_MSG(graph_distance_computer != nullptr, "The index is not build yet.");

        auto &retset = threadRetset[threadid];
        auto &usefulset = threadUsefulset[threadid];
        auto &candidate_ids = thread_candidate_ids_[threadid];
        // Score the entry points in batches of four.
        if (num_initialize % 4 != 0)
            throw std::runtime_error("num_initialize must be a multiple of 4");
        retset.resize(num_initialize);
        const size_t batch_capacity = padded_degree(max_degree) * beam_size;
        usefulset.resize(batch_capacity);
        for (int i = 0; i < num_initialize; i += 4) {
            retset[i + 0].id = i + 0;
            retset[i + 1].id = i + 1;
            retset[i + 2].id = i + 2;
            retset[i + 3].id = i + 3;
            graph_distance_computer->distances_batch_4(queryid, i + 0, i + 1, i + 2, i + 3, retset[i].distance, retset[i + 1].distance, retset[i + 2].distance,
                                                       retset[i + 3].distance);
        }

        std::nth_element(retset.begin(), retset.begin() + search_L, retset.end());
        // Keep the active candidate pool sorted by distance.
        std::sort(retset.data(), retset.data() + search_L);
        // Mark admitted IDs until the query ends. A discarded ID cannot pass the shrinking cutoff again.
        candidate_ids.clear();
        for (int i = 0; i < search_L; ++i)
            candidate_ids.insert(retset[i].id);

#ifdef INTERNAL_CLOCK_TEST
        int bfs_length = 0, bfs_items = 0;
        int calcdis = 0;
        nowtime = std::chrono::high_resolution_clock::now();
        init_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        int for_count = 0, calculate_distance_count = 0, useful_count = 0;
#endif

        const auto merge_candidates_from_end = [](SingleNeighbor *pool, int length, const SingleNeighbor *incoming,
                                                   const SingleNeighbor *incoming_end) {
            int kept = 0;
            int upper = std::min(length, static_cast<int>(incoming_end - incoming));
            // The caller skips empty batches. Try the largest possible count first.
            // The last retained new candidate must beat the first evicted old candidate.
            if (incoming[upper - 1] < pool[length - upper])
                kept = upper;
            else
                --upper;

            while (kept < upper) {
                const int middle = kept + (upper - kept + 1) / 2;
                if (incoming[middle - 1] < pool[length - middle])
                    kept = middle;
                else
                    upper = middle - 1;
            }

            auto *current = pool + length - kept;
            auto *out = pool + length;
            const auto *next = incoming + kept;
            // Read and write from the tail. On ties, place the new candidate after the old one.
            while (next != incoming) {
                if (current != pool && next[-1] < current[-1])
                    *--out = *--current;
                else
                    *--out = *--next;
            }
            // The prefix before out stayed in place.
            return static_cast<int>(out - pool);
        };

        int smallest_pos = 0;
        // Continue until every candidate in the active pool has been expanded.
        while (smallest_pos < search_L) {
            SingleNeighbor *usefulsetEnd = usefulset.data();
            for (int beam = 0; smallest_pos < search_L && beam < beam_size; smallest_pos++) {
                if (retset[smallest_pos].id & 0x80000000)
                    continue;
                int n = retset[smallest_pos].id;
                // The high bit marks a candidate as expanded.
                retset[smallest_pos].id |= 0x80000000;
                beam++;

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                getneighbor_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
#endif
                auto &neighbors = final_graph_neighbors[n];

                // Score the full neighborhood before filtering and deduplicating candidates.
                // GIST1M (AVX2, 64D PCA, 16 threads, L=2560, refine=2048):
                // With a 0 GiB neighbor pool, bitmap deduplication before four-way scoring
                // cut INT8 evaluations by 52% and raised QPS by 32% with identical results.
                // With a full neighbor cache, checking the bitmap to skip fully seen groups
                // reduced QPS by 2% on the same workload.
                neighbors.compute_distance(queryid, neighborDistance[threadid]);

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                calculate_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
                bfs_items++;
                for_count += neighbors.size;
                calculate_distance_count += neighbors.size;
#endif

                float limit = retset[search_L - 1].distance;
#pragma unroll
                for (int k = 0; k < neighbors.size; k++) {
                    if (neighborDistance[threadid][k] < limit) {
                        if (!candidate_ids.insert(neighbors.edges[k]))
                            continue;
                        *usefulsetEnd++ = {neighbors.edges[k], neighborDistance[threadid][k]};
                    }
                }
#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
#endif
            }
#ifdef INTERNAL_CLOCK_TEST
            bfs_length++;
            useful_count += usefulsetEnd - usefulset.data();
#endif
            if (usefulsetEnd == usefulset.data())
                continue;
            std::sort(usefulset.data(), usefulsetEnd);

#ifdef INTERNAL_CLOCK_TEST
            nowtime = std::chrono::high_resolution_clock::now();
            update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
#endif

            const int first_changed = merge_candidates_from_end(retset.data(), search_L, usefulset.data(), usefulsetEnd);
            smallest_pos = std::min(smallest_pos, first_changed);
        }

        int refinemax = std::min(refine_max, (int)retset.size());
        retset.resize(refinemax);

        const int *original_ids = nullptr;
        if (reranker.uses_graph_ordered_codes()) {
            // The coarse ranker reads BFS-ordered rows by graph ID.
            for (auto &candidate : retset)
                candidate.id &= 0x7fffffff;
            original_ids = search_from_ids.data();
        } else {
            for (auto &candidate : retset)
                candidate.id = search_from_ids[candidate.id & 0x7fffffff];
        }

        reranker.rerank(queryid, retset, safe_search_config.fp32_refine_max, result, original_ids);

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        float recalculate_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        {
            // 需要atomic加锁
            std::lock_guard<std::mutex> lock(m_mutex);
            perf_stats.query_count++;
            perf_stats.init_time += init_time;
            perf_stats.getneighbor_time += getneighbor_time;
            perf_stats.update_time += update_time;
            perf_stats.calculate_time += calculate_time;
            perf_stats.recalculate_time += recalculate_time;
            perf_stats.calculate_distance_count += calcdis;
            perf_stats.bfs_length += bfs_length;
            perf_stats.for_count += for_count;
            perf_stats.bfs_items += bfs_items;
            perf_stats.calculate_distance_count += calculate_distance_count;
            perf_stats.useful_count += useful_count;
        }
#endif
    }
#ifdef INTERNAL_CLOCK_TEST
    std::mutex m_mutex;
#endif

    void flush_perf_stats() {
#ifdef INTERNAL_CLOCK_TEST
        if (perf_stats.query_count == 0) {
            return;
        }
        Logger::info("RNN Descent Running Time: mean bfs %f times; %f nodes; init_time = "
                     "%f, getneighbor_time = %f, calculate_time = %f, update_time = %f; "
                     "recalculate_time = %f; alltime = %f; calculate_dist_count = %f; "
                     "for_count = %f; useful_count = %f\n",
                     (float)perf_stats.bfs_length / perf_stats.query_count, (float)perf_stats.bfs_items / perf_stats.query_count,
                     perf_stats.init_time / perf_stats.query_count, perf_stats.getneighbor_time / perf_stats.query_count,
                     perf_stats.calculate_time / perf_stats.query_count, perf_stats.update_time / perf_stats.query_count,
                     perf_stats.recalculate_time / perf_stats.query_count,
                     (perf_stats.init_time + perf_stats.getneighbor_time + perf_stats.calculate_time + perf_stats.update_time + perf_stats.recalculate_time) /
                         perf_stats.query_count,
                     (float)perf_stats.calculate_distance_count / perf_stats.query_count, (float)perf_stats.for_count / perf_stats.query_count,
                     (float)perf_stats.useful_count / perf_stats.query_count);
        perf_stats.reset();
#endif
    }

    void reset() {
        Logger::line(true, "Resetting RNNDescent index.");
        std::vector<SelectedNeighborsContainerType>().swap(final_graph_neighbors);
        neighbor_pool_.reset();
        search_from_ids.resize(0);
        thread_candidate_ids_.clear();
        graph_distance_computer.reset();
        flush_perf_stats();
    }

    /// Initialize the KNN graph randomly
    void init_graph(KNNGraph &graph, int n, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        node_locks_.resize(n);
        graph.resize(n);

#pragma omp parallel num_threads(build_config.num_threads)
        {
            std::mt19937 rng(build_config.random_seed * 7741 + omp_get_thread_num());
#pragma omp for
            for (int i = 0; i < n; i++) {
                auto &neighbors = graph[i];
                neighbors.reserve(build_config.S);
                std::vector<int> tmp(build_config.S);

                gen_random(rng, tmp, n);

                for (int j = 0; j < build_config.S; j++) {
                    int id = tmp[j];
                    if (id == i)
                        continue;
                    float dist = qdis.symmetric_dis(i, id);

                    neighbors.emplace_back(id, dist, true);
                }
            }
        }
    }

    void update_neighbors(KNNGraph &graph, int n, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        std::vector<std::vector<XNeighbor>> new_pools(build_config.num_threads);
        std::vector<std::vector<XNeighbor>> old_pools(build_config.num_threads);
        // Reuse small adjacency buffers across passes. Detach larger buffers to shed their excess capacity.
        const size_t reusable_capacity = 2 * static_cast<size_t>(build_config.K0);
#pragma omp parallel for num_threads(build_config.num_threads) schedule(dynamic, 16)
        for (int u = 0; u < n; ++u) {
            auto &pool = graph[u];
            auto &new_pool = new_pools[omp_get_thread_num()];
            auto &old_pool = old_pools[omp_get_thread_num()];
            new_pool.clear();
            std::vector<XNeighbor> retired;
            {
                std::lock_guard<std::mutex> guard(node_locks_[u]);
                if (pool.capacity() > reusable_capacity)
                    retired.swap(pool);
                else
                    old_pool.assign(pool.begin(), pool.end());
                pool.clear();
            }
            auto &candidates = retired.capacity() == 0 ? old_pool : retired;
            detail::sort_and_merge_neighbors(candidates);

            for (auto &nn : candidates) {
                bool ok = true;
                for (auto &other_nn : new_pool) {
                    if (!nn.flag() && !other_nn.flag()) {
                        continue;
                    }
                    if (nn.id() == other_nn.id()) {
                        ok = false;
                        break;
                    }
                    float distance = qdis.symmetric_dis(nn.id(), other_nn.id());
                    if (distance < nn.distance) {
                        ok = false;
                        insert_nn(graph, other_nn.id(), nn.id(), distance, true);
                        break;
                    }
                }
                if (ok) {
                    new_pool.emplace_back(nn);
                }
            }

            for (auto &nn : new_pool) {
                nn.setflag(false);
            }
            {
                std::lock_guard<std::mutex> guard(node_locks_[u]);
                pool.insert(pool.end(), new_pool.begin(), new_pool.end());
            }
        }
    }
    void add_reverse_edges(KNNGraph &graph, int n, const BuildConfig &build_config) {
        std::vector<size_t> offsets(static_cast<size_t>(n) + 1);
        const auto collect_incoming_edges = [&] {
            std::fill(offsets.begin(), offsets.end(), 0);
            for (auto &neighbors : graph) {
                for (auto &neighbor : neighbors)
                    ++offsets[neighbor.id() + 1];
            }
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
            std::vector<XNeighbor> incoming(offsets.back());
            std::vector<size_t> ends(offsets.begin(), offsets.end() - 1);

#pragma omp parallel for num_threads(build_config.num_threads)
            for (int u = 0; u < n; ++u) {
                for (auto &neighbor : graph[u]) {
                    const int v = neighbor.id();
                    std::lock_guard<std::mutex> guard(node_locks_[v]);
                    incoming[ends[v]++] = XNeighbor(u, neighbor.distance, neighbor.flag());
                }
            }
            return incoming;
        };

        {
            const auto incoming = collect_incoming_edges();
#pragma omp parallel num_threads(build_config.num_threads)
            {
                std::vector<XNeighbor> merged;
#pragma omp for
                for (int u = 0; u < n; ++u) {
                    auto &outgoing = graph[u];
                    merged.assign(incoming.begin() + offsets[u], incoming.begin() + offsets[u + 1]);
                    for (auto &neighbor : outgoing) {
                        neighbor.setflag(true);
                        merged.push_back(neighbor);
                    }
                    detail::sort_and_merge_neighbors(merged);
                    const auto count = std::min<size_t>(merged.size(), build_config.R);
                    outgoing.assign(merged.begin(), merged.begin() + count);
                }
            }
        }

        // Transpose the selected edges after releasing the previous incoming-edge buffer.
        auto incoming = collect_incoming_edges();
#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) {
            const auto begin = incoming.begin() + offsets[u];
            const auto end = incoming.begin() + offsets[u + 1];
            std::sort(begin, end);
            const auto count = std::min<size_t>(end - begin, build_config.R);
            graph[u].assign(begin, begin + count);
        }
    }

    void insert_nn(KNNGraph &graph, int node_id, int neighbor_id, float distance, bool flag) {
        // Append under the destination lock. update_neighbors batches sorting and deduplication.
        std::lock_guard<std::mutex> guard(node_locks_[node_id]);
        graph[node_id].emplace_back(neighbor_id, distance, flag);
    }

    std::vector<SelectedNeighborsContainerType> final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> graph_distance_computer;

  private:
    std::vector<int> reorder_by_bfs(const std::vector<int> &edges, const std::vector<size_t> &edge_offsets,
                                    int entry_count, const BuildConfig &build_config, bool verbose) {
        const int n = static_cast<int>(edge_offsets.size() - 1);
        search_from_ids.reserve(n);
        search_from_ids.resize(entry_count);
        if (build_config.random_init) {
            std::mt19937 rng(build_config.random_seed);
            gen_random(rng, search_from_ids, n);
        } else {
            std::iota(search_from_ids.begin(), search_from_ids.end(), 0);
        }

        MyVisitedTable visited(n);
        for (int root : search_from_ids)
            visited.set(root);

        // The row order doubles as the BFS queue. Seed all entry points before visiting any edges.
        for (size_t head = 0, depth = 0; head < search_from_ids.size(); ++depth) {
            const size_t layer_end = search_from_ids.size();
            Logger::info(verbose, "BFS depth %zu: %zu nodes\n", depth, layer_end - head);
            for (; head < layer_end; ++head) {
                const int u = search_from_ids[head];
                for (size_t i = edge_offsets[u]; i < edge_offsets[u + 1]; ++i) {
                    const int v = edges[i];
                    if (!visited.get(v)) {
                        visited.set(v);
                        search_from_ids.push_back(v);
                    }
                }
            }
        }

        const size_t reachable = search_from_ids.size();
        // Keep unreachable rows in original order at the end of the permutation.
        for (int original_id = 0; original_id < n; ++original_id) {
            if (!visited.get(original_id))
                search_from_ids.push_back(original_id);
        }
        Logger::info(verbose, "BFS reached %zu of %d nodes; appended %zu unreachable nodes\n",
                     reachable, n, search_from_ids.size() - reachable);

        std::vector<int> original_to_bfs(n);
        for (int bfs_id = 0; bfs_id < n; ++bfs_id)
            original_to_bfs[search_from_ids[bfs_id]] = bfs_id;
        return original_to_bfs;
    }

    // Adjacency rows are padded for four-way distance scoring.
    static size_t padded_degree(int degree) { return (static_cast<size_t>(degree) + 3) / 4 * 4; }

    std::vector<CandidateBitmap> thread_candidate_ids_;
    SelectedNeighborsContainerType::Pool neighbor_pool_;
    std::vector<MutexWarpper> node_locks_;
};

} // namespace rnndescent

namespace rnndescent {} // namespace rnndescent
