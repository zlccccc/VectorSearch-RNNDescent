#include "GraphSearcher.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace rnndescent::search {


GraphSearcher::GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats)
    : storage_(storage), num_threads_(search_config.num_threads), max_degree_(max_degree), perf_stats_(perf_stats) {
    candidate_ids_.resize(num_threads_);
    retset_.resize(num_threads_);
    neighbor_distance_.resize(num_threads_);
    neighbor_ids_.resize(num_threads_);
    neighbor_spans_.resize(num_threads_);
    usefulset_.resize(num_threads_);
    const int beam_size = std::min(search_config.beam_size, search_config.search_L);
    const size_t beam_capacity = padded_degree(max_degree) * beam_size;
    for (int i = 0; i < num_threads_; ++i) {
        candidate_ids_[i].initialize(static_cast<int>(storage_.final_graph_neighbors.size()));
        retset_[i].reserve(std::max(search_config.num_initialize, search_config.search_L));
        neighbor_distance_[i].reserve(beam_capacity);
        neighbor_ids_[i].reserve(beam_capacity);
        neighbor_spans_[i].reserve(beam_size);
        usefulset_[i].reserve(beam_capacity);
    }
}

std::vector<SingleNeighbor> &GraphSearcher::search_candidates(int threadid, int queryid,
    const config::SearchConfig &search_config, int max_degree, int topk) {
        RNNDESCENT_ASSERT_MSG(max_degree > 0, "search max_degree must be positive");

        const int available_points = static_cast<int>(storage_.final_graph_neighbors.size());
        if (available_points <= 0)
            throw std::runtime_error("search entry points are not initialized");
        if (topk > available_points)
            throw std::runtime_error("search topk exceeds indexed point count");
        const config::SearchConfig safe_search_config = config::sanitize_search_config(search_config, topk, available_points);
        const int num_threads = safe_search_config.num_threads;
        const int search_L = safe_search_config.search_L;
        const int beam_size = std::min(safe_search_config.beam_size, search_L);
        const int num_initialize = safe_search_config.num_initialize;
        const int refine_max = safe_search_config.refine_max;
        if (!(0 <= threadid && threadid < num_threads && threadid < num_threads_))
            throw std::runtime_error("thread id out of range");
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
#endif
        if (num_initialize < topk)
            throw std::runtime_error("num_initialize must be >= topk");
        if (num_initialize < search_L)
            throw std::runtime_error("num_initialize must be >= search_L");
        RNNDESCENT_ASSERT_MSG(storage_.graph_distance_computer != nullptr, "The index is not build yet.");

        auto &retset = retset_[threadid];
        auto &usefulset = usefulset_[threadid];
        auto &candidate_ids = candidate_ids_[threadid];
        // Score the entry points in batches of four.
        if (num_initialize % 4 != 0)
            throw std::runtime_error("num_initialize must be a multiple of 4");
        retset.resize(num_initialize);
        const size_t beam_capacity = padded_degree(max_degree) * beam_size;
        usefulset.resize(beam_capacity);
        auto &score_buffer = neighbor_distance_[threadid];
        auto &id_buffer = neighbor_ids_[threadid];
        auto &span_buffer = neighbor_spans_[threadid];
        score_buffer.resize(beam_capacity);
        id_buffer.resize(beam_capacity);
        span_buffer.resize(beam_size);
        float *const neighbor_scores = score_buffer.data();
        int *const neighbor_ids = id_buffer.data();
        auto *const neighbor_spans = span_buffer.data();
        for (int i = 0; i < num_initialize; i += 4) {
            retset[i + 0].id = i + 0;
            retset[i + 1].id = i + 1;
            retset[i + 2].id = i + 2;
            retset[i + 3].id = i + 3;
            storage_.graph_distance_computer->distances_batch_4(queryid, i + 0, i + 1, i + 2, i + 3, retset[i].distance, retset[i + 1].distance, retset[i + 2].distance,
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
        nowtime = std::chrono::high_resolution_clock::now();
        init_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        long long calculate_distance_count = 0, useful_count = 0;
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
            size_t score_count = 0;
            int span_count = 0;
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
                auto &neighbors = storage_.final_graph_neighbors[n];

                // Score the full neighborhood before filtering and deduplicating candidates.
                // GIST1M (AVX2, 64D PCA, 16 threads, L=2560, refine=2048):
                // With a 0 GiB neighbor pool, bitmap deduplication before four-way scoring
                // cut INT8 evaluations by 52% and raised QPS by 32% with identical results.
                // With a full neighbor cache, checking the bitmap to skip fully seen groups
                // reduced QPS by 2% on the same workload.
                neighbors.compute_distance(queryid, neighbor_scores + score_count);

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                calculate_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
                bfs_items++;
                calculate_distance_count += neighbors.size;
#endif

                if (neighbors.size > 0) {
                    neighbor_spans[span_count++] = {neighbors.edges, neighbors.size};
                    score_count += neighbors.size;
                }
            }
#ifdef INTERNAL_CLOCK_TEST
            bfs_length++;
#endif
            if (score_count == 0)
                continue;
            // Copy IDs in score order after all neighborhoods in the beam are scored.
            auto *out = neighbor_ids;
            for (int i = 0; i < span_count; ++i) {
                const auto &span = neighbor_spans[i];
                std::memcpy(out, span.ids, span.count * sizeof(int));
                out += span.count;
            }
#ifdef INTERNAL_CLOCK_TEST
            nowtime = std::chrono::high_resolution_clock::now();
            getneighbor_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
#endif
            // The cutoff stays fixed during the beam. Keep the adjacency order for deduplication.
            auto *const usefulsetEnd = detail::append_neighbors_below_cutoff(
                neighbor_scores, neighbor_ids, score_count,
                retset[search_L - 1].distance, candidate_ids, usefulset.data());
            if (usefulsetEnd != usefulset.data()) {
                std::sort(usefulset.data(), usefulsetEnd);
                const int first_changed = merge_candidates_from_end(retset.data(), search_L, usefulset.data(), usefulsetEnd);
                smallest_pos = std::min(smallest_pos, first_changed);
            }
#ifdef INTERNAL_CLOCK_TEST
            nowtime = std::chrono::high_resolution_clock::now();
            update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
            useful_count += usefulsetEnd - usefulset.data();
#endif
        }
#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        getneighbor_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
#endif

        retset.resize(std::min(refine_max, search_L));
        for (auto &candidate : retset)
            candidate.id &= 0x7fffffff;

#ifdef INTERNAL_CLOCK_TEST
        {
            std::lock_guard<std::mutex> lock(mutex_);
            perf_stats_.query_count++;
            perf_stats_.init_time += init_time;
            perf_stats_.getneighbor_time += getneighbor_time;
            perf_stats_.update_time += update_time;
            perf_stats_.calculate_time += calculate_time;
            perf_stats_.bfs_length += bfs_length;
            perf_stats_.bfs_items += bfs_items;
            perf_stats_.calculate_distance_count += calculate_distance_count;
            perf_stats_.useful_count += useful_count;
        }
#endif
        return retset;
    }

#ifdef INTERNAL_CLOCK_TEST
void GraphSearcher::record_refinement_time(float elapsed_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    perf_stats_.recalculate_time += elapsed_ms;
}
#endif

void GraphSearcher::flush_perf_stats() {
#ifdef INTERNAL_CLOCK_TEST
        if (perf_stats_.query_count == 0) {
            return;
        }
        Logger::info("RNN Descent Running Time: mean bfs %f times; %f nodes; init_time = "
                     "%f, getneighbor_time = %f, calculate_time = %f, update_time = %f; "
                     "recalculate_time = %f; alltime = %f; calculate_dist_count = %f; "
                     "useful_count = %f\n",
                     (float)perf_stats_.bfs_length / perf_stats_.query_count, (float)perf_stats_.bfs_items / perf_stats_.query_count,
                     perf_stats_.init_time / perf_stats_.query_count, perf_stats_.getneighbor_time / perf_stats_.query_count,
                     perf_stats_.calculate_time / perf_stats_.query_count, perf_stats_.update_time / perf_stats_.query_count,
                     perf_stats_.recalculate_time / perf_stats_.query_count,
                     (perf_stats_.init_time + perf_stats_.getneighbor_time + perf_stats_.calculate_time + perf_stats_.update_time +
                      perf_stats_.recalculate_time) /
                         perf_stats_.query_count,
                     (float)perf_stats_.calculate_distance_count / perf_stats_.query_count,
                     (float)perf_stats_.useful_count / perf_stats_.query_count);
        perf_stats_.reset();
#endif
    }


} // namespace rnndescent::search
