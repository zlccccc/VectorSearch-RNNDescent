#include "GraphSearcher.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace rnndescent::search {

GraphSearcher::GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats)
    : storage_(storage),
      code_stride_(storage.graph_distance_computer ? storage.graph_distance_computer->code_stride() : 0),
      perf_stats_(perf_stats), scratch_(search_config.num_threads) {
    const int beam_size = std::min(search_config.beam_size, search_config.search_L);
    const size_t beam_capacity = padded_degree(max_degree) * beam_size;
    for (auto &scratch : scratch_) {
        scratch.candidate_ids.initialize(storage_.final_graph_neighbors.size());
        scratch.retset.reserve(search_config.num_initialize);
        scratch.usefulset.reserve(beam_capacity);
        scratch.scores.reserve(beam_capacity);
        scratch.masks.reserve(detail::score_mask_count(beam_capacity));
        scratch.spans.reserve(beam_size);
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
        const int pool_size = std::max(search_L, refine_max);
        if (!(0 <= threadid && threadid < num_threads && static_cast<size_t>(threadid) < scratch_.size()))
            throw std::runtime_error("thread id out of range");
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
#endif
        if (num_initialize < topk)
            throw std::runtime_error("num_initialize must be >= topk");
        RNNDESCENT_ASSERT_MSG(storage_.graph_distance_computer != nullptr, "The index is not build yet.");

        auto &distance = *storage_.graph_distance_computer;
        const auto code_stride = code_stride_;
        auto &scratch = scratch_[threadid];
        auto &retset = scratch.retset;
        auto &usefulset = scratch.usefulset;
        auto &candidate_ids = scratch.candidate_ids;
        // Score the entry points in batches of four.
        if (num_initialize % 4 != 0)
            throw std::runtime_error("num_initialize must be a multiple of 4");
        retset.resize(num_initialize);
        const size_t beam_capacity = padded_degree(max_degree) * beam_size;
        usefulset.resize(beam_capacity);
        scratch.scores.resize(beam_capacity);
        scratch.masks.resize(detail::score_mask_count(beam_capacity));
        scratch.spans.resize(beam_size);
        float *const neighbor_scores = scratch.scores.data();
        auto *const neighbor_masks = scratch.masks.data();
        auto *const neighbor_spans = scratch.spans.data();
        for (int i = 0; i < num_initialize; i += 4) {
            retset[i + 0].id = i + 0;
            retset[i + 1].id = i + 1;
            retset[i + 2].id = i + 2;
            retset[i + 3].id = i + 3;
            distance.distances_batch_4(queryid, i + 0, i + 1, i + 2, i + 3,
                                       retset[i].distance, retset[i + 1].distance, retset[i + 2].distance,
                                       retset[i + 3].distance);
        }

        std::nth_element(retset.begin(), retset.begin() + pool_size, retset.end());
        // Keep a sorted tail for reranking when refine_max exceeds the expansion width.
        std::sort(retset.data(), retset.data() + pool_size);
        // Mark admitted IDs until the query ends. A discarded ID cannot pass the shrinking cutoff again.
        candidate_ids.clear();
        for (int i = 0; i < pool_size; ++i)
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
        // Only the closest search_L candidates are eligible for expansion.
        while (smallest_pos < search_L) {
            size_t score_count = 0;
            auto *next_span = neighbor_spans;
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
                // Overlap the next record's header load with scoring this neighborhood.
                if (beam < beam_size && smallest_pos + 1 < search_L) {
                    const int next = retset[smallest_pos + 1].id;
                    if (next >= 0)
                        storage_.final_graph_neighbors[next].prefetch_header();
                }
                // Score the full neighborhood before filtering and deduplicating candidates.
                // GIST1M (AVX2, 64D PCA, 16 threads, L=2560, refine=2048):
                // With a 0 GiB neighbor pool, bitmap deduplication before four-way scoring
                // cut INT8 evaluations by 52% and raised QPS by 32% with identical results.
                // With a full neighbor cache, checking the bitmap to skip fully seen groups
                // reduced QPS by 2% on the same workload.
                const auto [ids, count] = storage_.final_graph_neighbors[n].compute_distance(
                    queryid, distance, code_stride, neighbor_scores + score_count);

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                calculate_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
                bfs_items++;
                calculate_distance_count += count;
#endif

                score_count += count;
                *next_span++ = {ids, score_count};
            }
#ifdef INTERNAL_CLOCK_TEST
            bfs_length++;
#endif
            if (score_count == 0)
                continue;
            // Scan the scores in one pass, then read only the IDs selected by the masks.
            const float cutoff = retset[pool_size - 1].distance;
            detail::make_neighbor_score_masks(neighbor_scores, score_count, cutoff, neighbor_masks);
            const auto *passing_end = detail::collect_neighbors_from_masks(
                neighbor_scores, score_count, neighbor_masks, neighbor_spans, usefulset.data());

            // Compact in adjacency order before sorting the new candidates.
            auto *usefulsetEnd = usefulset.data();
            for (const auto *candidate = usefulset.data(); candidate != passing_end; ++candidate) {
                if (candidate_ids.insert(candidate->id))
                    *usefulsetEnd++ = *candidate;
            }
            if (usefulsetEnd != usefulset.data()) {
                std::sort(usefulset.data(), usefulsetEnd);
                const int first_changed = merge_candidates_from_end(retset.data(), pool_size, usefulset.data(), usefulsetEnd);
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

        retset.resize(refine_max);
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
