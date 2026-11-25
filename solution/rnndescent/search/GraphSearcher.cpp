#include "GraphSearcher.h"
#include "CandidateReranker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <stdexcept>

namespace rnndescent::search {

GraphSearcher::GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats)
    : storage_(storage),
      code_stride_(storage.graph_distance_computer ? storage.graph_distance_computer->code_stride() : 0),
      padded_max_degree_((static_cast<size_t>(max_degree) + 3) / 4 * 4),
      perf_stats_(perf_stats),
      scratch_(std::max(search_config.num_threads, std::min({search_config.single_query_threads,
                                                           config::SearchConfig::max_single_query_threads,
                                                           search_config.num_initialize / 4}))) {
    RNNDESCENT_ASSERT_MSG(max_degree > 0, "search max_degree must be positive");
    const int beam_size = std::min(search_config.beam_size, search_config.search_L);
    const size_t beam_capacity = padded_max_degree_ * beam_size;
    for (auto &scratch : scratch_) {
        scratch.candidate_ids.initialize(storage_.final_graph_neighbors.size());
        scratch.reserve(search_config.num_initialize, beam_size, beam_capacity);
    }
}

void GraphSearcher::search_single_thread(int threadid, int queryid, const config::SearchConfig &search_config,
                                         const CandidateReranker &reranker, const SearchResultView &result) {
    const auto safe_config = checked_search_config(search_config, result);
    if (!(0 <= threadid && threadid < safe_config.num_threads && static_cast<size_t>(threadid) < scratch_.size()))
        throw std::runtime_error("thread id out of range");
    auto &candidates = search_from_entries<false>(threadid, queryid, safe_config, 0);
    rerank_serial(threadid, queryid, candidates, safe_config.fp32_refine_max, reranker, result);
}

config::SearchConfig GraphSearcher::checked_search_config(const config::SearchConfig &search_config,
                                                          const SearchResultView &result) const {
    result.validate();
    const int topk = result.topk();
    const int available_points = static_cast<int>(storage_.final_graph_neighbors.size());
    if (available_points <= 0)
        throw std::runtime_error("search entry points are not initialized");
    if (topk > available_points)
        throw std::runtime_error("search topk exceeds indexed point count");
    auto safe_config = config::sanitize_search_config(search_config, topk, available_points);
    if (safe_config.num_initialize < topk)
        throw std::runtime_error("num_initialize must be >= topk");
    if (safe_config.num_initialize % 4 != 0)
        throw std::runtime_error("num_initialize must be a multiple of 4");
    RNNDESCENT_ASSERT_MSG(storage_.graph_distance_computer != nullptr, "The index is not build yet.");
    return safe_config;
}

void GraphSearcher::search_parallel(int queryid, const config::SearchConfig &search_config,
                                    const CandidateReranker &reranker, const SearchResultView &result) {
    const int topk = result.topk();
    const auto safe_config = checked_search_config(search_config, result);
    const int groups = safe_config.num_initialize / 4;
    const int groups_per_topk = (topk - 1) / 4 + 1;
    const int workers = std::min({safe_config.single_query_threads, static_cast<int>(scratch_.size()), groups / groups_per_topk});
    if (workers <= 1) {
        auto &candidates = search_from_entries<false>(0, queryid, safe_config, 0);
        rerank_serial(0, queryid, candidates, safe_config.fp32_refine_max, reranker, result);
        return;
    }

    if (!shared_visited_)
        shared_visited_ = std::make_unique<SharedVisitedSet>(storage_.final_graph_neighbors.size());
    shared_visited_->begin_query();

    // Reserve for the largest shard in case OpenMP runs with fewer workers than requested.
    const int beam = std::min(safe_config.beam_size, safe_config.search_L);
    const size_t beam_capacity = padded_max_degree_ * beam;
    for (int worker = 0; worker < workers; ++worker)
        scratch_[worker].reserve(safe_config.num_initialize, beam, beam_capacity);
    merged_candidates_.reserve(safe_config.refine_max);
    // The total length of the worker lists cannot exceed the number of entries.
    if (merge_buffer_.size() < static_cast<size_t>(safe_config.num_initialize))
        merge_buffer_.resize(safe_config.num_initialize);

    int active_workers = 1;
    size_t fp32_count = 0;
    const auto limits = safe_config.single_query_limits;
    const int *const row_order = storage_.original_ids ? storage_.original_ids->data() : nullptr;
#ifdef INTERNAL_CLOCK_TEST
    std::chrono::steady_clock::time_point rerank_start;
#endif
#pragma omp parallel num_threads(workers)
    {
        const int worker = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        // Give each worker a disjoint entry range made up of whole four-way scoring groups.
        const int groups_per_worker = groups / team_size;
        const int remainder = groups % team_size;
        const int first_entry = 4 * (worker * groups_per_worker + std::min(worker, remainder));
        auto local_config = safe_config;
        local_config.num_initialize = 4 * (groups_per_worker + (worker < remainder));
        const int local_width = safe_config.search_L / team_size + (safe_config.search_L % team_size != 0);
        local_config.search_L = std::clamp(local_width, topk, local_config.num_initialize);
        local_config.refine_max = std::min(safe_config.refine_max, local_config.num_initialize);
        if (team_size > 1 && limits.candidates_per_worker > 0)
            local_config.refine_max = std::min(local_config.refine_max, std::max(topk, limits.candidates_per_worker));
        if (team_size > 1 && limits.expansions_per_worker > 0)
            search_from_entries<true, true>(worker, queryid, local_config, first_entry);
        else
            search_from_entries<true>(worker, queryid, local_config, first_entry);
        if (team_size > 1) {
            // The merge reads every worker's candidate pool.
#pragma omp barrier
            if (worker == 0) {
                active_workers = team_size;
                merge_worker_candidates(team_size, safe_config.refine_max);
#ifdef INTERNAL_CLOCK_TEST
                rerank_start = std::chrono::steady_clock::now();
#endif
                fp32_count = reranker.prepare_shortlist(merged_candidates_, safe_config.fp32_refine_max, topk);
            }
            // Workers need the merged pool and FP32 limit before scoring it.
#pragma omp barrier
            reranker.score_for_rerank(queryid, merged_candidates_, fp32_count, row_order, worker, team_size);
        }
    }
    if (active_workers == 1) {
        rerank_serial(0, queryid, scratch_.front().retset, safe_config.fp32_refine_max, reranker, result);
        return;
    }
    CandidateReranker::write_topk(merged_candidates_, result);
#ifdef INTERNAL_CLOCK_TEST
    scratch_.front().perf_stats.recalculate_time +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - rerank_start).count();
#endif
}

void GraphSearcher::rerank_serial([[maybe_unused]] int threadid, int queryid, std::vector<SingleNeighbor> &candidates,
                                  int fp32_limit, const CandidateReranker &reranker, const SearchResultView &result) {
#ifdef INTERNAL_CLOCK_TEST
    const auto start = std::chrono::steady_clock::now();
#endif
    const int *const row_order = storage_.original_ids ? storage_.original_ids->data() : nullptr;
    reranker.rerank(queryid, candidates, fp32_limit, result, row_order);
#ifdef INTERNAL_CLOCK_TEST
    scratch_[threadid].perf_stats.recalculate_time +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
#endif
}

void GraphSearcher::merge_worker_candidates(int workers, int refine_max) {
    auto &candidate_ids = scratch_.front().candidate_ids;
    candidate_ids.clear();
    auto *const begin = merge_buffer_.data();
    auto *end = begin;
    // A duplicate leaves the write pointer in place, so the next candidate overwrites it.
    for (int worker = 0; worker < workers; ++worker) {
        for (const auto &candidate : scratch_[worker].retset) {
            *end = candidate;
            end += candidate_ids.test_and_set(candidate.id);
        }
    }
    if (end - begin > refine_max) {
        std::nth_element(begin, begin + refine_max, end);
        end = begin + refine_max;
    }
    merged_candidates_.assign(begin, end);
}

template <bool SharedVisit, bool LimitExpansions>
std::vector<SingleNeighbor> &GraphSearcher::search_from_entries(int threadid, int queryid, const config::SearchConfig &search_config,
                                                                int entry_begin) {
        const int search_L = search_config.search_L;
        const int beam_size = std::min(search_config.beam_size, search_L);
        const int num_initialize = search_config.num_initialize;
        const int refine_max = search_config.refine_max;
        const int pool_size = std::max(search_L, refine_max);
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
#endif
        auto &distance = *storage_.graph_distance_computer;
        const auto code_stride = code_stride_;
        auto &scratch = scratch_[threadid];
        auto &retset = scratch.retset;
        auto &usefulset = scratch.usefulset;
        auto &candidate_ids = scratch.candidate_ids;
        retset.resize(num_initialize);
        const size_t beam_capacity = padded_max_degree_ * beam_size;
        usefulset.resize(beam_capacity);
        scratch.scores.resize(beam_capacity);
        scratch.masks.resize(detail::score_mask_count(beam_capacity));
        scratch.spans.resize(beam_size);
        float *const neighbor_scores = scratch.scores.data();
        auto *const neighbor_masks = scratch.masks.data();
        auto *const neighbor_spans = scratch.spans.data();
        for (int i = 0; i < num_initialize; ++i)
            retset[i].id = entry_begin + i;
        distance.score_candidates(queryid, retset.data(), retset.size());

        std::nth_element(retset.begin(), retset.begin() + pool_size, retset.end());
        // Keep a sorted tail for reranking when refine_max exceeds the expansion width.
        std::sort(retset.data(), retset.data() + pool_size);
        // Mark admitted IDs until the query ends. A discarded ID cannot pass the shrinking cutoff again.
        if constexpr (SharedVisit)
            candidate_ids.begin_lazy_query();
        else
            candidate_ids.clear();
        for (int i = 0; i < pool_size; ++i)
            candidate_ids.insert<SharedVisit>(retset[i].id);

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
        int remaining_expansions = search_config.single_query_limits.expansions_per_worker;
        // Only the closest search_L candidates are eligible for expansion.
        while (smallest_pos < search_L && (!LimitExpansions || remaining_expansions > 0)) {
            const int beam_limit = LimitExpansions ? std::min(beam_size, remaining_expansions) : beam_size;
            size_t score_count = 0;
            auto *next_span = neighbor_spans;
            for (int beam = 0; smallest_pos < search_L && beam < beam_limit; smallest_pos++) {
                if (retset[smallest_pos].id & 0x80000000)
                    continue;
                int n = retset[smallest_pos].id;
                // The high bit marks a candidate as expanded.
                retset[smallest_pos].id |= 0x80000000;
                if constexpr (SharedVisit) {
                    if (!shared_visited_->insert(n))
                        continue;
                }
                beam++;
                if constexpr (LimitExpansions)
                    --remaining_expansions;

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                getneighbor_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
#endif
                // Overlap the next record's header load with scoring this neighborhood.
                if (beam < beam_limit && smallest_pos + 1 < search_L) {
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
                if (candidate_ids.insert<SharedVisit>(candidate->id))
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
        auto &stats = scratch.perf_stats;
        if (!SharedVisit || threadid == 0)
            stats.query_count++;
        stats.init_time += init_time;
        stats.getneighbor_time += getneighbor_time;
        stats.update_time += update_time;
        stats.calculate_time += calculate_time;
        stats.bfs_length += bfs_length;
        stats.bfs_items += bfs_items;
        stats.calculate_distance_count += calculate_distance_count;
        stats.useful_count += useful_count;
#endif
        return retset;
    }

void GraphSearcher::flush_perf_stats() {
#ifdef INTERNAL_CLOCK_TEST
        for (auto &scratch : scratch_) {
            perf_stats_.accumulate(scratch.perf_stats);
            scratch.perf_stats.reset();
        }
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
