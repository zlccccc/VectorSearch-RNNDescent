#include "GraphSearcher.h"
#include "CandidateReranker.h"

#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
#include <utility>

namespace rnndescent::search {

GraphSearcher::GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats)
    : storage_(storage),
      code_stride_(storage.graph_distance_computer ? storage.graph_distance_computer->code_stride() : 0),
      padded_max_degree_((static_cast<size_t>(max_degree) + 3) / 4 * 4),
      scratch_(std::max(search_config.num_threads, std::min({search_config.single_query_threads,
                                                           config::SearchConfig::max_single_query_threads,
                                                           search_config.num_initialize / 4}))),
      profile_(scratch_.size(), perf_stats) {
    RNNDESCENT_ASSERT_MSG(max_degree > 0, "search max_degree must be positive");
    const int beam_size = std::min(search_config.beam_size, search_config.search_L);
    const size_t beam_capacity = padded_max_degree_ * beam_size;
    for (auto &scratch : scratch_) {
        scratch.candidate_ids.initialize(storage_.final_graph_neighbors.size());
        scratch.reserve(search_config.num_initialize, beam_size, beam_capacity);
    }
}

QueryWorkers *GraphSearcher::persistent_workers(int active) {
    if (!QueryWorkers::can_run(active))
        return nullptr;
    if (!query_workers_ || !query_workers_->matches(active)) {
        auto team = QueryWorkers::create(active, static_cast<int>(scratch_.size()));
        if (!team)
            return nullptr;
        query_workers_ = std::move(team);
    }
    return query_workers_.get();
}

void GraphSearcher::search_batch(int query_count, const int *query_order, const config::SearchConfig &search_config,
                                 const CandidateReranker &reranker, const SearchResultView &result) {
    RNNDESCENT_ASSERT_MSG(query_count > 0, "search query count must be positive");
    const auto safe_config = checked_search_config(search_config, result);
    const int workers = std::min(safe_config.num_threads, static_cast<int>(scratch_.size()));
    const int topk = result.topk();
    auto *const ids = result.indices_ptr();
    auto *const distances = result.distances_ptr();
    const auto search_row = [&](int position, int worker) {
        const int id = query_order ? query_order[position] : position;
        const size_t offset = static_cast<size_t>(id) * topk;
        search_from_entries<false>(worker, id, safe_config, 0, reranker, {ids + offset, distances + offset, topk});
    };

    if (query_count == 1 || workers == 1) {
        for (int i = 0; i < query_count; ++i)
            search_row(i, 0);
        return;
    }
    if (safe_config.persistent_query_workers && query_count <= workers) {
        if (auto *team = persistent_workers(query_count)) {
            const ProfileTimer launch{};
            auto work = [&](int worker) {
                auto &sample = profile_.start_worker(worker, query_count);
                search_row(worker, worker);
                sample.job_finished();
                sample.stop();
            };
            team->run(query_count, work);
            profile_.finish_team(SearchTeam::Batch, launch);
            return;
        }
    }
    constexpr int queries_per_chunk = 4;
    // With only a few queries per worker, a slow query can dominate a static chunk.
    const bool balance_small_batch = query_count > workers && query_count <= 16;
    const bool static_schedule = !balance_small_batch && (query_count - 1) / queries_per_chunk < workers;
    const int chunk = balance_small_batch ? 1 : queries_per_chunk;
    // Record completion before the parallel region's implicit join.
    const ProfileTimer launch{};
#pragma omp parallel num_threads(workers)
    {
        const int worker = omp_get_thread_num();
        auto &sample = profile_.start_worker(worker);
        if (static_schedule) {
#pragma omp for schedule(static) nowait
            for (int i = 0; i < query_count; ++i) {
                search_row(i, worker);
                sample.job_finished();
            }
        } else {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < query_count; ++i) {
                search_row(i, worker);
                sample.job_finished();
            }
        }
        sample.stop();
    }
    profile_.finish_team(SearchTeam::Batch, launch);
}

void GraphSearcher::search_single_thread(int threadid, int queryid, const config::SearchConfig &search_config,
                                         const CandidateReranker &reranker, const SearchResultView &result) {
    const auto safe_config = checked_search_config(search_config, result);
    if (!(0 <= threadid && threadid < safe_config.num_threads && static_cast<size_t>(threadid) < scratch_.size()))
        throw std::runtime_error("thread id out of range");
    search_from_entries<false>(threadid, queryid, safe_config, 0, reranker, result);
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
    ProfileTimer setup{};
    const int topk = result.topk();
    const auto safe_config = checked_search_config(search_config, result);
    const int groups = safe_config.num_initialize / 4;
    const int groups_per_topk = (topk - 1) / 4 + 1;
    const int workers = std::min({safe_config.single_query_threads, static_cast<int>(scratch_.size()), groups / groups_per_topk});
    if (workers <= 1) {
        search_from_entries<false>(0, queryid, safe_config, 0, reranker, result);
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
    merged_candidates_.reserve(static_cast<size_t>(workers) * topk);

    int active_workers = 1;
    const auto overrides = safe_config.single_query_limits;
    const bool use_shortlist = safe_config.uses_coarse_shortlist(safe_config.refine_max);
    const int local_fp32_limit = overrides.candidates_per_worker > 0
                                     ? overrides.candidates_per_worker
                                     : config::SearchConfig::SingleQueryLimits::default_quantized_candidates;
    const auto work = [&](int worker, int team_size) {
        auto &sample = profile_.start_worker(worker, team_size);
        if (worker == 0)
            active_workers = team_size;
        if (team_size == 1) {
            search_from_entries<false>(0, queryid, safe_config, 0, reranker, result);
        } else {
            // Give each worker a disjoint entry range made up of whole four-way scoring groups.
            const int groups_per_worker = groups / team_size;
            const int remainder = groups % team_size;
            const int first_entry = 4 * (worker * groups_per_worker + std::min(worker, remainder));
            auto local_config = safe_config;
            local_config.num_initialize = 4 * (groups_per_worker + (worker < remainder));
            // Split the query's search and rerank widths. Each worker uses the configured beam for its own rounds.
            const int local_width = (safe_config.search_L - 1) / team_size + 1;
            local_config.search_L = std::clamp(local_width, topk, local_config.num_initialize);
            const int local_refine = (safe_config.refine_max - 1) / team_size + 1;
            local_config.refine_max = std::min({safe_config.refine_max, local_config.num_initialize,
                                                std::max(local_config.search_L, local_refine)});
            local_config.fp32_refine_max = use_shortlist ? local_fp32_limit : 0;
            if (!use_shortlist && overrides.candidates_per_worker > 0)
                local_config.refine_max = std::min(local_config.refine_max, std::max(topk, overrides.candidates_per_worker));

            if (overrides.expansions_per_worker > 0)
                search_from_entries<true, true>(worker, queryid, local_config, first_entry, reranker, result);
            else
                search_from_entries<true>(worker, queryid, local_config, first_entry, reranker, result);
        }
        sample.job_finished();
        sample.stop();
    };

    auto *const team = safe_config.persistent_query_workers ? persistent_workers(workers) : nullptr;
    profile_.single_setup_finished(setup);
    if (team) {
        auto work_shard = [&](int worker) { work(worker, workers); };
        team->run(workers, work_shard);
    } else {
#pragma omp parallel num_threads(workers)
        work(omp_get_thread_num(), omp_get_num_threads());
    }
    profile_.finish_team(SearchTeam::Single, setup);
    if (active_workers == 1)
        return;
    ProfileTimer merge{};
    auto &candidate_ids = scratch_.front().candidate_ids;
    candidate_ids.begin_lazy_query();
    merged_candidates_.clear();
    for (int worker = 0; worker < active_workers; ++worker) {
        for (const auto &candidate : scratch_[worker].retset) {
            if (candidate_ids.insert<true>(candidate.id))
                merged_candidates_.push_back(candidate);
        }
    }
    CandidateReranker::write_topk(merged_candidates_, result);
    profile_.single_merge_finished(merge);
}

template <bool SharedVisit, bool LimitExpansions>
void GraphSearcher::search_from_entries(int threadid, int queryid, const config::SearchConfig &search_config,
                                        int entry_begin, const CandidateReranker &reranker, const SearchResultView &result) {
        const int search_L = search_config.search_L;
        const int beam_size = std::min(search_config.beam_size, search_L);
        const int num_initialize = search_config.num_initialize;
        const int refine_max = search_config.refine_max;
        const int pool_size = std::max(search_L, refine_max);
        auto sample = profile_.query(threadid);
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

        sample.initialized();

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

                sample.adjacency_ready();
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

                sample.neighbors_scored(count);

                score_count += count;
                *next_span++ = {ids, score_count};
            }
            sample.beam_finished();
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
            sample.candidates_updated(usefulsetEnd - usefulset.data());
        }
        sample.adjacency_ready();

        retset.resize(refine_max);
        for (auto &candidate : retset)
            candidate.id &= 0x7fffffff;

        sample.rerank_started();
        const int *const row_order = storage_.original_ids ? storage_.original_ids->data() : nullptr;
        reranker.rerank(queryid, retset, search_config.fp32_refine_max, result.topk(), row_order);
        // A shared-visit worker leaves its Top K in retset for the final merge.
        if constexpr (!SharedVisit)
            CandidateReranker::write_topk(retset, result);
        sample.finish(!SharedVisit || threadid == 0);
    }
} // namespace rnndescent::search
