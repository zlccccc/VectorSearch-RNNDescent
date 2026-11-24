#pragma once

#include "../build/GraphConfig.h"
#include "../build/GraphTypes.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"
#include "CandidateBitmap.h"
#include "NeighborScoreFilter.h"

#include <memory>
#include <vector>

namespace rnndescent {
class CandidateReranker;
struct SearchResultView;

namespace search {

#ifdef INTERNAL_CLOCK_TEST
struct PerfStats {
    float init_time{0};
    float getneighbor_time{0};
    float calculate_time{0};
    float update_time{0};
    float recalculate_time{0};
    long long query_count{0};
    long long calculate_distance_count{0};
    long long bfs_length{0};
    long long useful_count{0};
    long long bfs_items{0};

    void accumulate(const PerfStats &other) {
        init_time += other.init_time;
        getneighbor_time += other.getneighbor_time;
        calculate_time += other.calculate_time;
        update_time += other.update_time;
        recalculate_time += other.recalculate_time;
        query_count += other.query_count;
        calculate_distance_count += other.calculate_distance_count;
        bfs_length += other.bfs_length;
        useful_count += other.useful_count;
        bfs_items += other.bfs_items;
    }

    void reset() { *this = {}; }
};
#else
struct PerfStats {
    void reset() {}
};
#endif

struct Storage {
    const std::vector<SelectedNeighborsContainerType> &final_graph_neighbors;
    const std::unique_ptr<MyDistanceComputer> &graph_distance_computer;
    const std::vector<int> *original_ids = nullptr;
};

class GraphSearcher {
  public:
    GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats);

    void search_single_thread(int threadid, int queryid, const config::SearchConfig &search_config,
                              const CandidateReranker &reranker, const SearchResultView &result);
    void search_parallel(int queryid, const config::SearchConfig &search_config,
                         const CandidateReranker &reranker, const SearchResultView &result);
    void flush_perf_stats();

  private:
    struct ThreadScratch {
        CandidateBitmap candidate_ids;
        std::vector<SingleNeighbor> retset;
        std::vector<SingleNeighbor> usefulset;
        std::vector<float, AlignedAllocator<float>> scores;
        std::vector<uint32_t, AlignedAllocator<uint32_t>> masks;
        std::vector<detail::NeighborIdSpan, AlignedAllocator<detail::NeighborIdSpan>> spans;
#ifdef INTERNAL_CLOCK_TEST
        PerfStats perf_stats;
#endif

        void reserve(int entries, int beam, size_t score_count) {
            retset.reserve(entries);
            usefulset.reserve(score_count);
            scores.reserve(score_count);
            masks.reserve(detail::score_mask_count(score_count));
            spans.reserve(beam);
        }
    };

    config::SearchConfig checked_search_config(const config::SearchConfig &search_config, const SearchResultView &result) const;

    template <bool SharedVisit>
    std::vector<SingleNeighbor> &search_from_entries(int threadid, int queryid, const config::SearchConfig &config,
                                                    int entry_begin);
    void merge_worker_candidates(int workers, int refine_max);
    void rerank_serial(int threadid, int queryid, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                       const CandidateReranker &reranker, const SearchResultView &result);

    Storage storage_;
    const uint32_t code_stride_;
    const size_t padded_max_degree_;
    PerfStats &perf_stats_;
    std::vector<ThreadScratch> scratch_;
    std::unique_ptr<SharedVisitedSet> shared_visited_;
    std::vector<SingleNeighbor> merged_candidates_;
    std::vector<SingleNeighbor> merge_buffer_;
};

} // namespace search
} // namespace rnndescent
