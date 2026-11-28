#pragma once

#include "../build/GraphConfig.h"
#include "../build/GraphTypes.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"
#include "CandidateBitmap.h"
#include "NeighborScoreFilter.h"
#include "SearchProfile.h"
#include "QueryWorkers.h"

#include <memory>
#include <vector>

namespace rnndescent {
class CandidateReranker;
struct SearchResultView;

namespace search {

struct Storage {
    const std::vector<SelectedNeighborsContainerType> &final_graph_neighbors;
    const std::unique_ptr<MyDistanceComputer> &graph_distance_computer;
    const std::vector<int> *original_ids = nullptr;
};

class GraphSearcher {
  public:
    GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats);

    void search_batch(int query_count, const int *query_order, const config::SearchConfig &search_config,
                      const CandidateReranker &reranker, const SearchResultView &result);
    void search_single_thread(int threadid, int queryid, const config::SearchConfig &search_config,
                              const CandidateReranker &reranker, const SearchResultView &result);
    void search_parallel(int queryid, const config::SearchConfig &search_config,
                         const CandidateReranker &reranker, const SearchResultView &result);
    void flush_perf_stats() { profile_.flush(); }

  private:
    struct ThreadScratch {
        CandidateBitmap candidate_ids;
        std::vector<SingleNeighbor> retset;
        std::vector<SingleNeighbor> usefulset;
        std::vector<float, AlignedAllocator<float>> scores;
        std::vector<uint32_t, AlignedAllocator<uint32_t>> masks;
        std::vector<detail::NeighborIdSpan, AlignedAllocator<detail::NeighborIdSpan>> spans;

        void reserve(int entries, int beam, size_t score_count) {
            retset.reserve(entries);
            usefulset.reserve(score_count);
            scores.reserve(score_count);
            masks.reserve(detail::score_mask_count(score_count));
            spans.reserve(beam);
        }
    };

    config::SearchConfig checked_search_config(const config::SearchConfig &search_config, const SearchResultView &result) const;
    QueryWorkers *persistent_workers(int active);

    template <bool SharedVisit, bool LimitExpansions = false>
    void search_from_entries(int threadid, int queryid, const config::SearchConfig &config, int entry_begin,
                             const CandidateReranker &reranker, const SearchResultView &result);

    Storage storage_;
    const uint32_t code_stride_;
    const size_t padded_max_degree_;
    std::vector<ThreadScratch> scratch_;
    GraphSearchProfile profile_;
    std::unique_ptr<SharedVisitedSet> shared_visited_;
    std::vector<SingleNeighbor> merged_candidates_;
    std::unique_ptr<QueryWorkers> query_workers_;
};

} // namespace search
} // namespace rnndescent
