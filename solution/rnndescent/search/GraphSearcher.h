#pragma once

#include "../build/GraphConfig.h"
#include "../build/GraphTypes.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"
#include "CandidateBitmap.h"
#include "NeighborScoreFilter.h"

#include <memory>
#include <mutex>
#include <vector>

namespace rnndescent {
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

    void reset() { *this = {}; }
};
#else
struct PerfStats {
    void reset() {}
};
#endif

struct Storage {
    std::vector<SelectedNeighborsContainerType> &final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> &graph_distance_computer;
    uint32_t code_stride;
};

class GraphSearcher {
  public:
    GraphSearcher(Storage storage, const config::SearchConfig &search_config, int max_degree, PerfStats &perf_stats);

    // The returned buffer belongs to this worker and contains BFS IDs.
    std::vector<SingleNeighbor> &search_candidates(int threadid, int queryid, const config::SearchConfig &search_config,
                                                  int max_degree, int topk);
#ifdef INTERNAL_CLOCK_TEST
    void record_refinement_time(float elapsed_ms);
#endif
    void flush_perf_stats();

  private:
    static size_t padded_degree(int degree) { return (static_cast<size_t>(degree) + 3) / 4 * 4; }

    Storage storage_;
    const int num_threads_;
    PerfStats &perf_stats_;
    std::mutex mutex_;

    std::vector<std::vector<SingleNeighbor>> retset_;
    std::vector<std::vector<SingleNeighbor>> usefulset_;
    std::vector<std::vector<float, AlignedAllocator<float>>> neighbor_distance_;
    std::vector<std::vector<uint32_t, AlignedAllocator<uint32_t>>> neighbor_masks_;
    std::vector<CandidateBitmap> candidate_ids_;
    std::vector<std::vector<detail::NeighborIdSpan, AlignedAllocator<detail::NeighborIdSpan>>> neighbor_spans_;
};

} // namespace search
} // namespace rnndescent
