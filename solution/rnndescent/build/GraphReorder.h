#pragma once

#include "../build/GraphConfig.h"
#include "../build/GraphTypes.h"
#include "../Views.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"
#include "../discomputer/utils.h"

#include <memory>
#include <vector>

namespace rnndescent {

struct GraphBuildResult {
    std::vector<int> edges;
    std::vector<size_t> edge_offsets;
};

struct GraphIndexStorage {
    std::vector<int> &search_from_ids;
    std::vector<SelectedNeighborsContainerType> &final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> &graph_distance_computer;
    SelectedNeighborsContainerType::Pool &neighbor_pool;
};

class GraphReorder {
  public:
    static void materialize(const FloatMatrixView &data_view, bool verbose, const config::BuildConfig &build_config,
                            const config::SearchConfig &search_config, GraphBuildResult result, GraphIndexStorage &storage);

  private:
    static std::vector<int> reorder_by_bfs(const std::vector<int> &edges, const std::vector<size_t> &edge_offsets,
                                           int entry_count, const config::BuildConfig &build_config, bool verbose,
                                           std::vector<int> &search_from_ids);
    static size_t padded_degree(int degree) { return (static_cast<size_t>(degree) + 3) / 4 * 4; }
};

} // namespace rnndescent
