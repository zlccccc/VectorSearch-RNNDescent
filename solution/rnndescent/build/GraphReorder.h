#pragma once

#include "GraphConfig.h"
#include "GraphTypes.h"
#include "../common/Views.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

#include <memory>
#include <vector>

namespace rnndescent {

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
};

} // namespace rnndescent
