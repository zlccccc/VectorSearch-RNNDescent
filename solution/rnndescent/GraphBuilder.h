#pragma once

#include "GraphConfig.h"
#include "GraphReorder.h"
#include "GraphTypes.h"
#include "Views.h"
#include "discomputer/PlatformSimdDistanceComputer.h"

#include <random>
#include <vector>

namespace rnndescent {

class GraphBuilder {
  public:
    GraphBuilder(const FloatMatrixView &data_view, bool verbose, const config::BuildConfig &build_config,
                 std::vector<MutexWarpper> &node_locks);

    GraphBuildResult build();

  private:
    void generate_graph(KNNGraph &graph, MyDistanceComputer &qdis);
    void init_graph(KNNGraph &graph, MyDistanceComputer &qdis);
    void update_neighbors(KNNGraph &graph, MyDistanceComputer &qdis);
    void add_reverse_edges(KNNGraph &graph);
    void insert_nn(KNNGraph &graph, int node_id, int neighbor_id, float distance, bool flag);
    static void gen_random(std::mt19937 &rng, std::vector<int> &addr, int n);

    const FloatMatrixView &data_view_;
    const bool verbose_;
    const config::BuildConfig build_config_;
    std::vector<MutexWarpper> &node_locks_;
};

} // namespace rnndescent
