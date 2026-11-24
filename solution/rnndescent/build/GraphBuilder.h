#pragma once

#include "GraphConfig.h"
#include "GraphTypes.h"
#include "../common/Views.h"

#include <memory>
#include <mutex>

namespace rnndescent {

struct MyDistanceComputer;

class GraphBuilder {
  public:
    GraphBuilder(const FloatMatrixView &data_view, bool verbose, const config::BuildConfig &build_config);

    GraphBuildResult build();

  private:
    void generate_graph(KNNGraph &graph, MyDistanceComputer &qdis);
    void init_graph(KNNGraph &graph, MyDistanceComputer &qdis);
    void update_neighbors(KNNGraph &graph, MyDistanceComputer &qdis);
    void add_reverse_edges(KNNGraph &graph);
    void insert_nn(KNNGraph &graph, int node_id, int neighbor_id, float distance, bool flag);
    const FloatMatrixView &data_view_;
    const bool verbose_;
    const config::BuildConfig build_config_;
    std::unique_ptr<std::mutex[]> node_locks_;
};

} // namespace rnndescent
