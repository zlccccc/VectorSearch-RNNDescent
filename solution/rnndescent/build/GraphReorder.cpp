#include "GraphReorder.h"

#include "../common/HugePages.h"
#include "../common/Logger.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>

namespace rnndescent {

static std::vector<int> reorder_by_bfs(const std::vector<int> &edges, const std::vector<size_t> &edge_offsets,
                                       int entry_count, const config::BuildConfig &build_config, bool verbose,
                                       std::vector<int> &search_from_ids) {
    const int n = static_cast<int>(edge_offsets.size() - 1);
    search_from_ids.reserve(n);
    search_from_ids.resize(entry_count);
    if (build_config.random_init) {
        std::mt19937 rng(build_config.random_seed);
        detail::sample_unique_ids(rng, search_from_ids, n);
    } else {
        std::iota(search_from_ids.begin(), search_from_ids.end(), 0);
    }

    std::vector<uint8_t> visited(n);
    for (int root : search_from_ids)
        visited[root] = 1;

    for (size_t head = 0, depth = 0; head < search_from_ids.size(); ++depth) {
        const size_t layer_end = search_from_ids.size();
        Logger::info(verbose, "BFS depth %zu: %zu nodes\n", depth, layer_end - head);
        for (; head < layer_end; ++head) {
            const int u = search_from_ids[head];
            for (size_t i = edge_offsets[u]; i < edge_offsets[u + 1]; ++i) {
                const int v = edges[i];
                if (!visited[v]) {
                    visited[v] = 1;
                    search_from_ids.push_back(v);
                }
            }
        }
    }

    const size_t reachable = search_from_ids.size();
    for (int original_id = 0; original_id < n; ++original_id) {
        if (!visited[original_id])
            search_from_ids.push_back(original_id);
    }
    Logger::info(verbose, "BFS reached %zu of %d nodes; appended %zu unreachable nodes\n", reachable, n,
                 search_from_ids.size() - reachable);

    std::vector<int> original_to_bfs(n);
    for (int bfs_id = 0; bfs_id < n; ++bfs_id)
        original_to_bfs[search_from_ids[bfs_id]] = bfs_id;
    return original_to_bfs;
}

void GraphReorder::materialize(const FloatMatrixView &data_view, bool verbose, const config::BuildConfig &build_config,
                               const config::SearchConfig &search_config, GraphBuildResult result, GraphIndexStorage &storage) {
    const int n = data_view.row_count();
    const int dim = data_view.dimension();
    const auto original_to_bfs = reorder_by_bfs(result.edges, result.edge_offsets, search_config.num_initialize, build_config,
                                                verbose, storage.search_from_ids);

    storage.graph_distance_computer =
        SelectedDistanceComputerFactory::create_cached_graph(data_view.data_ptr(), n, dim, storage.search_from_ids.data(),
                                                              search_config.graph_quantization);
    auto &distance = *storage.graph_distance_computer;
    const auto code_stride = distance.code_stride();
    const auto cache_limit = build_config.save_neighbor ? build_config.neighbor_pool_size_limit_bytes : 0;
    storage.neighbor_pool.initialize(code_stride, result.edge_offsets, cache_limit, storage.search_from_ids.data());
    storage.final_graph_neighbors.reserve(n);
    for (int bfs_id = 0; bfs_id < n; ++bfs_id) {
        const int original_id = storage.search_from_ids[bfs_id];
        const size_t begin = result.edge_offsets[original_id];
        const int count = static_cast<int>(result.edge_offsets[original_id + 1] - begin);
        const int *neighbors = count > 0 ? result.edges.data() + begin : nullptr;
        storage.final_graph_neighbors.emplace_back(storage.neighbor_pool, neighbors, count, original_to_bfs);
    }
    // Traversal reads the node descriptors in graph expansion order.
    detail::try_huge_pages(storage.final_graph_neighbors.data(),
                          storage.final_graph_neighbors.size() * sizeof(SelectedNeighborsContainerType));

    if (cache_limit > 0) {
#pragma omp parallel num_threads(build_config.num_threads)
        {
            std::vector<uint8_t> reorder_scratch(4 * static_cast<size_t>(code_stride));
#pragma omp for
            for (int i = 0; i < n; ++i)
                storage.final_graph_neighbors[i].populate_cache(distance, code_stride, reorder_scratch.data());
        }
    }
}

} // namespace rnndescent
