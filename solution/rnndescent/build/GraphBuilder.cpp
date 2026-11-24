#include "GraphBuilder.h"

#include "GraphPruning.h"
#include "../common/Logger.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <omp.h>

namespace rnndescent {

GraphBuilder::GraphBuilder(const FloatMatrixView &data_view, bool verbose, const config::BuildConfig &build_config)
    : data_view_(data_view), verbose_(verbose), build_config_(build_config) {}

void detail::sample_unique_ids(std::mt19937 &rng, std::vector<int> &ids, int row_count) {
    const int size = static_cast<int>(ids.size());
    if (size == 0)
        return;
    std::unordered_map<int, int> swapped;
    swapped.reserve(ids.size() * 2);
    for (int i = 0; i < size; ++i) {
        std::uniform_int_distribution<int> distribution(i, row_count - 1);
        const int j = distribution(rng);
        const auto it_i = swapped.find(i);
        const int value_i = it_i == swapped.end() ? i : it_i->second;
        const auto it_j = swapped.find(j);
        const int value_j = it_j == swapped.end() ? j : it_j->second;
        swapped[j] = value_i;
        ids[i] = value_j;
    }
}

void GraphBuilder::init_graph(KNNGraph &graph, MyDistanceComputer &qdis) {
    const int n = data_view_.row_count();
    node_locks_ = std::make_unique<std::mutex[]>(n);
    graph.resize(n);
#pragma omp parallel num_threads(build_config_.num_threads)
    {
        std::mt19937 rng(build_config_.random_seed * 7741 + omp_get_thread_num());
        std::vector<int> sampled_ids(build_config_.S);
#pragma omp for
        for (int i = 0; i < n; ++i) {
            auto &neighbors = graph[i];
            neighbors.reserve(build_config_.S);
            detail::sample_unique_ids(rng, sampled_ids, n);
            for (int id : sampled_ids) {
                if (id == i)
                    continue;
                neighbors.emplace_back(id, qdis.symmetric_dis(i, id), true);
            }
        }
    }
}

void GraphBuilder::update_neighbors(KNNGraph &graph, MyDistanceComputer &qdis) {
    const int n = data_view_.row_count();
    std::vector<std::vector<XNeighbor>> new_pools(build_config_.num_threads);
    std::vector<std::vector<XNeighbor>> old_pools(build_config_.num_threads);
    const size_t reusable_capacity = 2 * static_cast<size_t>(build_config_.K0);
#pragma omp parallel for num_threads(build_config_.num_threads) schedule(dynamic, 16)
    for (int u = 0; u < n; ++u) {
        auto &pool = graph[u];
        auto &new_pool = new_pools[omp_get_thread_num()];
        auto &old_pool = old_pools[omp_get_thread_num()];
        new_pool.clear();
        std::vector<XNeighbor> retired;
        {
            std::lock_guard<std::mutex> guard(node_locks_[u]);
            if (pool.capacity() > reusable_capacity)
                retired.swap(pool);
            else
                old_pool.assign(pool.begin(), pool.end());
            pool.clear();
        }
        auto &candidates = retired.capacity() == 0 ? old_pool : retired;
        detail::sort_and_merge_neighbors(candidates);
        for (auto &nn : candidates) {
            bool keep = true;
            for (auto &other_nn : new_pool) {
                if (!nn.flag() && !other_nn.flag())
                    continue;
                if (nn.id() == other_nn.id()) {
                    keep = false;
                    break;
                }
                const float distance = qdis.symmetric_dis(nn.id(), other_nn.id());
                if (distance < nn.distance) {
                    keep = false;
                    insert_nn(graph, other_nn.id(), nn.id(), distance, true);
                    break;
                }
            }
            if (keep)
                new_pool.emplace_back(nn);
        }
        for (auto &nn : new_pool)
            nn.setflag(false);
        {
            std::lock_guard<std::mutex> guard(node_locks_[u]);
            pool.insert(pool.end(), new_pool.begin(), new_pool.end());
        }
    }
}

void GraphBuilder::add_reverse_edges(KNNGraph &graph) {
    const int n = data_view_.row_count();
    std::vector<size_t> offsets(static_cast<size_t>(n) + 1);
    const auto collect_incoming_edges = [&] {
        std::fill(offsets.begin(), offsets.end(), 0);
        for (auto &neighbors : graph)
            for (auto &neighbor : neighbors)
                ++offsets[neighbor.id() + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        std::vector<XNeighbor> incoming(offsets.back());
        std::vector<size_t> ends(offsets.begin(), offsets.end() - 1);
#pragma omp parallel for num_threads(build_config_.num_threads)
        for (int u = 0; u < n; ++u) {
            for (auto &neighbor : graph[u]) {
                const int v = neighbor.id();
                std::lock_guard<std::mutex> guard(node_locks_[v]);
                incoming[ends[v]++] = XNeighbor(u, neighbor.distance, neighbor.flag());
            }
        }
        return incoming;
    };

    {
        const auto incoming = collect_incoming_edges();
#pragma omp parallel num_threads(build_config_.num_threads)
        {
            std::vector<XNeighbor> merged;
#pragma omp for
            for (int u = 0; u < n; ++u) {
                auto &outgoing = graph[u];
                merged.assign(incoming.begin() + offsets[u], incoming.begin() + offsets[u + 1]);
                for (auto &neighbor : outgoing) {
                    neighbor.setflag(true);
                    merged.push_back(neighbor);
                }
                detail::sort_and_merge_neighbors(merged);
                const auto count = std::min<size_t>(merged.size(), build_config_.R);
                outgoing.assign(merged.begin(), merged.begin() + count);
            }
        }
    }

    auto incoming = collect_incoming_edges();
#pragma omp parallel for num_threads(build_config_.num_threads)
    for (int u = 0; u < n; ++u) {
        auto begin = incoming.begin() + offsets[u];
        auto end = incoming.begin() + offsets[u + 1];
        std::sort(begin, end);
        const auto count = std::min<size_t>(end - begin, build_config_.R);
        graph[u].assign(begin, begin + count);
    }
}

void GraphBuilder::insert_nn(KNNGraph &graph, int node_id, int neighbor_id, float distance, bool flag) {
    std::lock_guard<std::mutex> guard(node_locks_[node_id]);
    graph[node_id].emplace_back(neighbor_id, distance, flag);
}

void GraphBuilder::generate_graph(KNNGraph &graph, MyDistanceComputer &qdis) {
    const int n = data_view_.row_count();
    Logger::info(verbose_, "generte graph ntotal = %d\n", n);
    init_graph(graph, qdis);
    for (int t1 = 0; t1 < build_config_.T1; ++t1) {
        Logger::info(verbose_, "Iter %d : ", t1);
        for (int t2 = 0; t2 < build_config_.T2; ++t2) {
            update_neighbors(graph, qdis);
            Logger::info(verbose_, "#");
        }
        Logger::line(verbose_, "");
        if (t1 != build_config_.T1 - 1)
            add_reverse_edges(graph);
    }
#pragma omp parallel for num_threads(build_config_.num_threads)
    for (int u = 0; u < n; ++u)
        detail::sort_and_merge_neighbors(graph[u]);
    if (build_config_.local_three_hop_pruning) {
        const auto replaced = detail::prune_local_three_hop(graph, build_config_.K0, build_config_.num_threads);
        Logger::info(verbose_, "Local three-hop pruning replaced %zu edges\n", replaced);
    }
    size_t all_edges_size = 0;
    for (int u = 0; u < n; ++u) {
        if (graph[u].size() > static_cast<size_t>(build_config_.K0))
            graph[u].resize(build_config_.K0);
        all_edges_size += graph[u].size();
    }
    Logger::info(verbose_, "graph edges size = %zu\n", all_edges_size);
}

GraphBuildResult GraphBuilder::build() {
    const int n = data_view_.row_count();
    auto qdis = SelectedDistanceComputerFactory::create_build_graph(data_view_.data_ptr(), data_view_.row_count(), data_view_.dimension());
    KNNGraph graph;
    generate_graph(graph, *qdis);
    node_locks_.reset();

    GraphBuildResult result;
    std::vector<size_t> &edge_offsets = result.edge_offsets;
    edge_offsets.resize(static_cast<size_t>(n) + 1);
    size_t padded_edge_count = 0;
    for (const auto &neighbors : graph)
        padded_edge_count += (neighbors.size() + 3) / 4 * 4;
    result.edges.reserve(padded_edge_count);
    std::mt19937 padding_rng(build_config_.random_seed + 1);
    std::uniform_int_distribution<int> padding_dist(0, n - 1);
    for (int i = 0; i < n; ++i) {
        auto &pool = graph[i];
        for (const auto &edge : pool)
            result.edges.push_back(edge.id());
        while ((result.edges.size() - edge_offsets[i]) % 4 != 0) {
            int id = padding_dist(padding_rng);
            const auto begin = result.edges.begin() + edge_offsets[i];
            while (std::find(begin, result.edges.end(), id) != result.edges.end())
                id = padding_dist(padding_rng);
            result.edges.push_back(id);
        }
        edge_offsets[i + 1] = result.edges.size();
    }
    return result;
}

} // namespace rnndescent
