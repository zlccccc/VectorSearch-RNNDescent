#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace rnndescent::detail {

// Reorder sorted, unique candidate rows before truncating them to max_degree.
// Every removed edge has a three-hop route through edges in the fixed prefixes.
template <typename Graph>
std::size_t prune_local_three_hop(Graph &graph, int max_degree, int threads) {
    if (max_degree < 2 || static_cast<std::size_t>(max_degree) >= graph.size())
        return 0;

    constexpr int max_swaps = 4;
    const int protected_count = std::min(32, max_degree / 2);
    const int shortlist = static_cast<int>(std::min(2 * static_cast<std::size_t>(max_degree), graph.size()));
    struct FirstHops {
        uint32_t direct = 0;
        uint32_t two_hop = 0;
        uint32_t three_hop = 0;
    };

    std::size_t replaced = 0;
#pragma omp parallel num_threads(threads) reduction(+:replaced)
    {
        std::vector<int> rank_by_id(graph.size(), -1);
        std::vector<FirstHops> routes(shortlist);
        std::vector<uint8_t> scores(shortlist), keep(shortlist);
        std::vector<int> removable(max_degree - protected_count), alternatives(shortlist - max_degree);
#pragma omp for schedule(dynamic,64)
        for (int u = 0; u < static_cast<int>(graph.size()); ++u) {
            auto &row = graph[u];
            const int width = static_cast<int>(std::min(row.size(), static_cast<std::size_t>(shortlist)));
            if (width <= max_degree)
                continue;

            std::fill_n(routes.begin(), width, FirstHops{});
            for (int i = 0; i < width; ++i)
                rank_by_id[row[i].id()] = i;
            for (int i = 0; i < protected_count; ++i)
                routes[i].direct = uint32_t{1} << i;

            // Visit candidates in increasing rank, so every path uses distinct
            // vertices. A bit represents a first hop regardless of path count.
            for (int a = 0; a < width; ++a) {
                const auto &from = routes[a];
                if ((from.direct | from.two_hop) == 0)
                    continue;
                const auto &neighbors = graph[row[a].id()];
                const int count = static_cast<int>(std::min(neighbors.size(), static_cast<std::size_t>(protected_count)));
                for (int i = 0; i < count; ++i) {
                    const int b = rank_by_id[neighbors[i].id()];
                    if (b > a) {
                        routes[b].two_hop |= from.direct;
                        routes[b].three_hop |= from.two_hop;
                    }
                }
            }
            for (int i = 0; i < width; ++i) {
                scores[i] = __builtin_popcount(routes[i].three_hop);
                rank_by_id[row[i].id()] = -1;
            }

            const auto less = [&](int a, int b) { return std::pair(scores[a], a) < std::pair(scores[b], b); };
            std::iota(removable.begin(), removable.end(), protected_count);
            const auto alternative_end = alternatives.begin() + (width - max_degree);
            std::iota(alternatives.begin(), alternative_end, max_degree);
            if (scores[*std::max_element(removable.begin(), removable.end(), less)] <=
                scores[*std::min_element(alternatives.begin(), alternative_end, less)])
                continue;

            const int swaps = std::min({max_swaps, max_degree - protected_count, width - max_degree});
            std::partial_sort(removable.begin(), removable.begin() + swaps, removable.end(),
                              [&](int a, int b) { return less(b, a); });
            std::partial_sort(alternatives.begin(), alternatives.begin() + swaps, alternative_end, less);
            std::fill_n(keep.begin(), max_degree, 1);
            std::fill(keep.begin() + max_degree, keep.begin() + width, 0);
            for (int i = 0; i < swaps && scores[removable[i]] > scores[alternatives[i]]; ++i) {
                keep[removable[i]] = 0;
                keep[alternatives[i]] = 1;
                ++replaced;
            }

            // Workers read other rows' protected prefixes and sizes. Write
            // only this row's tail and defer all resizing to the caller.
            int out = protected_count;
            for (int i = protected_count; i < width; ++i)
                if (keep[i])
                    row[out++] = row[i];
        }
    }
    return replaced;
}

} // namespace rnndescent::detail
