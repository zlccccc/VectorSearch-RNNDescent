#include "Checks.h"
#include "solution/rnndescent/discomputer/PlatformSimdDistanceComputer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace tests {
namespace {

using rnndescent::SelectedDistanceComputerFactory;
using rnndescent::SelectedNeighborsContainerType;
using Code = std::remove_pointer_t<decltype(SelectedNeighborsContainerType::pool)>;
using Norm = std::remove_pointer_t<decltype(SelectedNeighborsContainerType::l2norms)>;

constexpr int rows = 8;
constexpr int queries = 2;
constexpr int dot_factor = std::is_unsigned_v<Code> ? 2 : 1;
constexpr int dimensions[] = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 125, 128, 129, 256, 512, 1024};

std::vector<float> make_vectors(int count, int dim, int seed) {
    std::vector<float> vectors(count * dim);
    for (int r = 0; r < count; ++r)
        for (int k = 0; k < dim; ++k)
            vectors[r * dim + k] = float((17 * r + 19 * k + 3 * r * k + seed) % 113 - 56) / 16;
    return vectors;
}

int dot(const Code *a, const Code *b, int dim) {
    int sum = 0;
    for (int k = 0; k < dim; ++k)
        sum += int(a[k]) * int(b[k]);
    return sum;
}

std::array<float, 4> batch(rnndescent::MyDistanceComputer &computer, int q, const int *ids) {
    std::array<float, 4> result{};
    computer.distances_batch_4(q, ids[0], ids[1], ids[2], ids[3], result[0], result[1], result[2], result[3]);
    return result;
}

struct NeighborPool {
    ~NeighborPool() { SelectedNeighborsContainerType::clear_memory(); }
};

} // namespace

void test_simd() {
    for (int d : dimensions) {
        const std::string context = " at dimension " + std::to_string(d);
        const auto base = make_vectors(rows, d, 0);
        const auto query = make_vectors(queries, d, 31);
        auto quantized = SelectedDistanceComputerFactory::create_build_graph(base.data(), rows, d);
        auto exact = SelectedDistanceComputerFactory::create_refine_search(base.data(), rows, d);
        quantized->set_query(query.data(), queries);
        exact->set_query(query.data(), queries);

        std::vector<Code> codes(rows * d);
        std::array<Norm, rows> norms{};
        for (int i = 0; i < rows; ++i)
            quantized->copy_index(i, codes.data() + i * d, norms[i]);

        const int ids[rows] = {7, 1, 3, 4, 2, 5, 0, 6};
        for (int q = 0; q < queries; ++q) {
            const Code *query_code = reinterpret_cast<const Code *>(quantized->get_query_ptr_int8(q));
            for (int i = 0; i < rows; i += 4) {
                const auto quantized_batch = batch(*quantized, q, ids + i);
                const auto exact_batch = batch(*exact, q, ids + i);
                for (int k = 0; k < 4; ++k) {
                    const int id = ids[i + k];
                    const float expected_score = norms[id] - dot_factor * dot(codes.data() + id * d, query_code, d);
                    const double expected_l2 = scalar_l2(base.data() + id * d, query.data() + q * d, d);
                    require_near((*quantized)(q, id), expected_score, 0, "8-bit single query" + context);
                    require_near(quantized_batch[k], expected_score, 0, "8-bit batch of four" + context);
                    require_near((*exact)(q, id), expected_l2, 1e-5 * expected_l2 + 1e-6, "FP32 single query" + context);
                    require_near(exact_batch[k], expected_l2, 1e-5 * expected_l2 + 1e-6, "FP32 batch of four" + context);
                }
            }
        }

        for (int i = 1; i < rows; ++i) {
            const float expected_score = norms[i] + norms[0] - dot_factor * dot(codes.data() + i * d, codes.data(), d);
            const double expected_l2 = scalar_l2(base.data() + i * d, base.data(), d);
            require_near(quantized->symmetric_dis(i, 0), expected_score, 0, "8-bit build distance" + context);
            require_near(exact->symmetric_dis(i, 0), expected_l2, 1e-5 * expected_l2 + 1e-6, "FP32 symmetric distance" + context);
        }
    }
    constexpr int dim = 17;
    const std::vector<float> constant(rows * dim, 3.25f);
    auto quantized = SelectedDistanceComputerFactory::create_build_graph(constant.data(), rows, dim);
    quantized->set_query(constant.data(), rows);
    require_near(quantized->symmetric_dis(0, rows - 1), 0, 0, "constant vectors must have finite zero build distance");
    const float score = (*quantized)(0, 0);
    require(std::isfinite(score) && score == (*quantized)(1, rows - 1), "constant vectors must have equal finite query scores");
}

void test_cache() {
    using Neighbors = SelectedNeighborsContainerType;
    constexpr int lane_width = std::is_unsigned_v<Code> ? 64 : 16;
    static_assert(Neighbors::cache_lane_width == lane_width);
    const std::vector<int> remap{3, 6, 2, 7, 0, 4, 1, 5};
    const int edges[rows] = {5, 2, 7, 0, 6, 3, 1, 4};

    for (int d : dimensions) {
        const std::string context = " at dimension " + std::to_string(d);
        const auto base = make_vectors(rows, d, 5);
        const auto query = make_vectors(queries, d, 17);
        auto computer = SelectedDistanceComputerFactory::create_cached_graph(base.data(), rows, d);
        computer->set_query(query.data(), queries);
        NeighborPool pool;
        Neighbors::init_neighbors_pool(d, {0, rows, 2 * rows, 2 * rows}, 1 << 20);
        Neighbors cached(d, edges, rows, computer.get(), remap, true);
        Neighbors uncached(d, edges, rows, computer.get(), remap, false);
        Neighbors empty(d, nullptr, 0, computer.get(), remap, true);
        require(cached.pool && cached.l2norms && !uncached.pool && !uncached.l2norms, "pool allocation" + context);
        std::vector<uint8_t> scratch(4 * d);
        cached.populate_cache(scratch.data());
        uncached.populate_cache(scratch.data());
        empty.populate_cache(scratch.data());

        std::vector<std::vector<Code>> codes(rows, std::vector<Code>(d));
        for (int i = 0; i < rows; ++i) {
            const int expected_id = remap[edges[i]];
            require(cached.edges[i] == expected_id && uncached.edges[i] == expected_id, "edge ID remapping" + context);
            Norm norm = 0;
            computer->copy_index(expected_id, codes[i].data(), norm);
            require_near(cached.l2norms[i], norm, 0, "cached norm" + context);
        }
        for (int i = 0; i < rows; i += 4) {
            int offset = i * d;
            for (int col = 0; col < d; col += lane_width)
                for (int row = 0; row < 4; ++row)
                    for (int k = col; k < std::min(d, col + lane_width); ++k)
                        require(cached.pool[offset++] == codes[i + row][k], "cached byte layout" + context);
        }

        std::vector<float> from_cache, direct;
        for (int q = 0; q < queries; ++q) {
            cached.compute_distance(q, from_cache);
            uncached.compute_distance(q, direct);
            require(from_cache == direct && direct.size() == rows, "cached and uncached distances" + context);
        }
        empty.compute_distance(0, from_cache);
        require(from_cache.empty(), "empty neighborhood");
    }

    NeighborPool pool;
    Neighbors::init_neighbors_pool(65, {0, rows});
    Neighbors disabled(65, edges, rows, nullptr, remap, true);
    require(disabled.edges && !disabled.pool && !disabled.l2norms, "zero budget must still store edge IDs");

    Neighbors::init_neighbors_pool(65, {0, rows, 2 * rows}, 704);
    // Two 8-neighbor ID slices occupy 128 bytes. One cached slice adds 576 aligned bytes.
    Neighbors first(65, edges, rows, nullptr, remap, true);
    Neighbors second(65, edges, rows, nullptr, remap, true);
    require(first.pool && second.edges && !second.pool, "exhausted budget must fall back to edge IDs");
}

} // namespace tests
