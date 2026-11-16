#include "ReferenceDistance.h"
#include "solution/rnndescent/discomputer/PlatformSimdDistanceComputer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
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

class SimdDistanceTest : public ::testing::TestWithParam<int> {};

class NeighborCacheLayoutTest : public ::testing::TestWithParam<int> {};

TEST_P(SimdDistanceTest, MatchesScalarReference) {
    const int d = GetParam();
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
        SCOPED_TRACE(q);
        for (int i = 0; i < rows; i += 4) {
            const auto quantized_batch = batch(*quantized, q, ids + i);
            const auto exact_batch = batch(*exact, q, ids + i);
            for (int k = 0; k < 4; ++k) {
                const int id = ids[i + k];
                const float expected_score = norms[id] - dot_factor * dot(codes.data() + id * d, query_code, d);
                const double expected_l2 = scalar_l2(base.data() + id * d, query.data() + q * d, d);
                SCOPED_TRACE(id);
                EXPECT_EQ((*quantized)(q, id), expected_score) << "8-bit single query";
                EXPECT_EQ(quantized_batch[k], expected_score) << "8-bit batch of four";
                EXPECT_NEAR((*exact)(q, id), expected_l2, 1e-5 * expected_l2 + 1e-6) << "FP32 single query";
                EXPECT_NEAR(exact_batch[k], expected_l2, 1e-5 * expected_l2 + 1e-6) << "FP32 batch of four";
            }
        }
    }

    for (int i = 1; i < rows; ++i) {
        const float expected_score = norms[i] + norms[0] - dot_factor * dot(codes.data() + i * d, codes.data(), d);
        const double expected_l2 = scalar_l2(base.data() + i * d, base.data(), d);
        SCOPED_TRACE(i);
        EXPECT_EQ(quantized->symmetric_dis(i, 0), expected_score) << "8-bit build distance";
        EXPECT_NEAR(exact->symmetric_dis(i, 0), expected_l2, 1e-5 * expected_l2 + 1e-6) << "FP32 symmetric distance";
    }
}

TEST(SimdQuantizationTest, ConstantVectorsHaveFiniteScores) {
    constexpr int dim = 17;
    const std::vector<float> constant(rows * dim, 3.25f);
    auto quantized = SelectedDistanceComputerFactory::create_build_graph(constant.data(), rows, dim);
    quantized->set_query(constant.data(), rows);
    EXPECT_EQ(quantized->symmetric_dis(0, rows - 1), 0.0f);
    const float score = (*quantized)(0, 0);
    EXPECT_TRUE(std::isfinite(score));
    EXPECT_EQ(score, (*quantized)(1, rows - 1));
}

TEST(SimdQuantizationTest, RowOrderMatchesQuantizingACopy) {
    constexpr int dim = 65;
    const auto base = make_vectors(rows, dim, 5);
    const std::array<int, rows> order{7, 1, 3, 4, 2, 5, 0, 6};
    std::vector<float> reordered(base.size());
    for (int i = 0; i < rows; ++i)
        std::copy_n(base.data() + order[i] * dim, dim, reordered.data() + i * dim);

    auto expected = SelectedDistanceComputerFactory::create_cached_graph(reordered.data(), rows, dim);
    auto actual = SelectedDistanceComputerFactory::create_cached_graph(base.data(), rows, dim, order.data());
    for (int i = 0; i < rows; ++i) {
        std::array<Code, dim> expected_code{}, actual_code{};
        Norm expected_norm = 0, actual_norm = 0;
        expected->copy_index(i, expected_code.data(), expected_norm);
        actual->copy_index(i, actual_code.data(), actual_norm);
        EXPECT_EQ(actual_code, expected_code);
        EXPECT_EQ(actual_norm, expected_norm);
    }
}

using Neighbors = SelectedNeighborsContainerType;
constexpr int lane_width = std::is_unsigned_v<Code> ? 64 : 16;
static_assert(Neighbors::cache_lane_width == lane_width);
const std::vector<int> remap{3, 6, 2, 7, 0, 4, 1, 5};
const int edges[rows] = {5, 2, 7, 0, 6, 3, 1, 4};

TEST_P(NeighborCacheLayoutTest, PreservesLayoutAndDistances) {
    const int d = GetParam();
    const auto base = make_vectors(rows, d, 5);
    const auto query = make_vectors(queries, d, 17);
    auto computer = SelectedDistanceComputerFactory::create_cached_graph(base.data(), rows, d);
    computer->set_query(query.data(), queries);
    Neighbors::Pool storage;
    storage.initialize(d, {0, rows, 2 * rows, 2 * rows}, 1 << 20);
    Neighbors cached(storage, d, edges, rows, computer.get(), remap, true);
    Neighbors uncached(storage, d, edges, rows, computer.get(), remap, false);
    Neighbors empty(storage, d, nullptr, 0, computer.get(), remap, true);
    ASSERT_NE(cached.pool, nullptr);
    ASSERT_NE(cached.l2norms, nullptr);
    EXPECT_EQ(uncached.pool, nullptr);
    EXPECT_EQ(uncached.l2norms, nullptr);
    std::vector<uint8_t> scratch(4 * d);
    cached.populate_cache(scratch.data());
    uncached.populate_cache(scratch.data());
    empty.populate_cache(scratch.data());

    std::vector<std::vector<Code>> codes(rows, std::vector<Code>(d));
    for (int i = 0; i < rows; ++i) {
        const int expected_id = remap[edges[i]];
        EXPECT_EQ(cached.edges[i], expected_id);
        EXPECT_EQ(uncached.edges[i], expected_id);
        Norm norm = 0;
        computer->copy_index(expected_id, codes[i].data(), norm);
        EXPECT_EQ(cached.l2norms[i], norm);
    }
    for (int i = 0; i < rows; i += 4) {
        int offset = i * d;
        for (int col = 0; col < d; col += lane_width)
            for (int row = 0; row < 4; ++row)
                for (int k = col; k < std::min(d, col + lane_width); ++k)
                    EXPECT_EQ(cached.pool[offset++], codes[i + row][k]);
    }

    std::vector<float> from_cache, direct;
    for (int q = 0; q < queries; ++q) {
        cached.compute_distance(q, from_cache);
        uncached.compute_distance(q, direct);
        EXPECT_EQ(direct.size(), rows);
        EXPECT_EQ(from_cache, direct) << "query " << q;
    }
    empty.compute_distance(0, from_cache);
    EXPECT_TRUE(from_cache.empty());
}

TEST(NeighborCacheTest, FallsBackWhenBudgetIsExhausted) {
    Neighbors::Pool disabled_storage;
    disabled_storage.initialize(65, {0, rows});
    Neighbors disabled(disabled_storage, 65, edges, rows, nullptr, remap, true);
    EXPECT_NE(disabled.edges, nullptr);
    EXPECT_EQ(disabled.pool, nullptr);
    EXPECT_EQ(disabled.l2norms, nullptr);

    Neighbors::Pool storage;
    storage.initialize(65, {0, rows, 2 * rows}, 704);
    // Two 8-neighbor ID slices occupy 128 bytes. One cached slice adds 576 aligned bytes.
    Neighbors first(storage, 65, edges, rows, nullptr, remap, true);
    Neighbors second(storage, 65, edges, rows, nullptr, remap, true);
    EXPECT_NE(first.pool, nullptr);
    EXPECT_NE(second.edges, nullptr);
    EXPECT_EQ(second.pool, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, SimdDistanceTest, ::testing::ValuesIn(dimensions), ::testing::PrintToStringParamName{});
INSTANTIATE_TEST_SUITE_P(Dimensions, NeighborCacheLayoutTest, ::testing::ValuesIn(dimensions), ::testing::PrintToStringParamName{});

} // namespace
} // namespace tests
