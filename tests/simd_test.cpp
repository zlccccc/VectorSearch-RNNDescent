#include "ReferenceDistance.h"
#include "solution/rnndescent/discomputer/PlatformSimdDistanceComputer.h"
#include "solution/rnndescent/discomputer/flat/CblasDistComputerFP32.h"

#ifdef RNNDESCENT_FORCE_NEON
#include "solution/rnndescent/discomputer/flat/NeonSimdDistanceComputerFP16.h"
#include "solution/rnndescent/discomputer/flat/NeonSimdDistanceComputerInt16.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tests {
namespace {

using rnndescent::SelectedDistanceComputerFactory;
using rnndescent::SelectedNeighborsContainerType;
using Neighbors = SelectedNeighborsContainerType;
using Code = Neighbors::Code;
using Norm = int;

constexpr int rows = 8;
constexpr int queries = 2;
#if defined(RNNDESCENT_FORCE_AVX2_UINT8) || defined(RNNDESCENT_FORCE_AVX_VNNI)
constexpr int lane_width = 32;
#else
constexpr int lane_width = std::is_unsigned_v<Code> ? 64 : 16;
#endif
static_assert(Neighbors::cache_lane_width == lane_width);
static_assert(sizeof(Neighbors) == sizeof(void *));
constexpr int dimensions[] = {
    1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 125, 128, 129, 256, 384, 385, 512, 960, 961, 1024, 1537,
};

std::vector<float> make_vectors(int count, int dim, int seed) {
    std::vector<float> vectors(count * dim);
    for (int r = 0; r < count; ++r)
        for (int k = 0; k < dim; ++k)
            vectors[r * dim + k] = float((17 * r + 19 * k + 3 * r * k + seed) % 113 - 56) / 16;
    return vectors;
}

float quantized_l2(const Code *a, const Code *b, int dim) {
    int64_t sum = 0;
    for (int k = 0; k < dim; ++k) {
        const int difference = int(a[k]) - int(b[k]);
        sum += difference * difference;
    }
    return static_cast<float>(sum);
}

std::array<float, 4> batch(rnndescent::MyDistanceComputer &computer, int q, const int *ids) {
    std::vector<rnndescent::SingleNeighbor> candidates{{ids[0], 0}, {ids[1], 0}, {ids[2], 0}, {ids[3], 0}};
    computer.score_candidates(q, candidates.data(), candidates.size());
    return {candidates[0].distance, candidates[1].distance, candidates[2].distance, candidates[3].distance};
}

class SimdDistanceTest : public ::testing::TestWithParam<int> {};

class NeighborCacheLayoutTest : public ::testing::TestWithParam<int> {};

TEST_P(SimdDistanceTest, MatchesScalarReference) {
    const int d = GetParam();
    const auto base = make_vectors(rows, d, 0);
    const auto query = make_vectors(queries, d, 31);
    auto quantized = SelectedDistanceComputerFactory::create_build_graph(base.data(), rows, d);
    auto exact = SelectedDistanceComputerFactory::create_refine_search(base.data(), rows, d);
    const int stride = (d + lane_width - 1) / lane_width * lane_width;
    EXPECT_EQ(quantized->dimension(), d);
    ASSERT_EQ(quantized->code_stride(), stride);
    quantized->set_query(query.data(), queries);
    exact->set_query(query.data(), queries);

    std::vector<Code> codes(rows * d);
    std::array<Norm, rows> norms{};
    for (int i = 0; i < rows; ++i)
        quantized->copy_index(i, codes.data() + i * d, norms[i]);

    const int ids[rows] = {7, 1, 3, 4, 2, 5, 0, 6};
    for (int q = 0; q < queries; ++q) {
        const auto *query_code = static_cast<const Code *>(quantized->get_quantized_query(q).code);
        SCOPED_TRACE(q);
        EXPECT_TRUE(std::all_of(query_code + d, query_code + stride, [](Code value) { return value == 0; }));
        for (int i = 0; i < rows; i += 4) {
            const auto quantized_batch = batch(*quantized, q, ids + i);
            const auto exact_batch = batch(*exact, q, ids + i);
            for (int k = 0; k < 4; ++k) {
                const int id = ids[i + k];
                const float expected_score = quantized_l2(codes.data() + id * d, query_code, d);
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
        const float expected_score = quantized_l2(codes.data() + i * d, codes.data(), d);
        const double expected_l2 = scalar_l2(base.data() + i * d, base.data(), d);
        SCOPED_TRACE(i);
        EXPECT_EQ(quantized->symmetric_dis(i, 0), expected_score) << "8-bit build distance";
        EXPECT_NEAR(exact->symmetric_dis(i, 0), expected_l2, 1e-5 * expected_l2 + 1e-6) << "FP32 symmetric distance";
    }
}

TEST(CblasDistanceTest, UsesRequestedQueryForSingleAndBatchDistances) {
    constexpr int dim = 65;
    constexpr int ids[] = {0, 3, 7, 1};
    const auto base = make_vectors(rows, dim, 0);
    const auto query = make_vectors(queries, dim, 31);
    rnndescent::CblasDistanceComputerFP32L2 computer(base.data(), rows, dim);
    computer.set_query(query.data(), queries);

    const auto distances = batch(computer, 1, ids);
    for (int k = 0; k < 4; ++k) {
        const double expected = scalar_l2(query.data() + dim, base.data() + ids[k] * dim, dim);
        EXPECT_NEAR(computer(1, ids[k]), expected, 1e-5 * expected + 1e-6);
        EXPECT_FLOAT_EQ(distances[k], computer(1, ids[k]));
    }
}

TEST(SimdQuantizationTest, IdenticalVectorsHaveZeroDistance) {
    constexpr int dim = 17;
    const std::vector<float> constant(rows * dim, 3.25f);
    auto quantized = SelectedDistanceComputerFactory::create_build_graph(constant.data(), rows, dim);
    quantized->set_query(constant.data(), rows);
    EXPECT_EQ(quantized->symmetric_dis(0, rows - 1), 0.0f);
    EXPECT_EQ((*quantized)(0, 0), 0.0f);
    EXPECT_EQ((*quantized)(1, rows - 1), 0.0f);
}

TEST(SimdQuantizationTest, ExtremeCodesPreserveSquaredDistances) {
    constexpr int difference = std::is_unsigned_v<Code> ? 126 : 254;
    constexpr float scale = 2.0f / difference;
    for (const int dim : {65, 1024, 33286}) {
        SCOPED_TRACE(dim);
        std::vector<float> base(2 * dim, -1.0f);
        std::fill(base.begin() + dim, base.end(), 1.0f);
        auto quantized = SelectedDistanceComputerFactory::create_build_graph(base.data(), 2, dim);
        quantized->set_query(base.data(), 2);

        const float distance = static_cast<float>(dim * difference * difference);
        EXPECT_EQ(quantized->symmetric_dis(0, 1), distance);
        EXPECT_EQ(quantized->symmetric_dis(1, 1), 0.0f);
        EXPECT_FLOAT_EQ((*quantized)(1, 0) * (scale * scale), 4.0f * dim);
        EXPECT_EQ((*quantized)(1, 1), 0.0f);
        const int ids[] = {0, 1, 1, 0};
        const std::array<float, 4> expected{distance, 0.0f, 0.0f, distance};
        EXPECT_EQ(batch(*quantized, 1, ids), expected);

        Neighbors::Pool storage;
        const auto stride = quantized->code_stride();
        storage.initialize(stride, {0, 4}, 1 << 20);
        Neighbors cached(storage, ids, 4, {0, 1});
        ASSERT_TRUE(cached.has_cache());
        std::vector<uint8_t> scratch(stride);
        cached.populate_cache(*quantized, stride, scratch.data());
        std::vector<float> distances(4);
        EXPECT_EQ(cached.compute_distance(1, *quantized, stride, distances.data()).second, 4);
        EXPECT_EQ(distances, (std::vector<float>{distance, 0.0f, 0.0f, distance}));
    }
}

TEST(SimdQuantizationTest, RejectsDimensionsBeyondSquaredDistanceLimit) {
    constexpr int dim = 33287;
    const std::vector<float> data(dim, 0.0f);
    EXPECT_THROW(SelectedDistanceComputerFactory::create_build_graph(data.data(), 1, dim), std::invalid_argument);
    EXPECT_THROW(SelectedDistanceComputerFactory::create_cached_graph(data.data(), 1, dim), std::invalid_argument);
}

#ifdef RNNDESCENT_FORCE_NEON
TEST(NeonDistanceTest, FP16AndInt16MatchScalarOnOddDimensions) {
    constexpr int ids[] = {0, 3, 7, 1};
    for (int dim : {1, 3, 4, 7, 8, 9, 17, 125, 128, 960, 961}) {
        SCOPED_TRACE(dim);
        auto base = make_vectors(rows, dim, 9);
        auto query = make_vectors(queries, dim, 21);
        for (auto &value : base)
            value *= 0.83f;
        for (auto &value : query)
            value *= 0.83f;
        std::copy_n(base.data(), dim, query.data());

        rnndescent::SimdDistanceComputerFP16L2 fp16(base.data(), rows, dim);
        rnndescent::SimdDistanceComputerInt16L2 int16(base.data(), rows, dim);
        const auto check = [&](rnndescent::MyDistanceComputer &computer, double tolerance) {
            EXPECT_EQ(computer.row_count(), rows);
            EXPECT_EQ(computer.dimension(), dim);
            computer.set_query(query.data(), queries);
            EXPECT_FLOAT_EQ(computer(0, 0), 0.0f);
            EXPECT_FLOAT_EQ(computer.symmetric_dis(0, 0), 0.0f);
            for (int q = 0; q < queries; ++q) {
                const auto distances = batch(computer, q, ids);
                for (int k = 0; k < 4; ++k) {
                    const double expected = scalar_l2(query.data() + q * dim, base.data() + ids[k] * dim, dim);
                    EXPECT_NEAR(computer(q, ids[k]), expected, tolerance * std::max(1.0, expected));
                    EXPECT_FLOAT_EQ(distances[k], computer(q, ids[k]));
                }
            }
            const double expected = scalar_l2(base.data(), base.data() + 7 * dim, dim);
            EXPECT_NEAR(computer.symmetric_dis(0, 7), expected, tolerance * std::max(1.0, expected));
        };
        check(fp16, 2e-3);
        check(int16, 2e-4);
    }
}
#endif

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

const std::vector<int> remap{3, 6, 2, 7, 0, 4, 1, 5};
const int edges[rows] = {5, 2, 7, 0, 6, 3, 1, 4};

TEST_P(NeighborCacheLayoutTest, PreservesLayoutAndDistances) {
    // 28 neighbors exercise the four-row layout and, on GEMV backends, a 16-row tile with a 12-row remainder.
    constexpr int count = 28;
    const int d = GetParam();
    const auto base = make_vectors(count, d, 5);
    const auto query = make_vectors(queries, d, 17);
    auto computer = SelectedDistanceComputerFactory::create_cached_graph(base.data(), count, d);
    computer->set_query(query.data(), queries);
    const auto stride = computer->code_stride();
    std::array<int, count> edge_ids{};
    std::vector<int> id_map(count);
    for (int i = 0; i < count; ++i) {
        edge_ids[i] = (3 * i + 1) % count;
        id_map[i] = (5 * i + 9) % count;
    }
    Neighbors::Pool storage;
    Neighbors::Pool id_storage;
    storage.initialize(stride, {0, count, count}, 1 << 20);
    id_storage.initialize(stride, {0, count});
    Neighbors uncached(id_storage, edge_ids.data(), count, id_map);
    Neighbors cached(storage, edge_ids.data(), count, id_map);
    Neighbors empty(storage, nullptr, 0, id_map);
    ASSERT_TRUE(cached.has_cache());
    ASSERT_FALSE(uncached.has_cache());
    const auto *cached_ids = cached.neighbor_ids(stride);
    const auto *plain_ids = uncached.neighbor_ids(stride);
    // The packed codes and norms immediately precede the IDs.
    const auto *code_bytes = reinterpret_cast<const uint8_t *>(cached_ids) - count * stride * sizeof(Code);
    const auto *cached_codes = reinterpret_cast<const Code *>(code_bytes);
    const auto *cached_norms = reinterpret_cast<const Norm *>(code_bytes - count * sizeof(Norm));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(cached_codes) % lane_width, 0u);
    std::vector<uint8_t> scratch(stride);
    cached.populate_cache(*computer, stride, scratch.data());
    uncached.populate_cache(*computer, stride, scratch.data());
    empty.populate_cache(*computer, stride, scratch.data());

    std::vector<std::vector<Code>> codes(count, std::vector<Code>(stride));
    for (int i = 0; i < count; ++i) {
        const int expected_id = id_map[edge_ids[i]];
        EXPECT_EQ(cached_ids[i], expected_id);
        EXPECT_EQ(plain_ids[i], expected_id);
        Norm norm = 0;
        computer->copy_index(expected_id, codes[i].data(), norm);
        EXPECT_EQ(cached_norms[i], norm);
    }
#if defined(RNNDESCENT_FORCE_AVX2_UINT8) || defined(RNNDESCENT_FORCE_AVX_VNNI)
    const bool uses_gemv = true;
#else
    const bool uses_gemv = lane_width == 64 && stride == 64;
#endif
    for (int i = 0; i < count;) {
        const int tile = uses_gemv && count - i >= 16 ? 16 : 4;
        const size_t block_bytes = tile == 16 ? 4 : lane_width;
        size_t offset = i * stride;
        for (size_t col = 0; col < stride; col += block_bytes)
            for (int row = 0; row < tile; ++row)
                for (size_t k = col; k < col + block_bytes; ++k)
                    EXPECT_EQ(cached_codes[offset++], codes[i + row][k]);
        i += tile;
    }

    std::vector<float, rnndescent::AlignedAllocator<float>> from_cache(count);
    std::vector<float> direct(count);
    for (int q = 0; q < queries; ++q) {
        EXPECT_EQ(cached.compute_distance(q, *computer, stride, from_cache.data()), std::make_pair(cached_ids, count));
        EXPECT_EQ(uncached.compute_distance(q, *computer, stride, direct.data()), std::make_pair(plain_ids, count));
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(from_cache.data()) % rnndescent::alignSize, 0u);
        EXPECT_TRUE(std::equal(from_cache.begin(), from_cache.end(), direct.begin())) << "query " << q;
    }
    float untouched = -1.0f;
    EXPECT_EQ(empty.compute_distance(0, *computer, stride, &untouched).second, 0);
    EXPECT_EQ(untouched, -1.0f);
}

#if defined(RNNDESCENT_HAS_UINT4_KERNEL)
TEST(PackedUInt4CacheTest, MatchesScalarAndCacheFallback) {
    using Kernel = rnndescent::SelectedUInt4Kernel;
    using PackedGraph = rnndescent::QuantizedDistanceComputer<Kernel, Kernel::Encoding>;
    constexpr size_t block = Kernel::block_bytes;
    constexpr int count = 28;
    std::array<int, count> edges;
    std::vector<int> row_order(count);
    std::iota(edges.begin(), edges.end(), 0);
    std::iota(row_order.rbegin(), row_order.rend(), 0);
    for (int dimension : {31, 64, 65, 193}) {
        SCOPED_TRACE(dimension);
        const auto base = make_vectors(count, dimension, 5);
        const auto query = make_vectors(queries, dimension, 17);
        auto computer = std::make_unique<PackedGraph>(base.data(), count, dimension, row_order.data());
        const auto selected = SelectedDistanceComputerFactory::create_cached_graph(
            base.data(), count, dimension, row_order.data(), rnndescent::QuantizedFormat::UInt4);
        computer->set_query(query.data(), queries);
        const auto stride = computer->code_stride();
        ASSERT_EQ(stride, (dimension + 2 * block - 1) / (2 * block) * block);
        EXPECT_EQ(selected->code_stride(), stride);

        Neighbors::Pool pool;
        Neighbors::Pool id_pool;
        pool.initialize(stride, {0, count}, 1 << 20);
        id_pool.initialize(stride, {0, count});
        Neighbors cached(pool, edges.data(), count, row_order);
        Neighbors uncached(id_pool, edges.data(), count, row_order);
        ASSERT_TRUE(cached.has_cache());
        ASSERT_FALSE(uncached.has_cache());
        std::vector<uint8_t> row(stride);
        cached.populate_cache(*computer, stride, row.data());

        for (int q = 0; q < queries; ++q) {
            const auto *qcode = static_cast<const int8_t *>(computer->get_quantized_query(q).code);
            std::array<float, count> cached_distances, uncached_distances;
            const auto [ids, cached_count] = cached.compute_distance(q, *computer, stride, cached_distances.data());
            ASSERT_EQ(cached_count, count);
            EXPECT_EQ(uncached.compute_distance(q, *computer, stride, uncached_distances.data()).second, count);
            for (int i = 0; i < count; ++i) {
                const int id = ids[i];
                int norm;
                computer->copy_index(id, row.data(), norm, true);
                int sum = 0;
                for (size_t k = 0; k < 2 * stride; ++k) {
                    const uint8_t byte = row[k / (2 * block) * block + k % block];
                    const int code = k % (2 * block) < block ? byte & 15 : byte >> 4;
                    const int delta = 8 * code - qcode[k];
                    sum += delta * delta;
                    if (k >= static_cast<size_t>(dimension))
                        EXPECT_EQ(delta, 0);
                }
                EXPECT_EQ((*computer)(q, id), sum);
                EXPECT_EQ(cached_distances[i], sum);
                EXPECT_EQ(uncached_distances[i], sum);
            }
        }
    }
}
#endif

TEST(QuantizedKernelBaseTest, PacksAndScoresEightRowTiles) {
    using Format = std::conditional_t<std::is_signed_v<Code>, rnndescent::SignedInt8Quantization, rnndescent::UInt8Quantization>;
    constexpr size_t chunk = 8;
    using Tile = rnndescent::QuantizedKernelBase<Format, lane_width, 8, chunk>;
    constexpr int count = 2 * Tile::rows;
    constexpr int dim = lane_width + 3;
    const auto base = make_vectors(count, dim, 5);
    const auto query = make_vectors(1, dim, 17);
    auto computer = SelectedDistanceComputerFactory::create_cached_graph(base.data(), count, dim);
    computer->set_query(query.data(), 1);
    const auto stride = computer->code_stride();
    std::array<int, count> ids{}, norms{};
    for (int i = 0; i < count; ++i)
        ids[i] = count - i - 1;
    std::vector<Code> packed(count * stride);
    std::vector<uint8_t> scratch(stride);
    Tile::pack_tiles(*computer, ids.data(), count, stride, packed.data(), norms.data(), scratch.data());

    const auto query_data = computer->get_quantized_query(0);
    const auto *q = static_cast<const Code *>(query_data.code);
    std::array<float, count> scores{};
    Tile::compute_tiles(count, stride, packed.data(), norms.data(), scores.data(),
                         [&](const Code *block, const int *tile_norms, float *out) {
        std::array<int32_t, Tile::rows> dots{};
        for (size_t feature = 0; feature < stride; feature += chunk)
            for (int row = 0; row < Tile::rows; ++row)
                for (size_t k = 0; k < chunk; ++k)
                    dots[row] += q[feature + k] * block[feature * Tile::rows + row * chunk + k];
        Tile::store_scores(query_data.squared_norm, tile_norms, dots, out);
    });
    for (int i = 0; i < count; ++i)
        EXPECT_EQ(scores[i], (*computer)(0, ids[i]));
}

TEST(NeighborCacheTest, FallsBackWhenBudgetIsExhausted) {
    constexpr size_t stride = (65 + lane_width - 1) / lane_width * lane_width;
    Neighbors::Pool disabled_storage;
    disabled_storage.initialize(stride, {0, rows});
    Neighbors disabled(disabled_storage, edges, rows, remap);
    EXPECT_NE(disabled.neighbor_ids(stride), nullptr);
    EXPECT_FALSE(disabled.has_cache());

    Neighbors::Pool storage;
    constexpr size_t norm_end = sizeof(int) + rows * sizeof(Norm);
    constexpr size_t code_start = (norm_end + lane_width - 1) / lane_width * lane_width;
    constexpr size_t cached_bytes = code_start + rows * (stride + sizeof(int));
    constexpr size_t budget = cached_bytes + sizeof(int) + rows * sizeof(int);
    storage.initialize(stride, {0, rows, 2 * rows}, budget);
    Neighbors first(storage, edges, rows, remap);
    Neighbors second(storage, edges, rows, remap);
    EXPECT_TRUE(first.has_cache());
    EXPECT_FALSE(second.has_cache());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.neighbor_ids(stride)),
              reinterpret_cast<std::uintptr_t>(first.neighbor_ids(stride)) + (rows + 1) * sizeof(int));
}

INSTANTIATE_TEST_SUITE_P(Dimensions, SimdDistanceTest, ::testing::ValuesIn(dimensions), ::testing::PrintToStringParamName{});
INSTANTIATE_TEST_SUITE_P(Dimensions, NeighborCacheLayoutTest, ::testing::ValuesIn(dimensions), ::testing::PrintToStringParamName{});

} // namespace
} // namespace tests
