#pragma once

#include "../MyDistanceComputer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rnndescent {

// The quantized row stride and the cache interleave width are independent.
template <class Format, size_t BlockBytes, int Rows, size_t InterleaveBytes = BlockBytes>
struct QuantizedKernelBase {
    static_assert(Rows > 0 && BlockBytes > 0 && InterleaveBytes > 0 && BlockBytes % InterleaveBytes == 0);
    using Quantization = Format;
    using Code = typename Format::Code;
    static constexpr size_t block_bytes = BlockBytes;
    static constexpr int rows = Rows;

    static void pack_tile(MyDistanceComputer &distance, const int *ids, size_t stride, Code *codes, int *norms,
                          uint8_t *scratch) {
        for (int lane = 0; lane < rows; ++lane) {
            distance.copy_index(ids[lane], scratch, norms[lane], true);
            for (size_t feature = 0; feature < stride; feature += InterleaveBytes)
                std::memcpy(codes + feature * rows + lane * InterleaveBytes, scratch + feature, InterleaveBytes);
        }
    }

    // The caller supplies full tiles of this shape.
    static void pack_tiles(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                           Code *codes, int *norms, uint8_t *scratch) {
        for (int i = 0; i < count; i += rows)
            pack_tile(distance, ids + i, stride, codes + static_cast<size_t>(i) * stride, norms + i, scratch);
    }

    // Mixed layouts place every large tile before the smaller tail tiles.
    template <class TailTile>
    static void pack_tiles_with_tail(MyDistanceComputer &distance, const int *ids, int count, uint32_t stride,
                                    Code *codes, int *norms, uint8_t *scratch, bool use_large_tiles = true) {
        static_assert(rows % TailTile::rows == 0);
        const int split = use_large_tiles ? count - count % rows : 0;
        pack_tiles(distance, ids, split, stride, codes, norms, scratch);
        if (split < count)
            TailTile::pack_tiles(distance, ids + split, count - split, stride,
                                 codes + static_cast<size_t>(split) * stride, norms + split, scratch);
    }

    template <class Compute>
    static void compute_tiles(int count, size_t stride, const Code *codes, const int *norms, float *result,
                              Compute compute) {
        for (int i = 0; i < count; i += rows)
            compute(codes + static_cast<size_t>(i) * stride, norms + i, result + i);
    }

    static void store_scores(int query_norm, const int *norms, const std::array<int32_t, rows> &dots, float *result) {
        for (int lane = 0; lane < rows; ++lane)
            result[lane] = static_cast<float>(query_norm + norms[lane] - 2 * dots[lane]);
    }
};

} // namespace rnndescent
