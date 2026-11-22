#pragma once

#include "../../common/Assertions.h"
#include "../../common/HugePages.h"
#include "../../common/Logger.h"
#include "../MyDistanceComputer.h"
#include "../utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace rnndescent {

template <typename Derived, typename CodeType, typename NormType> struct NeighborsStorageBase {
    int *edges = nullptr;
    NormType *l2norms = nullptr;
    CodeType *pool = nullptr;
    uint32_t code_stride = 0;
    int size = 0;
    MyDistanceComputer *dis = nullptr;

    // One pool per index. Neighborhoods refer to fixed slices of this buffer.
    class Pool {
      public:
        Pool() = default;
        Pool(const Pool &) = delete;
        Pool &operator=(const Pool &) = delete;

        void reset() {
            decltype(bytes_)().swap(bytes_);
            code_stride_ = 0;
            pool_start_ptr = 0;
            preserve_limit = 0;
        }

        void initialize(uint32_t code_stride, const std::vector<size_t> &offsets, long long max_pool_size = 0,
                        bool save_neighbor = true) {
            RNNDESCENT_ASSERT_MSG(bytes_.empty(), "neighbor pool must be reset before initialization");
            RNNDESCENT_ASSERT_MSG(code_stride % Derived::cache_lane_width == 0, "neighbor codes must occupy whole SIMD blocks");
            code_stride_ = code_stride;
            const bool cache_enabled = save_neighbor && max_pool_size > 0;

            long long total_edges = 0;
            long long required_size = 0;
            long long cached_bytes = 0;
            for (size_t i = 1; i < offsets.size(); ++i) {
                const auto count = static_cast<long long>(offsets[i] - offsets[i - 1]);
                total_edges += count;
                required_size += aligned_size(count * sizeof(int));
                if (cache_enabled)
                    cached_bytes += additional_cache_bytes(count, code_stride_);
            }

            const long long requested_size = std::min(required_size + cached_bytes, max_pool_size);
            long long target_size = std::max(required_size, requested_size);

            const long long available_bytes = read_mem_available_bytes();
            if (available_bytes > 0) {
                constexpr long long kSystemReserveBytes = 2ll * 1024 * 1024 * 1024;
                const long long usable_bytes = std::max(0ll, available_bytes - kSystemReserveBytes);
                const long long checked_cap = std::max(required_size, usable_bytes);
                if (target_size > checked_cap) {
                    Logger::info("neighbor cache pool capped by free memory: requested %.2fG -> %.2fG (available %.2fG)\n",
                                 (double)target_size / 1024 / 1024 / 1024, (double)checked_cap / 1024 / 1024 / 1024,
                                 (double)available_bytes / 1024 / 1024 / 1024);
                    target_size = checked_cap;
                }
            }

            while (true) {
                try {
                    bytes_.resize(target_size);
                    break;
                } catch (const std::bad_alloc &) {
                    if (target_size <= required_size)
                        throw;
                    const long long reduced_cache = std::max(0ll, (target_size - required_size) / 2);
                    target_size = required_size + reduced_cache;
                    target_size = (target_size / alignSize) * alignSize;
                }
            }

            preserve_limit = target_size - required_size;
            if (preserve_limit > 0)
                detail::try_huge_pages(bytes_.data(), bytes_.size());

            const double cache_ratio = cached_bytes > 0 ? (100.0 * preserve_limit / cached_bytes) : 0.0;
            Logger::info("%lld edges; total of %.2f%% neighbors can be cached; cache size %.2fG\n", total_edges, cache_ratio,
                         (double)target_size / 1024 / 1024 / 1024);
        }

      private:
        friend NeighborsStorageBase;

        std::vector<uint8_t, AlignedAllocator<uint8_t>> bytes_;
        uint32_t code_stride_ = 0;
        long long pool_start_ptr = 0;
        long long preserve_limit = 0;

        static long long read_mem_available_bytes() {
            std::ifstream meminfo("/proc/meminfo");
            if (!meminfo.is_open())
                return -1;

            std::string key;
            long long value = 0;
            std::string unit;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemAvailable:")
                    return value * 1024;
            }
            return -1;
        }

        static long long aligned_size(long long bytes) { return (bytes + alignSize - 1) / alignSize * alignSize; }

        // Reserve the incremental space used by a cached neighborhood, including padding before the next one.
        static long long additional_cache_bytes(long long count, uint32_t code_stride) {
            const long long edge_bytes = count * sizeof(int);
            const long long cache_bytes = count * (static_cast<long long>(code_stride) * sizeof(CodeType) + sizeof(NormType));
            return aligned_size(edge_bytes + cache_bytes) - aligned_size(edge_bytes);
        }

        void *allocate_ptr(size_t size) {
            RNNDESCENT_ASSERT_MSG(pool_start_ptr + (long long)size <= (long long)bytes_.size(), "neighbor pool allocation overflow");
            void *ptr = bytes_.data() + pool_start_ptr;
            pool_start_ptr += (long long)size;
            return ptr;
        }
    };

    NeighborsStorageBase() = default;

    NeighborsStorageBase(Pool &storage, const int *neighbor, int count, MyDistanceComputer *dis,
                         const std::vector<int> &original_to_bfs, bool save_neighbor)
        : code_stride(storage.code_stride_), size(count), dis(dis) {
        if (size == 0)
            return;
        RNNDESCENT_ASSERT_MSG(!storage.bytes_.empty(), "neighbors pool has not been initialized");
        RNNDESCENT_ASSERT_MSG(size % 4 == 0, "neighbor cache size must be a multiple of 4");

        storage.pool_start_ptr = Pool::aligned_size(storage.pool_start_ptr);
        const long long cache_cost = Pool::additional_cache_bytes(size, code_stride);
        if (save_neighbor && cache_cost <= storage.preserve_limit) {
            pool = reinterpret_cast<CodeType *>(storage.allocate_ptr(static_cast<size_t>(size) * code_stride * sizeof(CodeType)));
            l2norms = reinterpret_cast<NormType *>(storage.allocate_ptr(static_cast<size_t>(size) * sizeof(NormType)));
            storage.preserve_limit -= cache_cost;
        }

        edges = reinterpret_cast<int *>(storage.allocate_ptr(size * sizeof(int)));
        for (int i = 0; i < size; ++i)
            edges[i] = original_to_bfs[neighbor[i]];
    }
};

} // namespace rnndescent
