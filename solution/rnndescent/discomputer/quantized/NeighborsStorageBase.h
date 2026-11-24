#pragma once

#include "../../common/Assertions.h"
#include "../../common/HugePages.h"
#include "../../common/Logger.h"
#include "../utils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace rnndescent {

template <typename CodeType, typename NormType, size_t CodeAlignment> class NeighborsStorageBase {
    static_assert(alignSize % CodeAlignment == 0 && CodeAlignment % alignof(int) == 0 &&
                  CodeAlignment % alignof(NormType) == 0);

  public:
    void prefetch_header() const { __builtin_prefetch(record_, 0, 3); }

    bool has_cache() const { return read_header() & kCached; }

    const int *neighbor_ids(uint32_t code_stride) const {
        const int header = read_header();
        if (header & kCached)
            return cached_ids(header & ~kCached, code_stride);
        return reinterpret_cast<const int *>(payload());
    }

    // One pool per index. The node table contains one pointer into this buffer per node.
    class Pool {
      public:
        Pool() = default;
        Pool(const Pool &) = delete;
        Pool &operator=(const Pool &) = delete;

        void reset() {
            decltype(bytes_)().swap(bytes_);
            code_stride_ = 0;
            used_bytes_ = 0;
            cache_budget_ = 0;
        }

        void initialize(uint32_t code_stride, const std::vector<size_t> &offsets, long long max_pool_size = 0,
                        const int *row_order = nullptr) {
            RNNDESCENT_ASSERT_MSG(bytes_.empty(), "neighbor pool must be reset before initialization");
            RNNDESCENT_ASSERT_MSG(code_stride > 0, "neighbor codes require a positive stride");
            code_stride_ = code_stride;
            const bool cache_enabled = max_pool_size > 0;

            long long required_size = 0;
            long long full_size = 0;
            // The code alignment depends on the previous record's end, so size the pool in write order.
            for (size_t i = 0; i + 1 < offsets.size(); ++i) {
                const size_t row = row_order ? row_order[i] : i;
                const auto count = static_cast<long long>(offsets[row + 1] - offsets[row]);
                if (count == 0)
                    continue;
                const auto ids = id_record_bytes(count);
                required_size += ids;
                if (cache_enabled)
                    full_size += ids + cache_bytes(count, code_stride_) + cache_padding(full_size, count);
            }

            long long target_size = cache_enabled ? std::max(required_size, std::min(full_size, max_pool_size)) : required_size;
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
                    target_size = required_size + (target_size - required_size) / 2;
                }
            }

            cache_budget_ = target_size - required_size;
            detail::try_huge_pages(bytes_.data(), bytes_.size());

            const long long cached_bytes = cache_enabled ? full_size - required_size : 0;
            const double cache_ratio = cached_bytes > 0 ? (100.0 * cache_budget_ / cached_bytes) : 0.0;
            const long long total_edges = offsets.empty() ? 0 : static_cast<long long>(offsets.back());
            Logger::info("%lld edges; total of %.2f%% neighbors can be cached; cache size %.2fG\n", total_edges, cache_ratio,
                         (double)target_size / 1024 / 1024 / 1024);
        }

      private:
        friend NeighborsStorageBase;

        std::vector<uint8_t, AlignedAllocator<uint8_t>> bytes_;
        uint32_t code_stride_ = 0;
        long long used_bytes_ = 0;
        long long cache_budget_ = 0;

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

        uint8_t *allocate_ptr(long long size) {
            RNNDESCENT_ASSERT_MSG(used_bytes_ + size <= static_cast<long long>(bytes_.size()), "neighbor pool allocation overflow");
            auto *ptr = bytes_.data() + used_bytes_;
            used_bytes_ += size;
            return ptr;
        }
    };

    NeighborsStorageBase() = default;
    NeighborsStorageBase(Pool &storage, const int *neighbor, int count, const std::vector<int> &original_to_bfs) {
        if (count == 0)
            return;
        RNNDESCENT_ASSERT_MSG(count > 0 && count % 4 == 0, "neighbor count must be a multiple of four");

        long long padding = 0;
        long long payload_bytes = 0;
        if (storage.cache_budget_ > 0) {
            const auto gap = cache_padding(storage.used_bytes_, count);
            const auto payload = cache_bytes(count, storage.code_stride_);
            if (gap + payload <= storage.cache_budget_) {
                padding = gap;
                payload_bytes = payload;
                storage.cache_budget_ -= gap + payload;
            }
        }

        record_ = storage.allocate_ptr(padding + payload_bytes + id_record_bytes(count)) + padding;
        ::new (record_) int(count | (payload_bytes > 0 ? kCached : 0));
        auto *edges = reinterpret_cast<int *>(record_ + sizeof(int) + payload_bytes);
        for (int i = 0; i < count; ++i)
            edges[i] = original_to_bfs[neighbor[i]];
    }

  protected:
    // The degree is a multiple of four, leaving bit zero free for the cache flag.
    static constexpr int kCached = 1;

    // Cached records are [count/flag][norms][packed codes][IDs]. Uncached records are [count][IDs].
    // Any alignment padding precedes the record.
    int read_header() const { return *reinterpret_cast<const int *>(record_); }
    uint8_t *payload() const { return record_ + sizeof(int); }

    CodeType *cached_codes(int count) const {
        return reinterpret_cast<CodeType *>(payload() + norm_bytes(count));
    }

    const int *cached_ids(int count, uint32_t code_stride) const {
        return reinterpret_cast<const int *>(payload() + cache_bytes(count, code_stride));
    }

  private:
    inline static int empty_header_ = 0;
    uint8_t *record_ = reinterpret_cast<uint8_t *>(&empty_header_);

    static long long id_record_bytes(long long count) { return sizeof(int) + count * sizeof(int); }
    static long long norm_bytes(long long count) { return count * sizeof(NormType); }

    static long long cache_bytes(long long count, uint32_t code_stride) {
        return norm_bytes(count) + count * code_stride * sizeof(CodeType);
    }

    static long long cache_padding(long long offset, long long count) {
        const auto code_offset = offset + sizeof(int) + norm_bytes(count);
        return (CodeAlignment - code_offset % CodeAlignment) % CodeAlignment;
    }
};

} // namespace rnndescent
