#pragma once

#include "../Logger.h"
#include "MyDistanceComputer.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rnndescent {

template <typename Derived, typename CodeType, typename NormType> struct NeighborsStorageBase {
    int *edges = nullptr;
    NormType *l2norms = nullptr;
    CodeType *pool = nullptr;
    int dim = 0, size = 0;
    MyDistanceComputer *dis = nullptr;

    inline static std::vector<uint8_t, AlignedAllocator<uint8_t>> global_pool;
    inline static long long pool_start_ptr = 0;
    inline static long long preserve_limit = 0;

    NeighborsStorageBase() = default;

    NeighborsStorageBase(int dim, std::vector<int> &neighbor, MyDistanceComputer *dis, std::vector<int> &rollback_ids, bool save_neighbor) : dim(dim) {
        assert(global_pool.size() != 0);
        size = neighbor.size();
        this->dis = dis;
        assert(size % 4 == 0);

        alignment_ptr();
        if (save_neighbor && 1ll * size * dim * (long long)sizeof(CodeType) <= preserve_limit) {
            pool = reinterpret_cast<CodeType *>(allocate_ptr(size * dim * sizeof(CodeType)));
            l2norms = reinterpret_cast<NormType *>(allocate_ptr(size * sizeof(NormType)));
            preserve_limit -= 1ll * size * dim * (long long)sizeof(CodeType);
        }

        edges = reinterpret_cast<int *>(allocate_ptr(size * sizeof(int)));
        for (int i = 0; i < size; ++i)
            edges[i] = rollback_ids[neighbor[i]];

        if (pool != nullptr) {
#pragma omp parallel for
            for (int i = 0; i < size; ++i)
                dis->copy_index(edges[i], pool + i * dim, l2norms[i]);
        }
    }

    static void clear_memory() {
        std::vector<uint8_t, AlignedAllocator<uint8_t>>().swap(global_pool);
        pool_start_ptr = 0;
    }

    static void init_neighbors_pool(int dim, std::vector<std::vector<int>> &edges, long long max_pool_size = 16ll * 1024 * 1024 * 1024,
                                    bool save_neighbor = true) {
        constexpr long long kDefaultPoolCap = 2ll * 1024 * 1024 * 1024;
        pool_start_ptr = 0;

        long long total_edges = 0;
        long long required_size = 0;
        long long cached_bytes = 0;
        for (auto &nhood : edges) {
            total_edges += nhood.size();
            required_size += nhood.size() * (long long)sizeof(int);
            required_size = (required_size + alignSize - 1) / alignSize * alignSize;
            if (save_neighbor) {
                cached_bytes += nhood.size() * (long long)dim * sizeof(CodeType);
                cached_bytes += nhood.size() * (long long)sizeof(NormType);
            }
        }

        const long long configured_pool_cap = max_pool_size > 0 ? max_pool_size : kDefaultPoolCap;
        const long long requested_size = std::min(required_size + cached_bytes, configured_pool_cap);
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
                global_pool.clear();
                global_pool.resize(target_size);
                break;
            } catch (const std::bad_alloc &) {
                if (target_size <= required_size)
                    throw;
                const long long reduced_cache = std::max(0ll, (target_size - required_size) / 2);
                target_size = required_size + reduced_cache;
                target_size = (target_size / alignSize) * alignSize;
            }
        }

        if (save_neighbor)
            preserve_limit = std::max(0ll, target_size - required_size);
        else
            preserve_limit = 0;

        const double cache_ratio = (save_neighbor && cached_bytes > 0) ? (100.0 * preserve_limit / cached_bytes) : 0.0;
        Logger::info("%lld edges; total of %.2f%% neighbors can be cached; cache size %.2fG\n", total_edges, cache_ratio,
                     (double)target_size / 1024 / 1024 / 1024);
    }

  protected:
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

    static void alignment_ptr() { pool_start_ptr = (pool_start_ptr + alignSize - 1) / alignSize * alignSize; }

    static void *allocate_ptr(size_t size) {
        assert(pool_start_ptr + (long long)size <= (long long)global_pool.size());
        void *ptr = &global_pool[(size_t)pool_start_ptr];
        pool_start_ptr += (long long)size;
        return ptr;
    }
};

} // namespace rnndescent
