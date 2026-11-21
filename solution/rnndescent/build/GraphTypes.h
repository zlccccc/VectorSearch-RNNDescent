#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

namespace rnndescent {

struct MutexWarpper : std::mutex {
    MutexWarpper() = default;
    MutexWarpper(MutexWarpper const &) noexcept : std::mutex() {}
    bool operator==(MutexWarpper const &other) noexcept { return this == &other; }
};

struct XNeighbor {
    int id_;
    float distance;

    XNeighbor() = default;
    XNeighbor(int id, float distance, bool flag) : id_(id), distance(distance) {
        if (flag)
            id_ |= 0x80000000;
    }

    int id() const { return id_ & 0x7fffffff; }
    bool flag() const { return (id_ & 0x80000000) != 0; }

    void setflag(bool flag) {
        if (flag)
            id_ |= 0x80000000;
        else
            id_ &= 0x7fffffff;
    }

    bool operator<(const XNeighbor &other) const {
        if (distance == other.distance)
            return (id_ & 0x7fffffff) < (other.id_ & 0x7fffffff);
        return distance < other.distance;
    }
};

using KNNGraph = std::vector<std::vector<XNeighbor>>;

inline void sort_and_merge_neighbors_impl(std::vector<XNeighbor> &neighbors) {
    std::sort(neighbors.begin(), neighbors.end());
    const auto same_edge = [](const XNeighbor &a, const XNeighbor &b) { return a.id() == b.id(); };
    auto kept = std::adjacent_find(neighbors.begin(), neighbors.end(), same_edge);
    if (kept == neighbors.end())
        return;
    for (auto next = kept + 1; next != neighbors.end(); ++next) {
        if (same_edge(*kept, *next)) {
            kept->id_ |= next->id_;
        } else {
            *++kept = *next;
        }
    }
    neighbors.erase(kept + 1, neighbors.end());
}

namespace detail {
inline void sort_and_merge_neighbors(std::vector<XNeighbor> &neighbors) {
    sort_and_merge_neighbors_impl(neighbors);
}
} // namespace detail

} // namespace rnndescent
