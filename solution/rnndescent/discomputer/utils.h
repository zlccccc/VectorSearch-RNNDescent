#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace rnndescent {

// constexpr int alignSize = 16;
// constexpr int alignSize = 32;
constexpr int alignSize = 64;
// constexpr int alignSize = 128;
// constexpr int alignSize = 256;
template <typename T> class AlignedAllocator {
  public:
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <typename U> AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        return static_cast<T *>(::operator new(n * sizeof(T), alignment));
    }

    void deallocate(T *p, std::size_t) noexcept { ::operator delete(p, alignment); }

    template <typename U> bool operator==(const AlignedAllocator<U> &) const noexcept { return true; }

    template <typename U> bool operator!=(const AlignedAllocator<U> &) const noexcept { return false; }

  private:
    static constexpr std::align_val_t alignment{std::max<std::size_t>(alignSize, alignof(T))};
};

/// set implementation optimized for fast access.
struct MyVisitedTable {
    std::vector<uint8_t> visited;
    uint8_t visno;
    MyVisitedTable() {};
    MyVisitedTable(int size) { init(size); };

    /// set flag #no to true
    void set(int no) { visited[no] = visno; }

    /// get flag #no
    bool get(int no) const { return visited[no] == visno; }

    void init(int size) {
        visited.resize(size);
        std::fill(visited.begin(), visited.end(), 0);
        visno = 1;
    }

    /// reset all flags to false
    void advance() {
        visno++;
        if (visno == 250) {
            // 250 rather than 255 because sometimes we use visno and visno+1
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};

struct SingleNeighbor {
    int id;
    float distance;

    SingleNeighbor() = default;
    SingleNeighbor(int id, float distance) : id(id), distance(distance) {}

    inline bool operator<(const SingleNeighbor &other) const {
        return distance < other.distance;
    }

    // Insert a new point into the candidate pool in ascending order
    static int insert_into_pool(SingleNeighbor *addr, int size, SingleNeighbor nn) {
        // find the location to insert
        int left = 0, right = size - 1;
        if (addr[left].distance >= nn.distance) {
            memmove(&addr[left + 1], &addr[left], (size - 1) * sizeof(SingleNeighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            // addr[size] = nn;
            return size;
        }
        while (left + 1 < right) {
            int mid = (left + right) / 2;
            if (addr[mid].distance >= nn.distance)
                right = mid;
            else
                left = mid;
        }
        memmove(&addr[right + 1], &addr[right], (size - 1 - right) * sizeof(SingleNeighbor));
        addr[right] = nn;
        return right;
    }
};

} // namespace rnndescent
