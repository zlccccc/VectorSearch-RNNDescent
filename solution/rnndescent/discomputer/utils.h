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

struct SingleNeighbor {
    int id;
    float distance;

    SingleNeighbor() = default;
    SingleNeighbor(int id, float distance) : id(id), distance(distance) {}

    inline bool operator<(const SingleNeighbor &other) const {
        return distance < other.distance;
    }
};

} // namespace rnndescent
