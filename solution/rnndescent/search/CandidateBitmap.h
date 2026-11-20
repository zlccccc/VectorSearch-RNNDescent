#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rnndescent {

// Records candidate IDs admitted during one query, using one bit per indexed point.
class alignas(64) CandidateBitmap {
  public:
    void initialize(std::size_t point_count) { words_.assign((point_count + 63) / 64, 0); }

    void clear() { std::fill(words_.begin(), words_.end(), 0); }

    bool insert(int id) {
        const auto index = static_cast<uint32_t>(id);
        const uint64_t mask = uint64_t{1} << (index % 64);
        auto &word = words_[index / 64];
        if (word & mask)
            return false;
        word |= mask;
        return true;
    }

  private:
    std::vector<uint64_t> words_;
};

} // namespace rnndescent
