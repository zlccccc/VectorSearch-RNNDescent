#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rnndescent {

// Tracks the current candidate pool and the pending batch for one search thread.
class alignas(64) ActiveCandidateIds {
  public:
    void reset(std::size_t max_ids) {
        const auto slots_in_use = ids_.size();
        if (slots_in_use != 0 && max_ids <= slots_in_use / 2 && (slots_in_use == 2 || max_ids > slots_in_use / 4)) {
            std::fill(ids_.begin(), ids_.end(), empty_id);
            return;
        }

        // The caller bounds max_ids by the number of indexed points, which fits in an int.
        std::size_t slots = 2;
        shift_ = 31;
        while (slots / 2 < max_ids) {
            slots *= 2;
            --shift_;
        }
        ids_.assign(slots, empty_id);
        mask_ = static_cast<uint32_t>(slots - 1);
    }

    bool insert(int id) {
        const uint32_t slot = find(id);
        if (ids_[slot] == id)
            return false;
        ids_[slot] = id;
        return true;
    }

    void erase(int id) {
        uint32_t hole = find(id);
        assert(ids_[hole] == id);
        // Move an entry back only when the hole lies on its probe path.
        // This keeps later lookups correct without accumulating tombstones.
        for (uint32_t next = (hole + 1) & mask_; ids_[next] != empty_id; next = (next + 1) & mask_) {
            const uint32_t home = bucket(ids_[next]);
            if (((hole - home) & mask_) < ((next - home) & mask_)) {
                ids_[hole] = ids_[next];
                hole = next;
            }
        }
        ids_[hole] = empty_id;
    }

  private:
    static constexpr int empty_id = -1;
    std::vector<int> ids_;
    uint32_t mask_ = 0;
    unsigned shift_ = 31;

    uint32_t bucket(int id) const { return (static_cast<uint32_t>(id) * 0x9e3779b1u) >> shift_; }

    uint32_t find(int id) const {
        uint32_t slot = bucket(id);
        while (ids_[slot] != empty_id && ids_[slot] != id)
            slot = (slot + 1) & mask_;
        return slot;
    }
};

} // namespace rnndescent
