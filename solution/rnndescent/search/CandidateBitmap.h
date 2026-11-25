#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rnndescent {

// Records candidate IDs admitted during one query, using one bit per indexed point.
class alignas(64) CandidateBitmap {
  public:
    void initialize(std::size_t point_count) {
        const auto block_count = (point_count + ids_per_block - 1) / ids_per_block;
        words_.assign(block_count * words_per_block, 0);
        block_generations_.assign(block_count, 0);
        generation_ = 0;
    }

    void clear() { std::fill(words_.begin(), words_.end(), 0); }

    void begin_lazy_query() {
        if (++generation_ == 0) {
            std::fill(block_generations_.begin(), block_generations_.end(), 0);
            generation_ = 1;
        }
    }

    // Each block holds 512 IDs. Initialize its bits on the first access in a query.
    template <bool LazyReset = false>
    bool insert(int id) {
        const auto index = static_cast<uint32_t>(id);
        if constexpr (LazyReset) {
            const auto block = index / ids_per_block;
            if (block_generations_[block] != generation_) {
                std::fill_n(words_.data() + block * words_per_block, words_per_block, 0);
                block_generations_[block] = generation_;
            }
        }
        const uint64_t mask = uint64_t{1} << (index % 64);
        auto &word = words_[index / 64];
        if (word & mask)
            return false;
        word |= mask;
        return true;
    }

  private:
    static constexpr uint32_t words_per_block = 8;
    static constexpr uint32_t ids_per_block = 64 * words_per_block;
    std::vector<uint64_t> words_;
    std::vector<uint8_t> block_generations_;
    uint8_t generation_ = 0;
};

// This shared hint allows concurrent first visits. Its byte accesses use plain
// machine loads and stores, with no ordering guarantees for other data.
class SharedVisitedSet {
  public:
    explicit SharedVisitedSet(std::size_t point_count) : marks_(point_count, 0) {}

    void begin_query() {
        if (++generation_ == 0) {
            std::fill(marks_.begin(), marks_.end(), 0);
            generation_ = 1;
        }
    }

    bool insert(int id) {
        auto &mark = marks_[id];
        uint8_t seen;
        // Use explicit memory operands for the deliberately racy visit hint.
#if defined(__x86_64__)
        __asm__ volatile("mov{b %1, %0| %0, %1}" : "=q"(seen) : "m"(mark));
#elif defined(__aarch64__)
        __asm__ volatile("ldrb %w0, %1" : "=r"(seen) : "Q"(mark));
#else
#error "Shared visited marks require x86-64 or AArch64"
#endif
        if (seen == generation_)
            return false;
#if defined(__x86_64__)
        __asm__ volatile("mov{b %1, %0| %0, %1}" : "=m"(mark) : "q"(generation_));
#elif defined(__aarch64__)
        __asm__ volatile("strb %w1, %0" : "=Q"(mark) : "r"(generation_));
#endif
        return true;
    }

  private:
    std::vector<uint8_t> marks_;
    uint8_t generation_{0};
};

} // namespace rnndescent
