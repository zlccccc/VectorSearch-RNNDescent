#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace rnndescent::detail {

// Use only whole 2 MiB spans within an owned, populated allocation.
inline void try_huge_pages([[maybe_unused]] void *buffer, [[maybe_unused]] std::size_t size) noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr std::size_t huge_page_size = 2 * 1024 * 1024;
    if (size < huge_page_size || ::sysconf(_SC_PAGESIZE) != 4096)
        return;

    const auto address = reinterpret_cast<std::uintptr_t>(buffer);
    const auto prefix = (huge_page_size - address % huge_page_size) % huge_page_size;
    const auto length = (size - prefix) / huge_page_size * huge_page_size;
    if (length == 0)
        return;

    auto *pages = static_cast<unsigned char *>(buffer) + prefix;
    if (::madvise(pages, length, MADV_HUGEPAGE) != 0)
        return;
#if defined(MADV_COLLAPSE)
    // Try synchronous promotion before search. Failed spans keep their backing.
    (void)::madvise(pages, length, MADV_COLLAPSE);
#endif
#endif
}

} // namespace rnndescent::detail
