#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace benchmark {

inline void mark_query_window(const char *phase) {
    if (!std::getenv("RNNDESCENT_PROFILE_WINDOW"))
        return;
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        throw std::runtime_error("could not read the monotonic profiling clock");
    std::printf("query_window %s %lld.%09ld\n", phase, static_cast<long long>(now.tv_sec), now.tv_nsec);
    std::fflush(stdout);
}

} // namespace benchmark
