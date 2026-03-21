#pragma once

#include <cstdarg>
#include <cstdio>

namespace rnndescent {

struct Logger {
    static void info(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
    }

    static void info(bool enabled, const char *fmt, ...) {
        if (!enabled)
            return;
        va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
    }

    static void line(bool enabled, const char *message) {
        if (!enabled)
            return;
        std::puts(message);
    }

    static void line(const char *message) { std::puts(message); }
};

} // namespace rnndescent
