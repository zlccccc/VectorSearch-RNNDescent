#pragma once

#include <stdexcept>

namespace rnndescent::detail {

inline void throw_assertion_error(const char *message) {
    throw std::runtime_error(message);
}

} // namespace rnndescent::detail

#define RNNDESCENT_ASSERT_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            ::rnndescent::detail::throw_assertion_error(message); \
        } \
    } while (false)

#define RNNDESCENT_ASSERT(condition) RNNDESCENT_ASSERT_MSG((condition), #condition)
