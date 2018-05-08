#pragma once

#include <sstream>

namespace sacabench::util {

#ifdef DEBUG
#ifndef DCHECK
/// Internal macro for checking a boolean value
/// and printing the two comparison values.
#define DCHECK_BINARY(x,y,z)                                     \
    if (!(x)) {                                                  \
        std::stringstream msg;                                   \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << ":\n";                                            \
        msg << ("the check failed: " #x) ;                       \
        msg << ", we got " << y << " vs " << z;                  \
        throw std::runtime_error(msg.str());                     \
    }
/// Macro for checking a boolean value.
#define DCHECK(x)                                                \
    if (!(x)) {                                                  \
        std::stringstream msg;                                   \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << ":\n";                                            \
        msg << ("the check failed: " #x) ;                       \
        throw std::runtime_error(msg.str());                     \
    }
/// Macro for checking a boolean value
/// and printing a custom error message
#define DCHECK_MSG(x, s)                                         \
    if (!(x)) {                                                  \
        std::stringstream msg;                                   \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << ":\n";                                            \
        msg << "the check failed: ";                             \
        msg << s;                                                \
        throw std::runtime_error(msg.str());                     \
    }
#endif //DCHECK
#else //DEBUG
// Define macros as empty
#define DCHECK_BINARY(x,y,z)
#define DCHECK(x)
#define DCHECK_MSG(x, s)
#endif //DEBUG

/// Check for equality (==)
#define DCHECK_EQ(x, y) DCHECK_BINARY((x) == (y), x, y)
/// Check for inequality (!=)
#define DCHECK_NE(x, y) DCHECK_BINARY((x) != (y), x, y)
/// Check for less-than or equal (<=)
#define DCHECK_LE(x, y) DCHECK_BINARY((x) <= (y), x, y)
/// Check for less-than (<)
#define DCHECK_LT(x, y) DCHECK_BINARY((x) <  (y), x, y)
/// Check for greater-than or equal (>=)
#define DCHECK_GE(x, y) DCHECK_BINARY((x) >= (y), x, y)
/// Check for greater-than (<)
#define DCHECK_GT(x, y) DCHECK_BINARY((x) >  (y), x, y)

template<typename integer_type>
bool can_represent_all_values(uint64_t distinct_values) {
    return std::numeric_limits<integer_type>::max() >= (distinct_values - 1);
}

}
