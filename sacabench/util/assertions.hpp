#pragma once

#include <stringstream>

namespace sacabench::util {

#ifdef DEBUG
#ifndef DCHECK
/// Internal macro for checking a boolean value
/// and printing two comparison values.
#define DCHECK_(x,y,z)                                           \
    if (!(x)) {                                                  \
        std::sstream msg;                                        \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << ("the check failed: " #x) ;                       \
        msg << ", we got " << y << " vs " << z;                  \
        throw std::runtime_error(msg.str());                     \
    }
/// Macro for checking a boolean value.
#define DCHECK(x)                                                \
    if (!(x)) {                                                  \
        std::sstream msg;                                        \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << ("the check failed: " #x) ;                       \
        throw std::runtime_error(msg.str());                     \
    }
/// Macro for checking a boolean value, and printing a custom error message
#define DCHECK_MSG(x, s)                                         \
    if (!(x)) {                                                  \
        std::sstream msg;                                        \
        msg << " in file " << __FILE__ << ":" << __LINE__;       \
        msg << "the check failed: ";                             \
        msg << s;                                                \
        throw std::runtime_error(msg.str());                     \
    }
#endif //DCHECK
#else //DEBUG
// Define macros as empty
#define DCHECK_(x,y,z)
#define DCHECK(x)
#define DCHECK_MSG(x)
#endif //DEBUG

/// Check for equality (==)
#define DCHECK_EQ(x, y) DCHECK_((x) == (y), x,y)
/// Check for inequality (!=)
#define DCHECK_NE(x, y) DCHECK_((x) != (y), x,y)
/// Check for less-than or equal (<=)
#define DCHECK_LE(x, y) DCHECK_((x) <= (y), x,y)
/// Check for less-than (<)
#define DCHECK_LT(x, y) DCHECK_((x) < (y) ,x,y)
/// Check for greater-than or equal (>=)
#define DCHECK_GE(x, y) DCHECK_((x) >= (y),x,y )
/// Check for greater-than (<)
#define DCHECK_GT(x, y) DCHECK_((x) > (y) ,x,y)

}
