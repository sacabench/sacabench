/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <sstream>

#include "stacktrace.hpp"

namespace sacabench::util {

#ifdef DEBUG
#ifndef DCHECK
/// Internal macro for checking a boolean value
/// and printing the two comparison values.
#define DCHECK_BINARY(x, y, z)                                                 \
    if (!(x)) {                                                                \
        std::cerr << ::sacabench::backtrace::Backtrace(1);                     \
        std::stringstream msg;                                                 \
        msg << " in file " << __FILE__ << ":" << __LINE__;                     \
        msg << ":\n";                                                          \
        msg << ("the check failed: " #x);                                      \
        msg << ", we got " << y << " vs " << z;                                \
        throw std::runtime_error(msg.str());                                   \
    }
/// Macro for checking a boolean value.
#define DCHECK(x)                                                              \
    if (!(x)) {                                                                \
        std::cerr << ::sacabench::backtrace::Backtrace(1);                     \
        std::stringstream msg;                                                 \
        msg << " in file " << __FILE__ << ":" << __LINE__;                     \
        msg << ":\n";                                                          \
        msg << ("the check failed: " #x);                                      \
        throw std::runtime_error(msg.str());                                   \
    }
/// Macro for checking a boolean value
/// and printing a custom error message
#define DCHECK_MSG(x, s)                                                       \
    if (!(x)) {                                                                \
        std::cerr << ::sacabench::backtrace::Backtrace(1);                     \
        std::stringstream msg;                                                 \
        msg << " in file " << __FILE__ << ":" << __LINE__;                     \
        msg << ":\n";                                                          \
        msg << "the check failed: ";                                           \
        msg << s;                                                              \
        throw std::runtime_error(msg.str());                                   \
    }
#endif // DCHECK
#else  // DEBUG
// Define macros as empty
#define DCHECK_BINARY(x, y, z)
#define DCHECK(x)
#define DCHECK_MSG(x, s)
#endif // DEBUG

/// Check for equality (==)
#define DCHECK_EQ(x, y) DCHECK_BINARY((x) == (y), x, y)
/// Check for inequality (!=)
#define DCHECK_NE(x, y) DCHECK_BINARY((x) != (y), x, y)
/// Check for less-than or equal (<=)
#define DCHECK_LE(x, y) DCHECK_BINARY((x) <= (y), x, y)
/// Check for less-than (<)
#define DCHECK_LT(x, y) DCHECK_BINARY((x) < (y), x, y)
/// Check for greater-than or equal (>=)
#define DCHECK_GE(x, y) DCHECK_BINARY((x) >= (y), x, y)
/// Check for greater-than (>)
#define DCHECK_GT(x, y) DCHECK_BINARY((x) > (y), x, y)

template <typename integer_type>
bool can_represent_all_values(uint64_t distinct_values) {
    return std::numeric_limits<integer_type>::max() >= (distinct_values - 1);
}

/// \brief Call this function once at the start of your SACA to check if the
///        amount of bits your algorithm uses for metadata (tagging, ...) is
///        usable because the text is short enough.
template<typename integer_type>
bool assert_text_length(const size_t text_length, const size_t reserved_bits) {

    // Actually, max_text_len is one larger than this number.
    // Therefore, we substract one below.
    const integer_type max_text_len = (std::numeric_limits<integer_type>::max() >> reserved_bits);
    
    // std::cout << (size_t)max_text_len << std::endl;

    if(text_length == 0) { return true; }
    return text_length - 1 <= max_text_len;
}

} // namespace sacabench::util
