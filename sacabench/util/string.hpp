/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>

#include "util/container.hpp"
#include "util/span.hpp"

namespace sacabench::util {
using character = unsigned char;

/// A container of characters
using string = container<character>;

/// A span (pointer+len pair) to a sequence of
/// read-only characters.
using string_span = span<character const>;

/// Creates a `string` from a `string_span`.
///
/// Example:
/// ```
/// string s = make_string("hello"_s);
/// ```
inline string make_string(string_span s) { return s; }

/// Creates a `string` from a C-string literal.
///
/// Example:
/// ```
/// string s = make_string("hello");
/// ```
inline string make_string(char const* cs) {
    string_span s{(character const*)cs, std::strlen(cs)};
    return s;
}

/// Special `character` values that is smaller than all possible
/// input characters.
constexpr character SENTINEL = 0;

} // namespace sacabench::util

/// Custom literal operator for creating a `string_span`.
///
/// This mainly exists for writing tests and debug code.
///
/// Example:
/// ```
/// string_span s = "hello"_s;
/// ```
inline sacabench::util::string_span operator"" _s(char const* ptr,
                                                  size_t length) {
    using namespace sacabench::util;
    return string_span((character const*)(ptr), length);
}

/// Custom `std::ostream` operator for a `string_span`
inline std::ostream& operator<<(std::ostream& out,
                                sacabench::util::string_span const& span) {
    return out.write((char const*)span.data(), span.size());
}

/******************************************************************************/
