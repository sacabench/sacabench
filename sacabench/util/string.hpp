/*******************************************************************************
 * bench/string.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstring>

#include "util/container.hpp"
#include "util/span.hpp"

namespace sacabench::util {
    using character = unsigned char;
    using string = container<character>;

    using string_span = span< character const >;

    inline string make_string(string_span s) {
        string r;
        r.reserve(s.size());
        r.resize(s.size());
        std::copy(s.begin(), s.end(), r.begin());
        return r;
    }

    inline string make_string(char const* cs) {
        string_span s { (character const*) cs, std::strlen(cs) };
        return make_string(s);
    }
}

/// Custom literal operator for creating a `string_span`.
///
/// Example:
/// ```
/// string_span s = "hello"_s;
/// ```
inline constexpr sacabench::util::string_span operator"" _s(
    char const* ptr, size_t length) {
    using namespace sacabench::util;
    return string_span((character const*)(ptr), length);
}

/******************************************************************************/
