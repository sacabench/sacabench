/*******************************************************************************
 * bench/string.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "util/container.hpp"

using character = unsigned char;
using string = container<character>;

using string_span = span< character const >;
inline constexpr string_span operator"" _s(
    char const* ptr, unsigned long length) {
    return string_span(reinterpret_cast<character const*>(ptr), length);
}

/******************************************************************************/
