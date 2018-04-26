/*******************************************************************************
 * bench/container.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>

template <typename element_type>
using container = std::vector<element_type>;

/**\brief Creates a container with space for exactly for `size` elements.
 */
template <typename element_type>
container<element_type> make_container(size_t size) {
    container<element_type> r;
    r.reserve(size);
    r.resize(size);
    return r;
}

/******************************************************************************/
