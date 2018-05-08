/*******************************************************************************
 * bench/container.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>
#include "span.hpp"

namespace sacabench::util {
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

    /**\brief Creates a container as a copy of the elements of a `span`.
    */
    template <typename element_type>
    container<element_type> make_container(span<element_type> s) {
        container<element_type> r;
        r.reserve(s.size());
        r.resize(s.size());
        std::copy(s.begin(), s.end(), r.begin());
        return r;
    }
}

/******************************************************************************/
