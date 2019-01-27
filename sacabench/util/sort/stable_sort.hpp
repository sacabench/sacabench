/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>

#include "util/container.hpp"
#include "util/span.hpp"

/*
    Wrapper class for std::stable_sort
*/

namespace sacabench::util::sort {

// Wrapper call for using a container
template <typename T, typename Compare>
void stable_sort(container<T>& data, Compare comp) {
    std::stable_sort(std::begin(data), std::end(data), comp);
}

// Wrapper call for using a span
template <typename T, typename Compare>
void stable_sort(span<T> data, Compare comp) {
    std::stable_sort(std::begin(data), std::end(data), comp);
}

}
