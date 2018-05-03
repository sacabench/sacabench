/*******************************************************************************
 * util/sort.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>

#include "util/container.hpp"

namespace sacabench::util::sort {

template <typename T, typename Compare>
void std_sort(container<T>& data, Compare comp) {
    std::sort(std::begin(data), std::end(data), comp);
}

} // namespace sacabench::util::sort

/******************************************************************************/
