/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>

#include "util/container.hpp"
#include "util/span.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "../external/ips4o/ips4o.hpp"
#pragma GCC diagnostic pop

namespace sacabench::util::sort {

template <typename T, typename Compare>
void ips4o_sort(container<T>& data, Compare comp) {
    ips4o::sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void ips4o_sort(span<T> data, Compare comp) {
    ips4o::sort(std::begin(data), std::end(data), comp);
}

} // namespace sacabench::util::sort

/******************************************************************************/
