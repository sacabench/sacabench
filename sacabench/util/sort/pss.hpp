/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *                    David Piper <david.piper@tu-dortmund.de>
 *                    Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>

#include "util/container.hpp"
#include "util/span.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wdangling-else"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wsign-compare"
#define USE_OPENMP
#include "../external/pss/parallel_stable_sort.h"
#pragma GCC diagnostic pop

namespace sacabench::util::sort {

template <typename T, typename Compare>
void parallel_stable(container<T>& data, Compare comp) {
    pss::parallel_stable_sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void parallel_stable(span<T> data, Compare comp) {
    pss:parallel_stable_sort(std::begin(data), std::end(data), comp);
}

} // namespace sacabench::util::sort

/******************************************************************************/
