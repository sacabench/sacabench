/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "span.hpp"

namespace sacabench::util {
// Checks the sorting of the array in O(n).
// Returns true if correctly sorted.
template <typename content, typename Compare>
bool is_sorted(const span<content> array, Compare less) {
    for (size_t i = 1; i < array.size(); ++i) {
        if (less(array[i], array[i-1])) {
            // if A[i-1] > A[i], the array cannot be sorted.
            return false;
        }
    }
    return true;
}
} // namespace sacabench::util
