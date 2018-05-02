/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../sacabench/util/span.hpp"

// Checks the sorting of the array in O(n).
// Returns true if correctly sorted.
template <typename content, typename key_func_type>
bool is_sorted(const span<content> array, key_func_type cmp) {
    for (size_t i = 1; i < array.size(); ++i) {
        if (cmp(array[i - 1], array[i]) > 0) {
            return false;
        }
    }
    return true;
}
