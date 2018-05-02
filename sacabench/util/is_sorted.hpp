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
template <typename content, typename key_func_type>
bool is_sorted(const span<content> array, key_func_type cmp) {
    for (size_t i = 1; i < array.size(); ++i) {
        // result <= 0 means correctly sorted
        // result > 0 means array[i] is smaller than array[i-1]
        bool result = cmp(array[i - 1], array[i]) > 0;
        if (result) {
            return false;
        }
    }
    return true;
}
} // namespace sacabench::util
