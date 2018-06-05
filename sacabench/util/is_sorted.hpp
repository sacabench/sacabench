/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "container.hpp"
#include "span.hpp"
#include "sort/std_sort.hpp"

namespace sacabench::util {
// Checks the sorting of the array in O(n).
// Returns true if correctly sorted.
template <typename content, typename Compare = std::less<content>>
bool is_sorted(const span<content> array, Compare less = Compare()) {
    for (size_t i = 1; i < array.size(); ++i) {
        if (less(array[i], array[i-1])) {
            // if A[i-1] > A[i], the array cannot be sorted.
            return false;
        }
    }
    return true;
}

template <typename sa_index_type>
bool is_partially_suffix_sorted(const span<sa_index_type> array, util::string_span text) {
    auto copy = make_container(array);

    // Construct a SA by sorting according
    // to the suffix starting at that index.
    sort::std_sort(copy, compare_key([&](size_t i) {
        return text.slice(i);
    }));

    return array == span<sa_index_type>(copy);
}
} // namespace sacabench::util
