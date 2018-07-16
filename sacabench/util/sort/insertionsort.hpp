/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#include "util/compare.hpp"
#include "util/span.hpp"
#include <util/signed_size_type.hpp>

namespace sacabench::util::sort {

template <typename T, typename F = std::less<T>>
void insertion_sort(span<T> A, F compare_fun = F()) {
    // Adapter for "a > b"
    auto greater = as_greater(compare_fun);
    for (size_t i = 1; i < A.size(); i++) {
        auto to_sort = A[i];
        auto j = i;
        while ((j > 0) && greater(A[j - 1], to_sort)) {
            A[j] = A[j - 1];
            j = j - 1;
        }
        A[j] = to_sort;
    }
}

template <typename Content, typename Compare = std::less<Content>>
inline void insertion_sort2(util::span<Content> data,
                            Compare less = Compare()) {
    // Invariant: data[0 .. end_of_sorted_partition] is already correctly
    // sorted.
    for (size_t end_of_sorted_partition = 1;
         end_of_sorted_partition < data.size(); ++end_of_sorted_partition) {
        DCHECK(is_sorted(data.slice(0, end_of_sorted_partition), less));

        // end_of_sorted_partition is now the new element.
        // bubble it up to the front.
        for (ssize i = end_of_sorted_partition; i > 0; --i) {

            // Check if they defy the ordering and swap them if needed
            if (less(data[i], data[i - 1])) {
                std::swap(data[i], data[i - 1]);
            }
        }
    }
}

template <typename Content, typename Compare = std::less<Content>>
inline void insertion_sort_hybrid(util::span<Content> data,
                                  Compare less = Compare()) {
    if (data.size() < 15) {
        insertion_sort2(data, less);
    } else {
        insertion_sort(data, less);
    }
}
} // namespace sacabench::util::sort
