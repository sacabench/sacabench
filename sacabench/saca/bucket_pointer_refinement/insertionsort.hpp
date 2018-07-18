/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#include "util/compare.hpp"
#include "util/span.hpp"

namespace sacabench::bucket_pointer_refinement::sort {

template <typename T, typename F = std::less<T>>
inline void insertion_sort(span<T> A, F compare_fun = F()) {
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
} // namespace sacabench::util::sort
