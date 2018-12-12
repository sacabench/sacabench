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
void insertion_sort(span<T> A, F compare_fun = F()) SB_NOEXCEPT {
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
