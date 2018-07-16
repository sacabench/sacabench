/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>

namespace sacabench::util::sort::binary_quicksort {
template<typename Content, typename Compare = std::less<Content>>
inline size_t partition(util::span<Content> data, const Content& pivot, Compare less = Compare()) {
    size_t left = 0;
    size_t right = data.size() - 1;

    for(;;) {
        while(left < right && left < data.size() && less(data[left], pivot)) ++left;
        while(left < right && right > 0 && less(pivot, data[right])) --right;
        if(left < right && left < data.size()) {
            std::swap(data[left], data[right]);
        } else {
            break;
        }
    }

    return left;
}
}
