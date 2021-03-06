/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <parallel/algorithm>

#include "util/container.hpp"
#include "util/span.hpp"

namespace sacabench::util::sort {

template <typename T, typename Compare>
void std_sort(container<T>& data, Compare comp) {
    std::sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_sort(span<T> data, Compare comp) {
    std::sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_stable_sort(container<T>& data, Compare comp) {
    std::stable_sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_stable_sort(span<T> data, Compare comp) {
    std::stable_sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_par_sort(container<T>& data, Compare comp) {
    __gnu_parallel::sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_par_sort(span<T> data, Compare comp) {
    __gnu_parallel::sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_par_stable_sort(container<T>& data, Compare comp) {
    __gnu_parallel::stable_sort(std::begin(data), std::end(data), comp);
}

template <typename T, typename Compare>
void std_par_stable_sort(span<T> data, Compare comp) {
    __gnu_parallel::stable_sort(std::begin(data), std::end(data), comp);
}

} // namespace sacabench::util::sort

/******************************************************************************/
