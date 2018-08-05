/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

namespace sacabench::util::sort {

/// \brief The amount of elements in the array, from which median_of_nine is
///        used instead of median_of_three.
constexpr size_t MEDIAN_OF_NINE_THRESHOLD = 40;

/**\brief Returns pseudo-median according to three values
 * \param array array of elements
 * \param key_func key function for comparing elements with min/max methods
 *
 * Chooses the median of the given array by the median-of-three method
 * which chooses the median of the first, middle and last element of the array
 */
template <typename content, typename Compare>
const content& median_of_three(const span<content> array, Compare cmp) {
    using std::max;
    using std::min;

    const content& first = array[0];
    const content& middle = array[(array.size() - 1) / 2];
    const content& last = array[array.size() - 1];

    return max(min(first, middle, cmp), min(max(first, middle, cmp), last, cmp),
               cmp);
}

/**\brief Returns pseudo-median according to nine values
 * \param array array of elements
 * \param key_func key function for comparing elements
 *
 * Chooses the median of the given array by median-of-nine method
 * according to Bentley and McIlroy "Engineering a Sort Function".
 */
template <typename content, typename Compare>
const content& median_of_nine(const span<content> array, Compare cmp) {
    using std::max;
    using std::min;

    const size_t n = array.size() - 1;
    const size_t step = (n / 8);

    const content& lower = median_of_three(array.slice(0, 2 * step), cmp);
    const content& middle =
        median_of_three(array.slice((n / 2) - step, (n / 2) + step), cmp);
    const content& upper = median_of_three(array.slice(n - 2 * step, n), cmp);

    return max(min(lower, middle, cmp),
               min(max(lower, middle, cmp), upper, cmp), cmp);
}

}
