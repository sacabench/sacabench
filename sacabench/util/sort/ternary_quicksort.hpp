/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/compare.hpp>
#include <util/span.hpp>

#include "common.hpp"

namespace sacabench::util::sort::ternary_quicksort {

/**\brief Swaps elements so that the given array is a correct ternary
 *        partition according to pivot
 *\param array array of elements
 *\param cmp comparing function
 *\param pivot_element pivot element for partitioning
 *
 *\return tuple (i,j), which corresponds to the equal partition
 *
 *Swaps the elements until the array is a ternary partition.
 *
 */
template <typename content, typename Compare>
std::pair<size_t, size_t> partition(span<content> array, Compare cmp,
                                    const content& pivot_element) {
    const auto less = cmp;
    const auto equal = util::as_equal(cmp);
    const auto greater = util::as_greater(cmp);

    // Init values, which keep track of the partition position
    size_t left = 0;
    size_t mid = 0;
    size_t right = 0;
    for (size_t i = 0; i < array.size(); ++i) {
        // Count Elements in less-Partition
        if (less(array[i], pivot_element)) {
            ++mid;
        }
        // Count Elements in equal partition
        else if (equal(array[i], pivot_element)) {
            ++right;
        }
    }

    // Add #elements smaller than pivot to get correct start position
    // for greater-partition counter
    right = right + mid;

    // Save these values, because we need to return them afterwards
    size_t i = mid;
    size_t j = right;

    // Loop, which builds the less-partition
    while (left < i) {
        DCHECK_LT(left, array.size());

        // If current element is the pivot_element, swap it into equal-partition
        if (equal(array[left], pivot_element)) {
            DCHECK_LT(left, array.size());
            DCHECK_LT(mid, array.size());
            std::swap(array[left], array[mid]);
            ++mid;
        }
        // else if the element belongs in the greater-partition, swap it there
        else if (greater(array[left], pivot_element)) {
            DCHECK_LT(left, array.size());
            DCHECK_LT(right, array.size());
            std::swap(array[left], array[right]);
            ++right;
        }
        // else, the current element is already at the right place
        else {
            ++left;
        }
    }

    // Loop, which builds the equal partition
    while (mid < j) {
        // if current element is bigger than the pivot_element, swap it
        if (greater(array[mid], pivot_element)) {
            DCHECK_LT(right, array.size());
            DCHECK_LT(mid, array.size());
            std::swap(array[mid], array[right]);
            ++right;
        }
        // else, the element is ar the right place
        else {
            ++mid;
        }
        // we dont need to consider less elements, because they are already in
        // the right part of the array
    }
    // less- and equal-partitions are built -> greater-partition is built
    // implicitly

    // return the bounds
    return std::make_pair(i, j);
}

/**\brief Sorts an array with the given comparing function
 *
 *\param array array of elements
 *\param cmp comparing function
 *
 *Sorts an given array (according to the first symbol in case of strings).
 *The comparing function should return for given a,b a value <0 when a < b,
 *vice versa.
 */
template <typename content, typename Compare>
void ternary_quicksort(span<content> array, Compare cmp) {
    size_t n = array.size();

    // recursion termination
    if (n <= 1) {
        return;
    }

    if (n == 2) {
        if (cmp(array[1], array[0])) {
            std::swap(array[0], array[1]);
        }
        return;
    }

    // Choose pivot according to array size
    const content pivot = (n > MEDIAN_OF_NINE_THRESHOLD)
                              ? median_of_nine(array, cmp)
                              : median_of_three(array, cmp);
    auto result = partition(array, cmp, pivot);
    // sorts greater and less partiotion recursivly
    ternary_quicksort(array.slice(0, result.first), cmp);
    ternary_quicksort(array.slice(result.second), cmp);
    return;
}
} // namespace sacabench::util::sort::ternary_quicksort
