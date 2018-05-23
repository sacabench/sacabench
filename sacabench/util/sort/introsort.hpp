/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <cmath>
#include "util/span.hpp"
#include "heapsort.hpp"
#include "insertionsort.hpp"
#include "ternary_quicksort.hpp"

/** \file introsort.hpp
 * \brief Implements the introspective sort sorting method. (see Introspective
 * Sorting and Selection Algorithms, David R. Musser)
 */

namespace sacabench::util::sort {

    // Change threshold to empirically relevant value (e.g. 8 or 15)
    // Value of 2 for testing
    const size_t SIZE_THRESHOLD = 2; /*< \brief Defines the threshold for using
    insertionsort instead of quicksort. */

    /**
     * \brief The internal loop of introsort.
     *
     * Executes the main logic of introsort. While the max. depth is not
     * reached, it partitions the span according to ternary quicksort, otherwise
     * heapsort is used. If, at any time, the size of the span to be sorted is
     * smaller than the threshold, then insertionsort is used.
     *
     * @tparam T The data type to be sorted. Needs to be comparable.
     * @tparam F The comparable object. std::less<T> by default.
     * @param data The span containing all elements to be sorted.
     * @param depth_limit The maximum number of iterations, after heapsort is
     * used.
     * @param compare_fun The comparison function of type F to be used for
     * comparing two elements in data.
     */
    template<typename T, typename F=std::less<T>>
    void introsort_internal(span<T> data, size_t depth_limit,
                            F compare_fun = F()) {
        // "break"-condition for using insertionsort (data is small enough)
        while(data.size() > SIZE_THRESHOLD) {
            // Use heapsort to finish sorting, end loop.
            if(depth_limit == 0) {
                heapsort(data, compare_fun);
                return;
            }
            else {
                --depth_limit;
                // Compute pivot element using median of three
                auto pivot = sort::ternary_quicksort::median_of_three
                        (data, compare_fun);

                // Partition data with ternary_quicksort::partition
                auto bounds = sort::ternary_quicksort::partition
                        (data, compare_fun, pivot);

                // Create partitions with partitioning bounds
                span<T> lesser = data.slice(0, bounds.first);
                span<T> greater = data.slice(bounds.second, data.size());

                // Recursive call only on interval with elements greater pivot
                introsort_internal(greater, depth_limit, compare_fun);

                // Overwrite data with lesser (while-loop continues with lesser)
                data = lesser;
            }
        }
    }

    /**
     * \brief Sorts the given span in increasing order according to the given
     * comparison function.
     *
     * Sorts the submitted data via Introspective Sort (see Introspective
     * Sorting and Selection Algorithms, David R. Musser). It partitions the
     * data to be sorted in the ternary quicksort fashion using median of 3 as
     * pivot element. After a depth of floor(lg(|data|)) heapsort is used
     * instead of quicksort. At any time when |data| < SIZE_THRESHOLD, the final
     * interval is sorted by insertionsort.
     *
     * @tparam T The data type to be sorted. Needs to be comparable.
     * @tparam F The used compare function. Uses std::less<T> by default.
     * @param[in,out] data The span containing all elements to be sorted.
     * @param[in] compare_fun The comparison function of type F to be used for
     * comparing two elements
     */
    template<typename T, typename F=std::less<T>>
    void introsort
            (span<T> data, F compare_fun = F()) {
        // Max. number of allowed iterations: 2*log(size)
        // due to empirically good results
        size_t max_iterations = 2 * floor(log(data.size()));
        introsort_internal(data, max_iterations, compare_fun);

        // Call insertion sort at the end - finally sorts the intervals divided by
        // introsort_internal (but does not sort into different intervals)
        insertion_sort(data, compare_fun);
    }
}