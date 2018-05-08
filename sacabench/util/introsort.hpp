/*******************************************************************************
 * sacabench/util/introsort.hpp
 *
 * Copyright (C) 2018 Oliver Magiera
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <cmath>
#include "span.hpp"
#include "heapsort.hpp"
#include "insertionsort.hpp"
#include "sort/ternary_quicksort.hpp"

//TODO: Add test for introsort
//TODO: Add quicksort to base
namespace sacabench::util {

    // Change threshold to empirically relevant value (e.g. 8 or 15)
    // Value of 2 for testing
    const size_t SIZE_THRESHOLD = 2;

    /*
     * Sorts the given span in increasing order according to the given
     * comparison function
     *
     * data: The span containing all elements to be sorted.
     * compare_fun: The comparison function to be used.
     */
    template<typename T, typename F=std::less<int>>
    void introsort
            (span<T> data, F compare_fun = F()) {
        // Max. number of allowed iterations: 2*log(size)
        // due to empirically good results (see Introspective Sorting and
        // Selection Algorithms, David R. Musser)
        size_t max_iterations = 2 * floor(log(last_index - first_index));
        introsort_internal(data, max_iterations, compare_fun);

        // Call insertion sort at the end - finally sorts the intervals divided by
        // introsort_internal (but does not sort into different intervals)
        insertion_sort(data, compare_fun);
    }

    template<typename T, typename F=std::less<int>>
    void introsort_internal(span<T> data, size_t depth_limit,
                            F compare_fun = F()) {
        while(data.size() > SIZE_THRESHOLD) {
            // Use heapsort to finish sorting, end loop.
            if(depth_limit == 0) {
                heapsort(data, compare_fun);
                return;
            }
            else {
                --depth_limit;
                // Compute pivot element using median of three
                size_t pivot = sort::ternary_quicksort::median_of_three
                         (data, compare_fun);

                // Partition data with ternary_quicksort::partition
                auto bounds = sort::ternary_quicksort::partition
                         (data, compare_fun, pivot);

                // Create partition with partitioning bounds
                span<T> lesser = data.slice(0, bounds[0]);
                span<T> greater = data.slice(bounds[1], data.size());

                // Recursive call only on interval with elements greater pivot
                introsort_internal(greater, depth_limit, compare_fun);

                // Overwrite data with lesser (while loop continues with lesser)
                data = lesser;
            }
        }
    }
}