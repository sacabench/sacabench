/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/string.hpp>
#include <omp.h>
#include <math.h>

namespace sacabench::util::sort {

    void radixsort_parallel(container<int> &input, container<int> &output);
    void radixsort_parallel(container<int> &input, container<int> &output, int current_position);

    void radixsort_parallel(container<int> &input, container<int> &output) {
        radixsort_parallel(input, output, 2);
    }

    void radixsort_parallel(container<int> &input, container<int> &output, int current_position) {

        if (current_position < 0) { return; }
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();
        util::container<size_t> sorting_lists(num_threads * 10);
        auto items_per_thread = (input.size() / num_threads); 

        #pragma omp parallel
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t start_index = thread_id * items_per_thread;
            uint64_t end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (uint64_t index = start_index; index < end_index; index++) {
                auto current_number = input[index];
                int exponent = 2 - current_position;
                int current_digit =  (current_number / static_cast<int>(pow(10, exponent))) % 10;
                sorting_lists[thread_id * 10 + current_digit] = current_number;
            }
        }

        // sum up lists
        int current_insert_index = 0;
        for (size_t index = 0; index < 10; index++) {
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * 10 + index;
                auto value = sorting_lists[current_index];
                if (value != 0) {
                    output[current_insert_index] = value;
                    current_insert_index += 1;
                }
            } 
        }

        radixsort_parallel(output, output, current_position - 1);
    }

    void radixsort_parallel_verbose(container<int> &input, container<int> &output, int current_position) {

        if (current_position < 0) { return; }

        std::cout << "Current Position: " << current_position << std::endl;
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();
        util::container<size_t> sorting_lists(num_threads * 10);
        auto items_per_thread = (input.size() / num_threads); 

        #pragma omp parallel
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t start_index = thread_id * items_per_thread;
            uint64_t end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (uint64_t index = start_index; index < end_index; index++) {

                auto current_number = input[index];
                int exponent = 2 - current_position;
                int current_digit =  (current_number / static_cast<int>(pow(10, exponent))) % 10;

                #pragma omp critical
                std::cout << "Current digit: " << current_digit << std::endl;

                sorting_lists[thread_id * 10 + current_digit] = current_number;
            }
        }

        std::cout << "Sorting Lists: " << std::endl;
        for (int element: sorting_lists) {
            std::cout << element << ", ";
        }
        std::cout << std::endl;

        // sum up lists
        int current_insert_index = 0;
        for (size_t index = 0; index < 10; index++) {
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * 10 + index;
                auto value = sorting_lists[current_index];
                if (value != 0) {
                    output[current_insert_index] = value;
                    current_insert_index += 1;
                }
            } 
        }

        std::cout << "Single Lists: " << std::endl;
        for (int element: output) {
            std::cout << element << ", ";
        }
        std::cout << std::endl;

        radixsort_parallel(output, output, current_position - 1);
    }
}

