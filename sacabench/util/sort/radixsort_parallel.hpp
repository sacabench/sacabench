/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/string.hpp>
#include <util/alphabet.hpp>
#include <tuple>
#include <omp.h>
#include <math.h>

namespace sacabench::util::sort {

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            size_t character_count);
    
    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            size_t character_count,
                            size_t current_position);

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            size_t character_count) {
        radixsort_parallel(input, character_count, 1);
    }

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            size_t character_count,
                            size_t current_position) {
    
        // Setup lists for all threads in one big array.
        sa_index num_threads = omp_get_max_threads();       

        // Tuple --> Elements to be sorted.
        // inner Container --> Buckets for Elements with same value at current position.
        // outer Container --> List of Buckets.
        std::vector<std::vector<std::tuple<C, sa_index, sa_index>>> sorting_lists(num_threads * character_count);

        sa_index items_per_thread = (input.size() / num_threads) + 1; 

        #pragma omp parallel
        {
            sa_index thread_id = omp_get_thread_num();
            sa_index start_index = thread_id * items_per_thread;
            sa_index end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (sa_index index = start_index; index < end_index; index++) {
                std::tuple<C, sa_index, sa_index> current_triple = input[index];
                
                if (current_position == 0) {
                    C current_value = +std::get<0>(current_triple);
                    sa_index insert_position = thread_id * character_count + current_value;
                    sorting_lists[insert_position].push_back(current_triple);
                } else {
                    sa_index current_value = std::get<1>(current_triple);
                    sa_index insert_position = thread_id * character_count + current_value;
                    sorting_lists[insert_position].push_back(current_triple);
                }
            }
        }

        // sum up lists
        sa_index current_insert_index = 0;
        // for each possible value
        for (sa_index index = 0; index < character_count; index++) {
            // in each thread
            for (sa_index thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * character_count + index;
                std::vector bucket = sorting_lists[current_index];
                for (std::tuple<C, sa_index, sa_index> element : bucket) {
                    input[current_insert_index] = element;
                    current_insert_index += 1;
                }
            } 
        }

        if (current_position == 0) { 
            return; 
        } else { 
            radixsort_parallel(input, character_count, current_position - 1);
        }
    }
}
