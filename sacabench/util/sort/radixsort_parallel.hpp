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

    // ----------------------------------------------------------------------------------------------------
    // Declaration of (hopefully final) radix sort functions which sort triple
    // ----------------------------------------------------------------------------------------------------

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            alphabet &alphabet);
    
    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            alphabet &alphabet,
                            size_t current_position);

    // ----------------------------------------------------------------------------------------------------
    // Implementation of (hopefully final) radix sort functions which sort triple
    // ----------------------------------------------------------------------------------------------------

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            alphabet &alphabet) {
        // first sort position 0 (char) and than position 1 (isa)
        radixsort_parallel(input, alphabet, 0);
    }

    template <typename C, typename sa_index>
    void radixsort_parallel(util::container<std::tuple<C, sa_index, sa_index>> &input, 
                            alphabet &alphabet,
                            size_t current_position) {

        if (current_position > 1) { return; }
        std::cout << "Current Position: " << current_position << std::endl;

        //std::cout << "Setting number of threads to 1" << std:: endl;
        //omp_set_num_threads(1);
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();

        size_t current_alphabet_size;
        if (current_position == 0) {
            current_alphabet_size = alphabet.size_with_sentinel();
        } else {
            current_alphabet_size = 10;
        }

        // Tuple --> Elements to be sorted.
        // inner Container --> Buckets for Elements with same value at current position.
        // outer Container --> List of Buckets.
        std::vector<std::vector<std::tuple<char, int, int>>> sorting_lists(num_threads * current_alphabet_size);

        auto items_per_thread = (input.size() / num_threads) + 1; 

        std::cout << "Finished single threaded setup." << std::endl;
        std::cout << "Number of threads: " << num_threads << std::endl;
        std::cout << "Size of input: " << input.size() << std::endl;
        std::cout << "Size of alphabet: " << current_alphabet_size << std::endl;
        std::cout << "Size of sorting lists: " << sorting_lists.size() << std::endl;
        std::cout << "Items per thread: " << items_per_thread << std::endl;

        #pragma omp parallel
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t start_index = thread_id * items_per_thread;
            uint64_t end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (uint64_t index = start_index; index < end_index; index++) {
                std::tuple<char, int, int> current_triple = input[index];

                if (current_position == 0) {
                    char current_value = std::get<0>(current_triple);

                    #pragma omp critical
                    std::cout << "Current value: " << current_value << std::endl;

                    int insert_position = thread_id * alphabet.size_with_sentinel() + alphabet.effective_value(current_value);
                    
                    #pragma omp critical
                    std::cout << "Current insert_position: " << insert_position << std::endl;

                    sorting_lists[insert_position].push_back(current_triple);

                    #pragma omp critical
                    std::cout << "Finished inserting" << std::endl;

                } else {
                    int current_value = std::get<1>(current_triple);

                    #pragma omp critical
                    std::cout << "Current value: " << current_value << std::endl;

                    int insert_position = thread_id * 10 + current_value;

                    #pragma omp critical
                    std::cout << "Current insert_position: " << insert_position << std::endl;

                    sorting_lists[insert_position].push_back(current_triple);

                    #pragma omp critical
                    std::cout << "Finished inserting" << std::endl;
                }
            }
        }

        std::cout << "Sorting Lists: " << std::endl;

        for (auto bucket: sorting_lists) {
            for (auto element: bucket) {
                std::cout << "< " << std::get<0>(element) << ", " << std::get<1>(element) << ", " << std::get<2>(element) << " >" << std::endl;
            }
        }
        
        std::cout << std::endl;

        // sum up lists
        int current_insert_index = 0;
        // for each possible value
        for (size_t index = 0; index < current_alphabet_size; index++) {
            // in each thread
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * current_alphabet_size + index;
                std::vector bucket = sorting_lists[current_index];
                for (std::tuple<char, int, int> element : bucket) {

                    std::cout << "Updating position flag in triple from " << std::get<2>(element) << " to " << current_insert_index << std::endl; 
                    // Update last value of tripel and save its position.
                    std::get<2>(element) = current_insert_index;
                    std::cout << "Position flag in triple is now " << std::get<2>(element) << std::endl; 

                    std::cout << "Inserting element: ";
                    std::cout << "< " << std::get<0>(element) << ", " << std::get<1>(element) << ", " << std::get<2>(element) << " >";
                    std::cout << " into position: " << current_insert_index << std::endl;
                    input[current_insert_index] = element;

                    current_insert_index += 1;
                }
            } 
        }

        std::cout << "Single Lists: " << std::endl;
        for (auto element: input) {
            std::cout << "< " << std::get<0>(element) << ", " << std::get<1>(element) << ", " << std::get<2>(element) << " >," << std::endl;;
        }
        std::cout << std::endl;
        radixsort_parallel(input, alphabet, current_position + 1);
    }
}
