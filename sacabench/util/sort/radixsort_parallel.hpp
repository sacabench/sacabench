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
    // Declaration of radix sort functions for sorting strings.
    // ----------------------------------------------------------------------------------------------------

    template <typename type>
    void radixsort_parallel(container<type> &input, container<type> &output, alphabet alphabet);

    template <typename type>
    void radixsort_parallel(container<type> &input, container<type> &output, alphabet alphabet, int current_position);

    // ----------------------------------------------------------------------------------------------------
    // Declaration of radix sort functions for sorting three digit integers (values between 100 and 999).
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(container<int> &input, container<int> &output);
    void radixsort_parallel(container<int> &input, container<int> &output, int current_position);
    void radixsort_parallel_verbose(container<int> &input, container<int> &output, int current_position);

    // ----------------------------------------------------------------------------------------------------
    // Declaration of radix sort functions which sort triple of type int (each value between 0 and 9).
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(std::vector<std::tuple<int, int, int>> &input, 
                            std::vector<std::tuple<int, int, int>> &output);
    
    void radixsort_parallel(std::vector<std::tuple<int, int, int>> &input, 
                            std::vector<std::tuple<int, int, int>> &output,
                            int current_position);

    // ----------------------------------------------------------------------------------------------------
    // Declaration of radix sort functions which sort triple of type char with usage of alphabet.
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(std::vector<std::tuple<char, char, char>> &input, 
                            std::vector<std::tuple<char, char, char>> &output,
                            alphabet &alphabet);
    
    void radixsort_parallel(std::vector<std::tuple<char, char, char>> &input, 
                            std::vector<std::tuple<char, char, char>> &output,
                            alphabet &alphabet,
                            int current_position);

    // ----------------------------------------------------------------------------------------------------
    // Declaration of (hopefully final) radix sort functions which sort triple
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(util::container<std::tuple<char, int, int>> &input, 
                            util::container<std::tuple<char, int, int>> &output,
                            alphabet &alphabet);
    
    void radixsort_parallel(util::container<std::tuple<char, int, int>> &input, 
                            util::container<std::tuple<char, int, int>> &output,
                            alphabet &alphabet,
                            int current_position);

    // ----------------------------------------------------------------------------------------------------
    // Implementation of radix sort functions for sorting strings.
    // ----------------------------------------------------------------------------------------------------

    template <typename type>
    void radixsort_parallel(container<type> &input, container<type> &output, alphabet alphabet) {
        radixsort_parallel(input, output, alphabet, input[0].size() - 1);
    }

    template <typename type>
    void radixsort_parallel(container<type> &input, container<type> &output, alphabet alphabet, int current_position) {

        // Check that all strings have the same size.
        for (size_t index = 0; index < input.size() - 1; index++) {
            DCHECK_EQ(input[index].size(), input[index + 1].size());
        }

        if (current_position < 0) { return; }

        std::cout << "Current Position: " << current_position << std::endl;

        std::cout << "Setting number of threads to 1" << std:: endl;
        omp_set_num_threads(1);
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();
        util::container<std::vector<string>> sorting_lists(num_threads * alphabet.size_without_sentinel());
        auto items_per_thread = (input.size() / num_threads) + 1; 

        std::cout << "Finished single threaded setup." << std::endl;
        std::cout << "Number of threads: " << num_threads << std::endl;
        std::cout << "Size of input: " << input.size() << std::endl;
        std::cout << "Size of alphabet: " << alphabet.size_without_sentinel() << std::endl;
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
                string current_string = input[index];
                character current_char = current_string[current_position];

                #pragma omp critical
                std::cout << "Current char: " << current_char << std::endl;

                auto effective_char_value = alphabet.effective_value(current_char) - 1;
                uint64_t current_index = thread_id * alphabet.size_without_sentinel() + effective_char_value;

                #pragma omp critical
                std::cout << "Current index: " << current_index << std::endl;

                sorting_lists[current_index].push_back(current_string);
            }
        }

        std::cout << "Sorting Lists: " << std::endl;

        for (std::vector bucket: sorting_lists) {
            for (string element: bucket) {
                std::cout << element << ", ";
            }
        }
        
        std::cout << std::endl;

        // sum up lists
        int current_insert_index = 0;
        // for each number
        for (size_t index = 0; index < alphabet.size_without_sentinel(); index++) {
            // in each thread
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                size_t current_index = thread_index * alphabet.size_without_sentinel() + index;
                std::vector<string> bucket = sorting_lists[current_index];
                for (string element : bucket) {
                    std::cout << "Inserting element: " << element << " into position: " << current_insert_index << std::endl;
                    output[current_insert_index] = element;
                    current_insert_index += 1;
                }
            } 
        }

        std::cout << "Single Lists: " << std::endl;
        for (string element: output) {
            std::cout << element << ", ";
        }
        std::cout << std::endl;

        radixsort_parallel(output, output, alphabet, current_position - 1);
    }

    // ----------------------------------------------------------------------------------------------------
    // Implementation of radix sort functions for sorting three digit integers (values between 100 and 999).
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(container<int> &input, container<int> &output) {
        radixsort_parallel(input, output, 2);
    }

    void radixsort_parallel(container<int> &input, container<int> &output, int current_position) {

        if (current_position < 0) { return; }
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();
        util::container<std::vector<int>> sorting_lists(num_threads * 10);
        auto items_per_thread = (input.size() / num_threads) + 1; 

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
                int current_digit = (current_number / static_cast<int>(pow(10, exponent))) % 10;
                sorting_lists[thread_id * 10 + current_digit].push_back(current_number);
            }
        }

        // sum up lists
        int current_insert_index = 0;
        // for each number
        for (size_t index = 0; index < 10; index++) {
            // in each thread
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * 10 + index;
                std::vector bucket = sorting_lists[current_index];
                for (int element : bucket) {
                    output[current_insert_index] = element;
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
        util::container<std::vector<int>> sorting_lists(num_threads * 10);
        auto items_per_thread = (input.size() / num_threads) + 1; 

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

                sorting_lists[thread_id * 10 + current_digit].push_back(current_number);
            }
        }

        std::cout << "Sorting Lists: " << std::endl;

        for (std::vector bucket: sorting_lists) {
            for (int element: bucket) {
                std::cout << element << ", ";
            }
        }
        
        std::cout << std::endl;

        // sum up lists
        int current_insert_index = 0;
        for (size_t index = 0; index < 10; index++) {
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * 10 + index;
                auto value = sorting_lists[current_index];
                std::vector bucket = sorting_lists[current_index];
                for (int element : bucket) {
                    std::cout << "Inserting element: " << element << " into position: " << current_insert_index << std::endl;
                    output[current_insert_index] = element;
                    current_insert_index += 1;
                }
            } 
        }

        std::cout << "Single Lists: " << std::endl;
        for (int element: output) {
            std::cout << element << ", ";
        }
        std::cout << std::endl;

        radixsort_parallel_verbose(output, output, current_position - 1);
    }

    // ----------------------------------------------------------------------------------------------------
    // Implementation of radix sort functions which sort triple of type int (each value between 0 and 9).
    // ----------------------------------------------------------------------------------------------------
    
    void radixsort_parallel(std::vector<std::tuple<int, int, int>> &input, 
                            std::vector<std::tuple<int, int, int>> &output) {
        radixsort_parallel(input, output, 2);
    }

    void radixsort_parallel(std::vector<std::tuple<int, int, int>> &input, 
                            std::vector<std::tuple<int, int, int>> &output,
                            int current_position) {

        if (current_position < 0) { return; }
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();

        // Tuple --> Elements to be sorted.
        // Vector --> Buckets for Elements with same value at current position.
        // Container --> List of Buckets.
        std::vector<std::vector<std::tuple<int, int, int>>> sorting_lists(num_threads * 10);

        auto items_per_thread = (input.size() / num_threads) + 1; 

        #pragma omp parallel
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t start_index = thread_id * items_per_thread;
            uint64_t end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (uint64_t index = start_index; index < end_index; index++) {
                std::tuple<int, int, int> current_triple = input[index];

                // Check current position and insert it into get function for triple,
                // because there is no way to pass variable value current_position into it.
                int current_value;
                if (current_position == 0) {
                    current_value = std::get<0>(current_triple);
                } else if (current_position == 1) {
                    current_value = std::get<1>(current_triple);
                } else {
                    current_value = std::get<2>(current_triple);
                }

                sorting_lists[thread_id * 10 + current_value].push_back(current_triple);
            }
        }

        // sum up lists
        int current_insert_index = 0;
        // for each possible value
        for (size_t index = 0; index < 10; index++) {
            // in each thread
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * 10 + index;
                std::vector bucket = sorting_lists[current_index];
                for (std::tuple<int, int, int> element : bucket) {
                    output[current_insert_index] = element;
                    current_insert_index += 1;
                }
            } 
        }

        radixsort_parallel(output, output, current_position - 1);
    }

    // ----------------------------------------------------------------------------------------------------
    // Implementation of radix sort functions which sort triple of type char with usage of alphabet.
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(std::vector<std::tuple<char, char, char>> &input, 
                            std::vector<std::tuple<char, char, char>> &output,
                            alphabet &alphabet) {
        radixsort_parallel(input, output, alphabet, 2);
    }

    void radixsort_parallel(std::vector<std::tuple<char, char, char>> &input, 
                            std::vector<std::tuple<char, char, char>> &output,
                            alphabet &alphabet,
                            int current_position) {

        if (current_position < 0) { return; }
    
        // Setup lists for all threads in one big array.
        const size_t num_threads = omp_get_max_threads();

        // Tuple --> Elements to be sorted.
        // Vector --> Buckets for Elements with same value at current position.
        // Container --> List of Buckets.
        std::vector<std::vector<std::tuple<char, char, char>>> sorting_lists(num_threads * alphabet.size_with_sentinel());

        auto items_per_thread = (input.size() / num_threads) + 1; 

        #pragma omp parallel
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t start_index = thread_id * items_per_thread;
            uint64_t end_index = start_index + items_per_thread;

            if (input.size() < end_index) {
                end_index = input.size();
            }

            for (uint64_t index = start_index; index < end_index; index++) {
                std::tuple<char, char, char> current_triple = input[index];

                // Check current position and insert it into get function for triple,
                // because there is no way to pass variable value current_position into it.
                char current_value;
                if (current_position == 0) {
                    current_value = std::get<0>(current_triple);
                } else if (current_position == 1) {
                    current_value = std::get<1>(current_triple);
                } else {
                    current_value = std::get<2>(current_triple);
                }

                int insert_position = thread_id * alphabet.size_with_sentinel() + alphabet.effective_value(current_value);
                sorting_lists[insert_position].push_back(current_triple);
            }
        }

        // sum up lists
        int current_insert_index = 0;
        // for each possible value
        for (size_t index = 0; index < alphabet.size_with_sentinel(); index++) {
            // in each thread
            for (size_t thread_index = 0; thread_index < num_threads; thread_index++) {
                auto current_index = thread_index * alphabet.size_with_sentinel() + index;
                std::vector bucket = sorting_lists[current_index];
                for (std::tuple<char, char, char> element : bucket) {
                    output[current_insert_index] = element;
                    current_insert_index += 1;
                }
            } 
        }

        radixsort_parallel(output, output, alphabet, current_position - 1);
    }

    // ----------------------------------------------------------------------------------------------------
    // Implementation of (hopefully final) radix sort functions which sort triple
    // ----------------------------------------------------------------------------------------------------

    void radixsort_parallel(util::container<std::tuple<char, int, int>> &input, 
                            util::container<std::tuple<char, int, int>> &output,
                            alphabet &alphabet) {
        // first sort position 0 (char) and than position 1 (isa)
        radixsort_parallel(input, output, alphabet, 0);
    }

    void radixsort_parallel(util::container<std::tuple<char, int, int>> &input, 
                            util::container<std::tuple<char, int, int>> &output,
                            alphabet &alphabet,
                            int current_position) {

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
                    output[current_insert_index] = element;

                    current_insert_index += 1;
                }
            } 
        }

        std::cout << "Single Lists: " << std::endl;
        for (auto element: output) {
            std::cout << "< " << std::get<0>(element) << ", " << std::get<1>(element) << ", " << std::get<2>(element) << " >," << std::endl;;
        }
        std::cout << std::endl;
        // TODO: Save position in last element of triple
        radixsort_parallel(output, output, alphabet, current_position + 1);
    }
}
