/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <omp.h>

#include "util/container.hpp"

using namespace sacabench;

// Nutzung von OpenMP auf macOS:
// 1. brew install libomp
// 2. clang++ -Xpreprocessor -fopenmp -std=c++1z main.cpp -o main -lomp
// 3. ./main

// https://stackoverflow.com/a/32887614
util::container<uint64_t> generate_data(size_t size) {
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0, 9);

    util::container<uint64_t> data(size);
    std::generate(data.begin(), data.end(), [&] { return distribution(generator); });
    return data;
}

uint64_t getHighestNumber(util::container<uint64_t> const& data) {
    uint64_t highestNumber = 0;
    for (uint64_t element: data) {
        if (element > highestNumber) {
            highestNumber = element;
        }
    }
    return highestNumber;
}

void counting_sort(util::container<uint64_t> const& data,
                   util::container<uint64_t>& result) {
    uint64_t alphabet_size = getHighestNumber(data) + 1;
    util::container<uint64_t> sortingList(alphabet_size);
    DCHECK_EQ(sortingList.size(), alphabet_size);

    // count occurence of all elements in data
    for (uint64_t element: data) {
        sortingList[element] += 1;
    }

    // cumulate all entries of sortingList
    for (uint64_t index = 1; index < sortingList.size(); index++) {
        sortingList[index] += sortingList[index - 1];
    }

    // add elements sorted into result
    for (uint64_t element: data) {
        uint64_t count = sortingList[element] - 1;
        sortingList[element] -= 1;
        result[count] = element;
    }
}

void counting_sort_parallel(util::container<uint64_t> const& data,
                            util::container<uint64_t>& result) {
    uint64_t alphabet_size = getHighestNumber(data) + 1;
    util::container<uint64_t> sortingList(alphabet_size);
    DCHECK_EQ(sortingList.size(), alphabet_size);

    // count occurrence of all elements in data
#pragma omp parallel for
    for (uint64_t index = 0; index < data.size(); index++) {
        //for (uint64_t element: data) {
        auto element = data[index];
        sortingList[element] += 1;
    }

    // cumulate all entries of sortingList
#pragma omp parallel for
    for (uint64_t index = 1; index < sortingList.size(); index++) {
        sortingList[index] += sortingList[index - 1];
    }

    // add elements sorted into result
#pragma omp parallel for
    for (uint64_t index = 0; index < data.size(); index++) {
        //for (uint64_t element: data) {
        auto element = data[index];
        uint64_t count = sortingList[element] - 1;
        sortingList[element] -= 1;
        result[count] = element;
    }
}

void counting_sort_parallel2(util::container<uint64_t> const& data,
                             util::container<uint64_t>& result) {
    uint64_t alphabet_size = getHighestNumber(data) + 1;
    util::container<uint64_t> sortingList(alphabet_size);
    DCHECK_EQ(sortingList.size(), alphabet_size);

    #pragma omp parallel
    {
        //const auto omp_rank = omp_get_thread_num();
        //const auto omp_size = omp_get_num_threads();

        // count occurrence of all elements in data
        #pragma omp for
        for (uint64_t index = 0; index < data.size(); index++) {
            //for (uint64_t element: data) {
            auto element = data[index];
            sortingList[element] += 1;
        }

        // cumulate all entries of sortingList
        #pragma omp for
        for (uint64_t index = 1; index < sortingList.size(); index++) {
            sortingList[index] += sortingList[index - 1];
        }

        // add elements sorted into result
        #pragma omp for
        for (uint64_t index = 0; index < data.size(); index++) {
            //for (uint64_t element: data) {
            auto element = data[index];
            uint64_t count = sortingList[element] - 1;
            sortingList[element] -= 1;
            result[count] = element;
        }
    }
}

void counting_sort_parallel_flo(util::container<uint64_t> const& data,
                             util::container<uint64_t>& result) {

    uint64_t alphabet_size = getHighestNumber(data) + 1;
    util::container<uint64_t> sorting_list(alphabet_size);
    DCHECK_EQ(sorting_list.size(), alphabet_size);

    const uint64_t num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    struct Range { uint64_t start; uint64_t end; };
    auto get_local_range = [](size_t threads, size_t rank, auto& slice) {
      const auto size = slice.size();
      const uint64_t offset =
          (rank * (size / threads)) + std::min<uint64_t>(rank, size % threads);
      const uint64_t local_size =
          (size / threads) + ((rank < size % threads) ? 1 : 0);

      return Range { offset, offset + local_size };
    };
    auto get_local_slice = [](size_t threads, size_t rank, auto& slice) {
      const auto size = slice.size();
      const uint64_t offset =
          (rank * (size / threads)) + std::min<uint64_t>(rank, size % threads);
      const uint64_t local_size =
          (size / threads) + ((rank < size % threads) ? 1 : 0);

      return slice.slice(offset, offset + local_size);
    };

    util::container<uint64_t> sorting_lists(num_threads * alphabet_size);

#pragma omp parallel
    // count occurrences of all elements in data
    {
        const uint64_t thread_id = omp_get_thread_num();

        auto local_data = get_local_slice(num_threads, thread_id, data);
        auto local_sorting_list = get_local_slice(num_threads, thread_id, sorting_lists);
        DCHECK_EQ(local_sorting_list.size(), alphabet_size);

        for (auto element : local_data) {
            local_sorting_list[element]++;
        }
    }

    // sum up lists
    for (uint64_t index = 0; index < sorting_list.size(); index++) {
        for (uint64_t thread_id = 0; thread_id < num_threads - 1; thread_id++) {
            sorting_lists[(thread_id + 1) * alphabet_size + index] +=
                sorting_lists[thread_id * alphabet_size + index];
        }
    }
    for (uint64_t index = 1; index < sorting_list.size(); index++) {
        sorting_list[index] = sorting_lists[(num_threads - 1) * alphabet_size + index - 1] +
            sorting_list[index - 1];
    }

    // add offsets
    for (uint64_t index = 1; index < sorting_list.size(); index++) {
        for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            sorting_lists[thread_id * alphabet_size + index] += sorting_list[index];
        }
    }

#pragma omp parallel
    {
        // add elements sorted into result
        const uint64_t thread_id = omp_get_thread_num();
        const uint64_t start_index = thread_id * items_per_thread;
        uint64_t end_index = start_index + items_per_thread;
        if (data.size() < end_index) {
            end_index = data.size();
        }

        for (uint64_t index = start_index; index < end_index; index++) {
            auto element = data[index];
            uint64_t count = --sorting_lists[thread_id * alphabet_size + element];
            result[count] = element;
        }
    }
}

bool isSorted(util::container<uint64_t> const& vector) {
    return std::is_sorted(vector.begin(), vector.end());
}

int main() {

    std::cout << "Test auf Parallelität: " << std::endl;
#pragma omp parallel for
    for(int number = 0; number < 10; ++number) {
        printf(" %d", number);
    }
    printf(".\n");


    util::container<uint64_t> data = generate_data(1'000'000ull);

    util::container<uint64_t> correctly_sorted = data;
    std::sort(correctly_sorted.begin(), correctly_sorted.end());
    DCHECK(isSorted(correctly_sorted));

    auto q = [] (auto duration, auto name) {
        auto p = [&](auto count, auto unit) {
            std::cout << "Calculation with "<< name <<" took "
                    << count
                    << " " << unit << std::endl;
        };
        if (duration < std::chrono::milliseconds(1ull)) {
            p(
                std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
                "microseconds"
            );
        } else if (duration < std::chrono::seconds(1ull)) {
            p(
                std::chrono::duration_cast<std::chrono::milliseconds>(duration).count(),
                "milliseconds"
            );
        } else {
            p(
                std::chrono::duration_cast<std::chrono::seconds>(duration).count(),
                "seconds"
            );
        }
    };

    auto r = [&data, &correctly_sorted, &q](auto func, auto name) {
        util::container<uint64_t> result(data.size());

        std::cout << "Running " << name << std::endl;

        auto start_timer = std::chrono::high_resolution_clock::now();
        func(data, result);
        auto end_timer = std::chrono::high_resolution_clock::now();
        bool is_correct = (result == correctly_sorted);
        std::cout << "Result is sorted after " << name <<  ": " << is_correct << std::endl;

        auto duration = end_timer - start_timer;
        q(duration, name);
        std::cout << std::endl;
    };

    r(counting_sort, "non-parallel counting sort");
    //r(counting_sort_parallel, "parallel counting sort");
    //r(counting_sort_parallel2, "parallel counting sort 2");
    r(counting_sort_parallel_flo, "parallel counting sort flo");

    return 0;
}

