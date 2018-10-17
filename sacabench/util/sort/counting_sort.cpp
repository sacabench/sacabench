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
#include <CLI/CLI.hpp>

#include "util/container.hpp"
#include "util/read_text.hpp"

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

template<typename alphabet_size_type>
uint64_t getHighestNumber(util::container<alphabet_size_type> const& data) {
    uint64_t highestNumber = 0;
    for (uint64_t element: data) {
        if (element > highestNumber) {
            highestNumber = element;
        }
    }
    return highestNumber;
}

template<typename alphabet_size_type>
void counting_sort(util::container<alphabet_size_type> const& data,
                   util::container<alphabet_size_type>& result) {
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

template<typename alphabet_size_type>
void counting_sort_parallel(util::container<alphabet_size_type> const& data,
                            util::container<alphabet_size_type>& result) {
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

template<typename alphabet_size_type>
void counting_sort_parallel2(util::container<alphabet_size_type> const& data,
                             util::container<alphabet_size_type>& result) {
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

template<typename alphabet_size_type>
void counting_sort_parallel_flo(util::container<alphabet_size_type> const& data,
                             util::container<alphabet_size_type>& result) {

    uint64_t alphabet_size = getHighestNumber(data) + 1;
    util::container<uint64_t> global_sorting_list(alphabet_size);
    DCHECK_EQ(global_sorting_list.size(), alphabet_size);

    const uint64_t num_threads = omp_get_max_threads();

    struct Range { uint64_t start; uint64_t end; };
    auto get_local_range = [](size_t threads, size_t rank, size_t size) {
      const uint64_t offset =
          (rank * (size / threads)) + std::min<uint64_t>(rank, size % threads);
      const uint64_t local_size =
          (size / threads) + ((rank < size % threads) ? 1 : 0);

      return Range { offset, offset + local_size };
    };
    auto get_local_slice = [&](size_t threads, size_t rank, auto& slice) {
      auto range = get_local_range(threads, rank, slice.size());
      return slice.slice(range.start, range.end);
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
    for (uint64_t thread_id = 0; thread_id < num_threads - 1; thread_id++) {
        for (uint64_t index = 0; index < global_sorting_list.size(); index++) {
            sorting_lists[(thread_id + 1) * alphabet_size + index] +=
                sorting_lists[thread_id * alphabet_size + index];
        }
    }
    global_sorting_list[0] = sorting_lists[(num_threads - 1) * alphabet_size];
    for (uint64_t index = 1; index < global_sorting_list.size(); index++) {
        global_sorting_list[index] = sorting_lists[(num_threads - 1) * alphabet_size + index] + global_sorting_list[index - 1];
    }

#pragma omp parallel
    {
        // add elements sorted into result
        const uint64_t thread_id = omp_get_thread_num();
        auto local_range = get_local_range(num_threads, thread_id, data.size());
        const uint64_t start_index = local_range.start;
        uint64_t end_index = local_range.end;
        if (data.size() < end_index) {
            end_index = data.size();
        }
        uint64_t insert_index = start_index;

        for (uint64_t insert_element = 0; insert_element < alphabet_size; ++insert_element) {
            if (global_sorting_list[insert_element] <= start_index) {
                continue;
            }
            while (insert_index < global_sorting_list[insert_element] && insert_index < end_index) {
                result[insert_index] = insert_element;
                ++insert_index;
            }
            if (insert_index >= end_index) {
                break;
            }
        }
    }
}

template<typename alphabet_size_type>
bool isSorted(util::container<alphabet_size_type> const& vector) {
    return std::is_sorted(vector.begin(), vector.end());
}

template<typename alphabet_size_type>
void run(util::span<alphabet_size_type const> data) {
    util::container<alphabet_size_type> correctly_sorted = data;
    std::cout << "Running std::sort" << std::endl;
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
        } else if (duration < std::chrono::seconds(10ull)) {
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
        util::container<alphabet_size_type> result(data.size());

        std::cout << "Running " << name << std::endl;

        auto start_timer = std::chrono::high_resolution_clock::now();
        func(data, result);
        auto end_timer = std::chrono::high_resolution_clock::now();
        bool is_correct = (result == correctly_sorted);
        auto is_correct_s = is_correct ? "yes" : "NO";
        std::cout << "Result is sorted after " << name <<  ": " << is_correct_s << std::endl;

        auto duration = end_timer - start_timer;
        q(duration, name);
        std::cout << std::endl;
    };

    r(counting_sort<alphabet_size_type>, "non-parallel counting sort");
    //r(counting_sort_parallel, "parallel counting sort");
    //r(counting_sort_parallel2, "parallel counting sort 2");
    r(counting_sort_parallel_flo<alphabet_size_type>, "parallel counting sort flo");
}

std::int32_t main(std::int32_t argc, char const** argv) {
    CLI::App app{"CLI for SACABench."};
    app.failure_message(CLI::FailureMessage::help);

    std::string input_filename = "";
    app.add_option("--input", input_filename);

    CLI11_PARSE(app, argc, argv);

    std::cout << "Test auf Parallelität: " << std::endl;
#pragma omp parallel for
    for(int number = 0; number < 10; ++number) {
        printf(" %d", number);
    }
    printf(".\n");

    auto threads = omp_get_max_threads();
    std::cout << "Number threads used by OMP: " << threads << std::endl;

    if (input_filename.size() > 0) {
        std::cout << "Read input file" << std::endl;
        auto in = util::read_text_context(input_filename);
        util::container<uint8_t> data(in.size);
        in.read_text(data);
        run<uint8_t>(data);
    } else {
        std::cout << "Generating data" << std::endl;
        util::container<uint64_t> data = generate_data(4'000'000ull);
        run<uint64_t>(data);
    }


    return 0;
}

