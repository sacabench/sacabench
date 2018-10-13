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
    std::uniform_int_distribution<uint64_t> distribution(1, 1000);

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

    // count occurence of all elements in data
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

    util::container<uint64_t> result_non_parallel(data.size());
    util::container<uint64_t> result_parallel(data.size());

    auto non_parallel_start_timer = std::chrono::high_resolution_clock::now();
    counting_sort(data, result_non_parallel);
    auto non_parallel_end_timer = std::chrono::high_resolution_clock::now();
    bool non_parellel_is_correct = (result_non_parallel == correctly_sorted);
    std::cout << "Result is sorted after non parallel sort: " << non_parellel_is_correct << std::endl;

    auto parallel_start_timer = std::chrono::high_resolution_clock::now();
    counting_sort_parallel(data, result_parallel);
    auto parallel_end_timer = std::chrono::high_resolution_clock::now();
    bool parellel_is_correct = (result_non_parallel == correctly_sorted);
    std::cout << "Result is sorted after parallel sort: " << parellel_is_correct << std::endl;

    auto non_parallel_duration = non_parallel_end_timer - non_parallel_start_timer;
    auto parallel_duration = parallel_end_timer - parallel_start_timer;

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

    q(non_parallel_duration, "non-parallel counting sort");
    q(parallel_duration, "parallel counting sort");

    return 0;
}

