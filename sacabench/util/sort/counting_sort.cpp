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

// Nutzung von OpenMP auf macOS:
// 1. brew install libomp
// 2. clang++ -Xpreprocessor -fopenmp -std=c++1z main.cpp -o main -lomp
// 3. ./main

// https://stackoverflow.com/a/32887614
static std::vector<int> generate_data(size_t size) {
    using value_type = int;
    static std::uniform_int_distribution<value_type> distribution(1, 1000);
    static std::default_random_engine generator;

    std::vector<value_type> data(size);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}

int getHighestNumber(std::vector<int>& data) {
    int highestNumber = 0;
    for (int element: data) {
        if (element > highestNumber) {
            highestNumber = element;
        }
    }
    return highestNumber;
}

void counting_sort(std::vector<int>& data, std::vector<int>& result) {

    int highestNumber = getHighestNumber(data);
    std::vector<int> sortingList(highestNumber + 1);

    // count occurence of all elements in data
    for (int element: data) {
        sortingList[element] += 1;
    }

    // cumulate all entries of sortingList
    for (int index = 1; index <= sortingList.size(); index++) {
        sortingList[index] += sortingList[index - 1];
    }

    // add elements sorted into result
    for (int element: data) {
        int count = sortingList[element] - 1;
        sortingList[element] -= 1;
        result[count] = element;
    }
}

void counting_sort_parallel(std::vector<int>& data, std::vector<int>& result) {

    int highestNumber = getHighestNumber(data);
    std::vector<int> sortingList(highestNumber + 1);

    // count occurence of all elements in data
#pragma omp parallel for
    for (int index = 0; index < data.size(); index++) {
        //for (int element: data) {
        auto element = data[index];
        sortingList[element] += 1;
    }

    // cumulate all entries of sortingList
#pragma omp parallel for
    for (int index = 1; index <= sortingList.size(); index++) {
        sortingList[index] += sortingList[index - 1];
    }

    // add elements sorted into result
#pragma omp parallel for
    for (int index = 0; index < data.size(); index++) {
        //for (int element: data) {
        auto element = data[index];
        int count = sortingList[element] - 1;
        sortingList[element] -= 1;
        result[count] = element;
    }
}

bool isSorted(std::vector<int>& vector) {
    for (int index = 0; index < vector.size(); index++) {
        if (vector[index] > vector[index]) {
            return false;
        }
    }
    return true;
}

int main(int argc, const char * argv[]) {

    std::cout << "Test auf Parallelität: " << std::endl;
#pragma omp parallel for
    for(int number = 0; number < 10; ++number) {
        printf(" %d", number);
    }
    printf(".\n");


    std::vector<int> data = generate_data(10000);
    std::vector<int> result_non_parallel(data.size());
    std::vector<int> result_parallel(data.size());

    auto non_parallel_start_timer = std::chrono::high_resolution_clock::now();
    counting_sort(data, result_non_parallel);
    auto non_parallel_end_timer = std::chrono::high_resolution_clock::now();
    std::cout << "Result is sorted after non parallel sort: " << isSorted(result_non_parallel) << std::endl;

    auto parallel_start_timer = std::chrono::high_resolution_clock::now();
    counting_sort_parallel(data, result_parallel);
    auto parallel_end_timer = std::chrono::high_resolution_clock::now();
    std::cout << "Result is sorted after parallel sort: " << isSorted(result_parallel) << std::endl;

    auto non_parallel_duration = non_parallel_end_timer - non_parallel_start_timer;
    auto parallel_duration = parallel_end_timer - parallel_start_timer;

    if (non_parallel_duration < std::chrono::milliseconds(1000)) {

        std::cout << "Calculation with non parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(non_parallel_duration).count()
                  << " microseconds" << std::endl;

        std::cout << "Calculation with parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(parallel_duration).count()
                  << " microseconds" << std::endl;

    } else if (non_parallel_duration < std::chrono::milliseconds(1000000)) {

        std::cout << "Calculation with non parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(non_parallel_duration).count()
                  << " milliseconds" << std::endl;

        std::cout << "Calculation with parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(parallel_duration).count()
                  << " milliseconds" << std::endl;

    } else {
        std::cout << "Calculation with non parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::seconds>(non_parallel_duration).count()
                  << " seconds" << std::endl;

        std::cout << "Calculation with parallel counting sort took "
                  << std::chrono::duration_cast<std::chrono::seconds>(parallel_duration).count()
                  << " seconds" << std::endl;
    }

    return 0;
}

