/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <unordered_map>
#include <util/container.hpp>
#include <vector>

namespace sacabench::util {

// workaround, because you can only access tuples by constant expressions
template <typename T>
size_t get_tuple_entry_septuple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    case 2:
        return size_t(std::get<2>(tuple));
    case 3:
        return size_t(std::get<3>(tuple));
    case 4:
        return size_t(std::get<4>(tuple));
    case 5:
        return size_t(std::get<5>(tuple));
    case 6:
        return size_t(std::get<6>(tuple));
    }
    return -1;
}
template <typename T>
size_t get_tuple_entry_triple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    case 2:
        return size_t(std::get<2>(tuple));
    }
    return -1;
}
template <typename T>
size_t get_tuple_entry_tuple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    }
    return -1;
}

template <typename T>
void radixsort_septuple(container<T>& tuples, container<size_t>& result) {

    std::unordered_map<size_t, std::vector<size_t>> buckets;
    std::unordered_map<size_t, std::vector<size_t>> other_buckets;
    size_t number_of_tuples = tuples.size();

    // for iterating, iterator has not correct order
    size_t biggest = 0;
    size_t other_biggest = 0;

    // bucketing according to last number
    for (size_t index = 0; index < number_of_tuples; ++index) {
        auto entry = get_tuple_entry_septuple(tuples[index], 6);
        buckets[entry].push_back(index);
        biggest = (biggest < entry) ? entry : biggest;
    }
    size_t iterations = 1;

    while (iterations != 7) {
        if (iterations % 2 != 0) {
            other_buckets.clear();
            for (size_t i = 0; i <= biggest; i++) {
                for (auto s : buckets[i]) {
                    auto entry =
                        get_tuple_entry_septuple(tuples[s], 6 - iterations);
                    other_buckets[entry].push_back(s);
                    other_biggest =
                        (other_biggest < entry) ? entry : other_biggest;
                }
            }
        } else {
            buckets.clear();
            for (size_t i = 0; i <= other_biggest; i++) {
                for (auto s : other_buckets[i]) {
                    auto entry =
                        get_tuple_entry_septuple(tuples[s], 6 - iterations);
                    buckets[entry].push_back(s);
                    biggest = (biggest < entry) ? entry : biggest;
                }
            }
        }
        ++iterations;
    }
    // collect and save to result
    size_t counter = 0;
    for (size_t i = 0; i <= biggest; i++) {
        for (auto s : buckets[i]) {
            result[counter] = s;
            ++counter;
        }
    }
}

template <typename T>
void radixsort_triple(container<T>& tuples, container<size_t>& result) {

    std::unordered_map<size_t, std::vector<size_t>> buckets;
    std::unordered_map<size_t, std::vector<size_t>> other_buckets;
    size_t number_of_tuples = tuples.size();

    // for iterating, iterator has not correct order
    size_t biggest = 0;
    size_t other_biggest = 0;

    // bucketing according to last number
    for (size_t index = 0; index < number_of_tuples; ++index) {
        auto entry = get_tuple_entry_triple(tuples[index], 2);
        buckets[entry].push_back(index);
        biggest = (biggest < entry) ? entry : biggest;
    }
    size_t iterations = 1;

    // simplify
    while (iterations != 3) {
        if (iterations % 2 != 0) {
            other_buckets.clear();
            for (size_t i = 0; i <= biggest; i++) {
                for (auto s : buckets[i]) {
                    auto entry =
                        get_tuple_entry_triple(tuples[s], 2 - iterations);
                    other_buckets[entry].push_back(s);
                    other_biggest =
                        (other_biggest < entry) ? entry : other_biggest;
                }
            }
        } else {
            buckets.clear();
            for (size_t i = 0; i <= other_biggest; i++) {
                for (auto s : other_buckets[i]) {
                    auto entry =
                        get_tuple_entry_triple(tuples[s], 2 - iterations);
                    buckets[entry].push_back(s);
                    biggest = (biggest < entry) ? entry : biggest;
                }
            }
        }
        ++iterations;
    }
    // collect and save to result
    size_t counter = 0;
    for (size_t i = 0; i <= biggest; i++) {
        for (auto s : buckets[i]) {
            result[counter] = s;
            ++counter;
        }
    }
}
template <typename T>
void radixsort_tuple(container<T>& tuples, container<size_t>& result) {

    std::unordered_map<size_t, std::vector<size_t>> buckets;
    std::unordered_map<size_t, std::vector<size_t>> other_buckets;
    size_t number_of_tuples = tuples.size();

    // for iterating, iterator has not correct order
    size_t biggest = 0;
    size_t other_biggest = 0;

    // bucketing according to last number
    for (size_t index = 0; index < number_of_tuples; ++index) {
        auto entry = get_tuple_entry_tuple(tuples[index], 1);
        buckets[entry].push_back(index);
        biggest = (biggest < entry) ? entry : biggest;
    }
    for (size_t i = 0; i <= biggest; i++) {
        for (auto s : buckets[i]) {
            auto entry = get_tuple_entry_tuple(tuples[s], 0);
            other_buckets[entry].push_back(s);
            other_biggest = (other_biggest < entry) ? entry : other_biggest;
        }
    }
    // collect and save to result
    size_t counter = 0;
    for (size_t i = 0; i <= other_biggest; i++) {
        for (auto s : other_buckets[i]) {
            result[counter] = s;
            ++counter;
        }
    }
}
} // namespace sacabench::util
