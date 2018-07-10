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

template <typename T, typename S, typename N>
void radixsort_triple(T& text, S& tuples, S& result, const N& alphabet_size, const N position) {
    DCHECK_MSG(tuples.size() == result.size(),
               "tuples must have the same length as result");
               
    auto buckets = util::container<size_t>(alphabet_size);

    for(size_t i = 0; i < tuples.size(); ++i){
        ++buckets[text[tuples[i]+position]];
    }
    
    size_t sum = 0;
    for(size_t i = 0; i < buckets.size(); ++i){
        sum += buckets[i];
        buckets[i] = sum-buckets[i];
    }
        
    for(size_t  i = 0; i < tuples.size(); ++i){
        result[buckets[text[tuples[i]+position]]++] = tuples[i];
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

template <typename S, typename Key>
void radixsort_with_key(S& input, S& result, const size_t radix_size, 
        const size_t max_index, Key key_function) {
    DCHECK_MSG(input.size() == result.size(),
               "input must have the same length as result");
               
    auto buckets = util::container<size_t>(radix_size);
    for (size_t p = max_index; p >= 0; --p) {
        // reset buckets
        for (size_t i = 0; i < radix_size; ++i) { buckets[i] = 0; }
        
        // generate Buckets
        for(size_t i = 0; i < input.size(); ++i){
            ++buckets[key_function(input[i], p)];
        }
        size_t sum = 0;
        for(size_t i = 0; i < buckets.size(); ++i){
            sum += buckets[i];
            buckets[i] = sum-buckets[i];
        }
        
        // partition input elements by key
        for(size_t i = 0; i < input.size(); ++i){
            result[buckets[key_function(input[i], p)]++] = input[i];
        }
        
        if (p == 0) { break; }
        
        // collect elements
        std::copy(result.begin(), result.end(), input.begin());
    }
}
} // namespace sacabench::util
