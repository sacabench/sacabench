/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cinttypes>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::util::sort::multikey_quicksort {
// Create a function with compares at one character depth.
template <typename index_type>
struct compare_one_character_at_depth {
public:
    compare_one_character_at_depth(const string_span& input_text)
        : depth(0), input_text(input_text) {}

    // The depth at which we compare.
    index_type depth;

    // 0 if equal, < 0 if the first is smaller, > 0 if the first is larger.
    // Overwrites -> operator for quicksort
    int operator()(const index_type& a, const index_type& b) const {
        int diff = this->input_text[depth + a] - this->input_text[depth + b];
        return diff;
    }

private:
    // A reference to the input text.
    const string_span& input_text;
};

// Create a new key_func to compare two characters at depth 0.
template <typename index_type>
inline compare_one_character_at_depth<index_type>
generate_multikey_key_function(const string_span input_text) {
    compare_one_character_at_depth<index_type> obj(input_text);
    return obj;
}

// Our internal sort function.
template <typename index_type>
inline void multikey_quicksort_internal(
    span<index_type> array,
    compare_one_character_at_depth<index_type>& key_func) {
    // If the set size is only one element, we don't need to sort.
    if (array.size() < 2) {
        return;
    }

    // FIXME: Choose a simple pivot element.
    const index_type& pivot_element = array[0];

    // Swap elements using ternary quicksort partitioning.
    // Casts key_func into type std::function<int(index_type, index_type)>
    auto bounds = ::util::sort::ternary_quicksort::partition(array, key_func,
                                                             pivot_element);

    // Invariant: 0 .. bounds[0] is lesser than pivot_element
    // bounds[0] .. bounds[1] is equal to the pivot_element
    // bounds[1] .. n is greater than the pivot_element
    auto lesser = array.slice(0, bounds.first);
    auto equal = array.slice(bounds.first, bounds.second);
    auto greater = array.slice(bounds.second);

    // Recursively sort the lesser and greater partitions by the same depth.
    multikey_quicksort_internal(lesser, key_func);
    multikey_quicksort_internal(greater, key_func);

    // Sort the equal partition by the next character.
    ++key_func.depth;
    multikey_quicksort_internal(equal, key_func);
    --key_func.depth;
}

// Sort the suffix indices in array by comparing one character in
// input_text.
template <typename index_type>
inline void multikey_quicksort(span<index_type> array,
                               const string_span input_text) {
    // Generate key function which compares only the character at position 0.
    auto key_func = generate_multikey_key_function<index_type>(input_text);

    // Call internal function.
    multikey_quicksort_internal(array, key_func);
}
} // namespace sacabench::util::sort::multikey_quicksort
