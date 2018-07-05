/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cinttypes>
#include <functional>

#include <util/assertions.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::util::sort::multikey_quicksort {
// Create a function with compares at one character depth.
template <typename index_type>
struct compare_one_character_at_depth {
public:
    inline compare_one_character_at_depth(const string_span _input_text)
        : depth(0), input_text(_input_text) {}

    // The depth at which we compare.
    size_t depth;

    // This returns true, if a < b.
    inline bool operator()(const size_t& a, const size_t& b) const {
        DCHECK_LT(depth + a, input_text.size());
        DCHECK_LT(depth + b, input_text.size());

        const character at_a = this->input_text[depth + a];
        const character at_b = this->input_text[depth + b];
        const bool diff = at_a < at_b;

        return diff;
    }

private:
    // A reference to the input text.
    const string_span input_text;
};

// Create a new key_func to compare two characters at depth 0.
template <typename index_type>
inline compare_one_character_at_depth<index_type>
generate_multikey_key_function(const string_span input_text) {
    compare_one_character_at_depth<index_type> obj(input_text);
    return obj;
}

template <typename index_type>
struct default_recursion_function {
    inline void operator()(span<index_type>) {}
};

// Our internal sort function.
template <size_t abort_at_depth,
          typename index_type,
          typename Compare = compare_one_character_at_depth<index_type>,
          typename Fn = default_recursion_function<index_type>>
inline void multikey_quicksort_internal(
    span<index_type> array,
    Compare& key_func,
    Fn fn = Fn()) {
    // If the set size is only one element, we don't need to sort.
    if (array.size() < 2) {
        return;
    }

    // Choose a simple pivot element.
    const index_type pivot_element =
        (array.size() >= 9)
            ? sort::ternary_quicksort::median_of_nine(array, key_func)
            : array[0];

    // Swap elements using ternary quicksort partitioning.
    std::pair<size_t, size_t> bounds;
    size_t d = key_func.depth;

    // Skip recursion if no smaller/bigger partitions are found.
    for(;;++key_func.depth) {
        bounds = sort::ternary_quicksort::partition(array, key_func, pivot_element);
        if (bounds.first != 0 || bounds.second != array.size()) {
            break;
        }
    }

    // Invariant: 0 .. bounds[0] is lesser than pivot_element
    // bounds[0] .. bounds[1] is equal to the pivot_element
    // bounds[1] .. n is greater than the pivot_element
    auto lesser = array.slice(0, bounds.first);
    auto equal = array.slice(bounds.first, bounds.second);
    auto greater = array.slice(bounds.second);

    // Recursively sort the lesser and greater partitions by the same depth.
    multikey_quicksort_internal<abort_at_depth>(lesser, key_func, fn);
    multikey_quicksort_internal<abort_at_depth>(greater, key_func, fn);

    // Sort the equal partition by the next character.
    ++key_func.depth;
    if (abort_at_depth != 0 && static_cast<size_t>(key_func.depth) >= abort_at_depth) {
        fn(equal);
    } else {
        multikey_quicksort_internal<abort_at_depth>(equal, key_func, fn);
    }
    key_func.depth = d;
}

// Sort the suffix indices in array by comparing one character in
// input_text.
template <typename index_type>
inline void multikey_quicksort(span<index_type> array,
                               const string_span input_text) {
    // Generate key function which compares only the character at position 0.
    auto key_func = generate_multikey_key_function<index_type>(input_text);

    // Call internal function.
    multikey_quicksort_internal<0>(array, key_func);
}
// Sort the suffix indices in array by comparing one character according to
// the submitted compare_func. compare_func needs depth-attribute (used for
// comparing current index) in order to work properly.
template <typename index_type, typename Compare>
inline void multikey_quicksort(span<index_type> array,
        const string_span input_text, Compare& key_func) {
    multikey_quicksort_internal<0, index_type, Compare>(array, key_func);
}


// Sort the suffix indices in array by comparing one character in
// input_text according to the submitted compare function. Abort at depth
// max_depth and call deep_sort instead.
template <size_t abort_at_depth, typename index_type, typename Fn,
        typename Compare>
inline void multikey_quicksort(span<index_type> array,
                               const string_span input_text, Fn fn,
                               Compare& key_func) {
    // Call internal function.
    multikey_quicksort_internal<abort_at_depth>(array, key_func, fn);
}

// Sort the suffix indices in array by comparing one character in
// input_text. Abort at depth max_depth and call deep_sort instead.
template <size_t abort_at_depth, typename index_type, typename Fn>
inline void multikey_quicksort(span<index_type> array,
                               const string_span input_text, Fn fn) {
    // Generate key function which compares only the character at position 0.
    auto key_func = generate_multikey_key_function<index_type>(input_text);

    // Call internal function.
    multikey_quicksort_internal<abort_at_depth>(array, key_func, fn);
}
} // namespace sacabench::util::sort::multikey_quicksort
