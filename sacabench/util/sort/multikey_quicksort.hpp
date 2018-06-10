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
    index_type depth;

    // This returns true, if a < b.
    inline bool operator()(const index_type& a, const index_type& b) const {
        const bool a_is_too_short = depth + a >= input_text.size();
        const bool b_is_too_short = depth + b >= input_text.size();

        if (a_is_too_short) {
            if (b_is_too_short) {
                // but if both are over the edge, one cannot be smaller.
                return false;
            }

            // b should be larger
            return true;
        }

        if (b_is_too_short) {
            // a should be larger
            return false;
        }

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
template <typename index_type,
          typename recursion_function = default_recursion_function<index_type>>
inline void multikey_quicksort_internal(
    span<index_type> array,
    compare_one_character_at_depth<index_type>& key_func,
    const std::optional<size_t> abort_at_depth = std::nullopt,
    recursion_function fn = recursion_function()) {
    // If the set size is only one element, we don't need to sort.
    if (array.size() < 2) {
        return;
    }

    // FIXME: Choose a simple pivot element.
    const index_type pivot_element =
        (array.size() >= 9)
            ? sort::ternary_quicksort::median_of_nine(array, key_func)
            : array[0];

    // Swap elements using ternary quicksort partitioning.
    auto bounds =
        sort::ternary_quicksort::partition(array, key_func, pivot_element);

    // Invariant: 0 .. bounds[0] is lesser than pivot_element
    // bounds[0] .. bounds[1] is equal to the pivot_element
    // bounds[1] .. n is greater than the pivot_element
    auto lesser = array.slice(0, bounds.first);
    auto equal = array.slice(bounds.first, bounds.second);
    auto greater = array.slice(bounds.second);

    // Recursively sort the lesser and greater partitions by the same depth.
    multikey_quicksort_internal(lesser, key_func, abort_at_depth, fn);
    multikey_quicksort_internal(greater, key_func, abort_at_depth, fn);

    // Sort the equal partition by the next character.
    ++key_func.depth;
    if (abort_at_depth.has_value() && key_func.depth >= abort_at_depth) {
        fn(equal);
    } else {
        multikey_quicksort_internal(equal, key_func, abort_at_depth, fn);
    }
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

// Sort the suffix indices in array by comparing one character in
// input_text. Abort at depth max_depth and call deep_sort instead.
template <typename index_type, typename recursion_function>
inline void multikey_quicksort(span<index_type> array,
                               const string_span input_text,
                               const size_t max_depth, recursion_function fn) {
    // Generate key function which compares only the character at position 0.
    auto key_func = generate_multikey_key_function<index_type>(input_text);

    // Call internal function.
    multikey_quicksort_internal(array, key_func, max_depth, fn);
}
} // namespace sacabench::util::sort::multikey_quicksort
