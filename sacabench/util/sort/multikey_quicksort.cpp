/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

void util::sort::multikey_quicksort(span<index_type> array, const input_type& input_text) {
    // Begin with first character.
    index_type depth = 0;

    // Generate key function which compares only the character at position depth.
    auto key_func = generate_multikey_key_function(input_text, depth);

    // Call internal function.
    multikey_quicksort_internal(array, key_func, depth);
}

void util::sort::multikey_quicksort_internal(span<index_type> array,
        util::sort::multikey_quicksort::compare_one_character_at_depth& key_func) {
    // If the set size is only one element, we don't need to sort.
    if(array.size() < 2) { return; }

    // FIXME: Choose a simple pivot element.
    const index_type& pivot_element = array[0];

    // Swap elements using ternary quicksort partitioning.
    auto bounds = util::sort::ternary_quicksort::partition(array, key_func, pivot);

    // Invariant: 0 .. bounds[0] is lesser than pivot_element
    // bounds[0] .. bounds[1] is equal to the pivot_element
    // bounds[1] .. n is greater than the pivot_element
    auto lesser = array.slice(0, bounds[0]);
    auto equal = array.slice(bounds[0], bounds[1]);
    auto greater = array.slice(bounds[1]);

    // Recursively sort the lesser and greater partitions by the same depth.
    multikey_quicksort_internal(lesser, key_func);
    multikey_quicksort_internal(greater, key_func);

    // Sort the equal partition by the next character.
    ++key_func.depth;
    multikey_quicksort_internal(equal, key_func);
    --key_func.depth;
}
