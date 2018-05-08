/*******************************************************************************
 * sacabench/util/heapsort.hpp
 *
 * Copyright (C) 2018 Oliver Magiera
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <cmath>
#include "span.hpp"

namespace sacabench::util {

    /*
     * struct containing the default comparison function.
     * TODO: replace with new group decision (similar to standard)
     */
    struct greater_than {
        template<typename T>
        int64_t operator()(T const& a, T const& b) {
            return (int64_t)a - (int64_t)b;
        }
    };
    /*
     * Returns the position of the left child of a node in a heap.
     *
     * node: The parent to retrieve the child from.
     */
    size_t child_left(size_t node) {
        return 2 * (node+1) - 1;
    }

    /*
     * Returns the position of the right child of a node in a heap.
     *
     * node: The parent to retrieve the child from.
     */
    size_t child_right(size_t node) {
        return 2 * (node + 1);
    }


    /*
     * Applies the max-heap condition on a given node.
     *
     * data: The data to work on.
     * heap_size: The current size of the heap (not necessarily same with
     * data.size())
     * compare_fun: The comparison function to compare elements with.
     */
    template<typename T, typename F=greater_than> void max_heapify
            (span<T> data, size_t heap_size, size_t node, F compare_fun=F()) {
        //TODO: Add assertion: node contained in heap
        //Adapter for "a > b"
        auto greater = [&](auto a, auto b) { return compare_fun(a, b) > 0; };

        auto left = child_left(node);
        auto right = child_right(node);

        //Check wether left child is greater than it's parent
        size_t max_value_index = left < heap_size &&
                                 greater(data[left], data[node]) ? left : node;
        //Check wether right child is greater than current max value
        if (right < heap_size && greater(data[right], data[max_value_index]))
            { max_value_index = right; }

        // Swap elements if swap is needed
        if (max_value_index != node) {
            std::swap(data[node], data[max_value_index]);
            max_heapify(data, heap_size, max_value_index, compare_fun);
        }
    }

    /*
     * Build a heap for a given span, i.e. ensure the max-heap condition for
     * all inner nodes.
     */
    template<typename T, typename F=greater_than> void
        build_max_heap(span<T> data, F compare_fun=F()) {
        // +1 to work around size_t being unsigned
        for (size_t i = std::floor(data.size() / 2); i != 0; --i) {
            // Call max_heapify for all non-leaves
            max_heapify(data, data.size(), i - 1, compare_fun);
        }
    }

    /*
     * Sort an input according to heapsort in ascending order.
     *
     * data: The input data to be sorted.
     * compare_fun: The comparison function to be used.
     */
    template<typename T, typename F=greater_than> void
        heapsort(span<T> data, F compare_fun=F()) {
        build_max_heap(data);
        // Invariant: data[0...heap_size) unsorted,
        // data[heap_size, data.size()) sorted
        for (size_t heap_size = data.size(); heap_size != 1; --heap_size) {
            //Swap first (biggest) and last element in heap
            std::swap(data[0], data[heap_size - 1]);
            // Restore max-heap property for heap with heap_size - 1
            max_heapify(data, heap_size - 1, 0, compare_fun);
        }
    }
}