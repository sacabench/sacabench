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

    struct greater_than {
        template<typename T>
        int64_t operator()(T const& a, T const& b) {
            return (int64_t)a - (int64_t)b;
        }
    };
    /*
     * Returns the position of the left child of a node in a heap.
     */
    size_t child_left(size_t node) {
        return 2 * node;
    }

    /*
     * Returns the position of the right child of a node in a heap.
     */
    size_t child_right(size_t node) {
        return 2 * node + 1;
    }


    /*
     * Applies the max-heap condition on a given node.
     * FIXME: Not sorting correctly (max-heap condition not correctly executed)
     */
    template<typename T, typename F=greater_than> void max_heapify
            (span<T> data, size_t heap_size, size_t node, F compare_fun=F()) {
        //TODO: Add assertion: node contained in heap
        //Adapter for "a > b"
        auto greater = [&](auto a, auto b) { return compare_fun(a, b) > 0; };

        auto left = child_left(node);
        auto right = child_right(node);

        size_t max_value;
        //Check wether left child is greater than it's parent
        left < heap_size && greater(data[left], data[node]) ? max_value = left
                                                     : max_value = node;

        //Check wether right child is greater than current max value
        if (right < heap_size &&
            greater(data[right], data[node])) { max_value = right; }

        // Swap elements if swap is needed
        if (max_value != node) {
            std::swap(data[node], data[max_value]);
            max_heapify(data, heap_size, max_value);
        }
    }

    template<typename T> void build_max_heap(span<T> data) {
        // +1 to work around size_t being unsigned
        for (size_t i = std::floor(data.size() - 1 / 2) + 1; i != 0; --i) {
            max_heapify(data, data.size(), i - 1);
        }
    }

    /*
     * Sort an input according to heapsort in ascending order.
     *
     * data: The input data to be sorted.
     */
    template<typename T> void heapsort(span<T> data) {
        build_max_heap(data);
        // Invariant: data[0...heap_size) unsorted,
        // data[heap_size, data.size()) sorted
        for (size_t heap_size = data.size(); heap_size != 0; --heap_size) {
            //Swap first (biggest) and last element in heap
            std::swap(data[0], data[heap_size - 1]);
            //Restore max-heap property
            max_heapify(data, heap_size, 0);
        }
    }
}