/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <cmath>
#include "span.hpp"
#include "compare.hpp"

namespace sacabench::util {

    /**
     * \file heapsort.hpp
     * \brief Implements heapsort for spans of input.
     */

    /**
     * \brief Returns the left child of the submitted node.
     * @param node The parent node.
     * @return The left child of node.
     */
    size_t child_left(const size_t node) {
        return 2 * (node+1) - 1;
    }

    /**
     * \brief Returns the right child of the submitted node.
     * @param node The parent node.
     * @return The right child of the node.
     */
    size_t child_right(const size_t node) {
        return 2 * (node + 1);
    }

    /**
     * \brief Applies the max-heap condition for the given node.
     *
     *
     * @tparam T The type of the sorted elements.
     * @tparam F The comparison function. std::less<T> by default.
     * @param data The span to be sorted.
     * @param heap_size The current heap size (considered elements in heap).
     * May be smaller than data.size().
     * @param node The node to check the max-heap condition for.
     * @param compare_fun The function used for comparison.
     */
    template<typename T, typename F=std::less<T>> void max_heapify
            (span<T> data, const size_t heap_size, size_t node, F compare_fun=F()) {
        //TODO: Add assertion: node contained in heap
        //Adapter for "a > b"
        auto greater = as_greater(compare_fun);

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

    /**
     * \brief Constructs a max-heap by rearranging all inner nodes to satisfy
     * the max-heap condition.
     *
     * @tparam T The type of the sorted elements.
     * @tparam F The comparison function type. std::less<T> by default.
     * @param data Span containing all elements to be sorted.
     * @param compare_fun The function to be used when comparing two elements.
     */
    template<typename T, typename F=std::less<T>> void
        build_max_heap(span<T> data, F compare_fun=F()) {
        // +1 to work around size_t being unsigned
        for (size_t i = (data.size() / 2); i != 0; --i) {
            // Call max_heapify for all non-leaves
            // -1 for correct index (first element computed in loop + 1)
            max_heapify(data, data.size(), i - 1, compare_fun);
        }
    }

    /**
     * \brief Sorts a span of elements according to heapsort.
     *
     * Heapsort first builds a heap with max-heap condition. After each
     * iteration, the first (largest) element is swapped with the last element
     * in the heap and restores the max-heap condition for the root (and
     * recursively for all inner nodes which are swapped).
     *
     * @tparam T The type of the sorted elements.
     * @tparam F The comparison function type. std::less<T> by default.
     * @param data Span containing all elements to be sorted.
     * @param compare_fun The function to be used when comparing two elements.
     */
    template<typename T, typename F=std::less<T>> void
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