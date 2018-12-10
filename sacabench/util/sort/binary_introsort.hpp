/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "common.hpp"
#include "heapsort.hpp"
#include "insertionsort.hpp"

#include <util/bits.hpp>
#include <util/is_sorted.hpp>
#include <util/macros.hpp>
#include <util/signed_size_type.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>

namespace sacabench::util::sort::binary_introsort {

// Anonymous namespace
namespace {

using util::ssize;

template <typename Content, typename Compare>
class introsort_run {
private:
    static constexpr size_t INSERTIONSORT_THRESHOLD = 25;

    const Compare less;
    const size_t max_depth;

    inline SB_FORCE_INLINE const Content&
    smart_pivot(const util::span<Content> data) const SB_NOEXCEPT {
        if(data.size() > 128) {
            return median_of_nine(data, less);
        } else {
            return median_of_three(data, less);
        }
    }

    inline size_t partition(util::span<Content> data,
                            const Content& pivot) const SB_NOEXCEPT {
        size_t left = 0;
        size_t right = data.size() - 1;

        for (;;) {
            while (left < right && left < data.size() &&
                   less(data[left], pivot))
                ++left;
            while (left < right && right > 0 && less(pivot, data[right]))
                --right;
            if (left < right && left < data.size()) {
                std::swap(data[left], data[right]);
            } else {
                break;
            }
        }

        // for (const Content& a : data.slice(0, left)) {
        //     DCHECK_LT(a, pivot);
        //     (void)a;
        // }
        //
        // for (const Content& a : data.slice(left + 1)) {
        //     DCHECK_GE(a, pivot);
        //     (void)a;
        // }

        return left;
    }

    inline void sort(util::span<Content> data, size_t depth) const SB_NOEXCEPT {
        while (data.size() > INSERTIONSORT_THRESHOLD) {

            // Fall back to heapsort
            if (depth >= max_depth) {
                return sort::heapsort(data, less);
            }

            // Choose pivot element: Either by median of nine or by using the
            // first element.
            const Content pivot = smart_pivot(data);

            // Swap elements around ...
            const size_t mid = partition(data, pivot);
            // const auto p = ternary_quicksort::partition(data, less, pivot);
            // const auto left = data.slice(0, p.first);
            // const auto right = data.slice(p.second);

            // These are the left and right partitions
            const auto left = data.slice(0, mid);
            const auto right = data.slice(mid + 1);

            ++depth;

            // if(left.size() < data.size() / 8 || right.size() < data.size() / 8) {
            //     // TODO: Use idea from pdqsort and swap 4 elements.
            // }

            // Sort smaller partition first
            if (left.size() > right.size()) {
                sort(right, depth);
                data = left;
            } else {
                sort(left, depth);
                data = right;
            }
        }
    }

public:
    inline introsort_run(util::span<Content> _data, Compare _less)
        : less(_less), max_depth(2 * floor_log2(_data.size())) {
        if (_data.size() > 1) {
            sort(_data, 0);

            // Final call to insertion sort to sort the partitions we left
            // unsorted.
            sort::insertion_sort(_data, less);
        }
    }
};
} // namespace

template <typename Content, typename Compare = std::less<Content>>
inline void sort(util::span<Content> data, Compare less = Compare()) {
    const introsort_run<Content, Compare> r(data, less);
    DCHECK(is_sorted(data, less));
}
} // namespace sacabench::util::sort::binary_introsort
