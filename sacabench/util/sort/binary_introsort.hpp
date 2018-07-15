/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/bits.hpp>
#include <util/is_sorted.hpp>
#include <util/macros.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>

namespace sacabench::util::sort::binary_introsort {

// Anonymous namespace
namespace {

using util::ssize;

template <typename Content, typename Compare>
class introsort_run {
private:
    static constexpr size_t INSERTIONSORT_THRESHOLD = 310;

    const Compare less;
    const size_t max_depth;

    inline void insertion_sort(util::span<Content> data) const SB_NOEXCEPT {
        DCHECK_LT(data.size(), INSERTIONSORT_THRESHOLD);

        // Invariant: data[0 .. end_of_sorted_partition] is already correctly
        // sorted.
        for (size_t end_of_sorted_partition = 1;
             end_of_sorted_partition < data.size(); ++end_of_sorted_partition) {
            DCHECK(is_sorted(data.slice(0, end_of_sorted_partition), less));

            // end_of_sorted_partition is now the new element.
            // bubble it up to the front.
            for (ssize i = end_of_sorted_partition; i > 0; --i) {

                // Check if they defy the ordering and swap them if needed
                if (less(data[i], data[i - 1])) {
                    std::swap(data[i], data[i - 1]);
                }
            }
        }
    }

    inline void heapsort(util::span<Content> data) const SB_NOEXCEPT {
        // FIXME
        std::sort(data.begin(), data.end());
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

        return left;
    }

    inline void sort(util::span<Content> data,
                     const size_t depth) const SB_NOEXCEPT {
        DCHECK_GE(data.size(), 2);

        if (data.size() < INSERTIONSORT_THRESHOLD) {
            return insertion_sort(data);
        }

        if (depth >= max_depth) {
            return heapsort(data);
        }

        // FIXME: Choose simple pivot element
        const Content& pivot = data[0];

        // Swap elements around ...
        const size_t mid = partition(data, pivot);

        // These are the left and right partitions
        const auto left = data.slice(0, mid);
        const auto right = data.slice(mid + 1);

        // Recursive call on left and right partitions
        if (left.size() > 1)
            sort(left, depth + 1);
        if (right.size() > 1)
            sort(right, depth + 1);

        DCHECK(is_sorted(left, less));
        DCHECK(is_sorted(right, less));
    }

public:
    inline introsort_run(util::span<Content> _data, Compare _less)
        : less(_less), max_depth(ceil_log2(_data.size())) {
        if (_data.size() > 1) {
            sort(_data, 0);
        }
    }
};
} // namespace

template <typename Content, typename Compare = std::less<Content>>
inline void sort(util::span<Content> data, Compare less = Compare()) {
    // std::cout << data << std::endl;
    introsort_run<Content, Compare> r(data, less);
    // std::cout << data << std::endl;
    DCHECK(is_sorted(data, less));
}
} // namespace sacabench::util::sort::binary_introsort
