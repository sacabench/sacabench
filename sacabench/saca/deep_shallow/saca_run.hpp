/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>

#include <util/is_sorted.hpp>
#include <util/ringbuffer.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/introsort.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include "anchor_data.hpp"
#include "blind/sort.hpp"
#include "bucket_data.hpp"
#include "parameters.hpp"

namespace sacabench::deep_shallow {

inline void print_text(const util::string_span text) {
    for (const util::character& c : text) {
        std::cout << (char)(c + 'a' - 1) << " ";
    }
    std::cout << std::endl;
}

template <typename T>
using span = util::span<T>;
using u_char = util::character;

/// \brief Represents a run on a specific input of the Deep Shallow SACA.
///        After construction, the suffix array construction is completed.
template <typename sa_index_type>
class saca_run {
private:
    util::string_span input_text;
    size_t alphabet_size;
    span<sa_index_type> suffix_array;
    bucket_data_container<sa_index_type> bd;
    anchor_data<sa_index_type> ad;

    size_t max_blind_sort_size;

    /// \brief Sorts the suffix arrays in suffix_array by the first two
    ///        characters. Then saves the bucket bounds to `bd`.
    inline void bucket_sort() {

        // Create bucket_data_container object with the right size.
        bd = bucket_data_container<sa_index_type>(alphabet_size);

        // Use bucket sort to sort the suffix array by the first two characters.
        const auto bucket_bounds = util::sort::bucketsort_presort(
            input_text, alphabet_size, 2, suffix_array);

        // Store the bucket bounds in bd.
        bd.set_bucket_bounds(bucket_bounds);
    }

    /// \brief Use Multikey-Quicksort to sort the bucket.
    inline void shallow_sort(const span<sa_index_type> bucket) {
        // We use multikey quicksort. Abort at depth DEEP_SORT_DEPTH.
        util::sort::multikey_quicksort::multikey_quicksort<DEEP_SORT_DEPTH>(
            bucket, input_text, [&](const span<sa_index_type> equal_partition) {
                deep_sort(equal_partition, DEEP_SORT_DEPTH);
            });
    }

    /// \brief Use Induce Sorting, Blind Sorting and Ternary Quicksort to sort
    ///        the bucket.
    /// \param common_prefix_length The number of characters every string in
    ///        bucket shares with each other.
    inline void deep_sort(const span<sa_index_type> bucket,
                          const size_t common_prefix_length) {
        // Try induced sorting. This call returns false, if no ANCHOR and
        // OFFSET are suitable. We then use blind-/quicksort.
        bool induce_sorted_succeeded =
            try_induced_sort(bucket, common_prefix_length);

        if (!induce_sorted_succeeded) {
            if (bucket.size() < max_blind_sort_size) {
                // If the bucket is small enough, we can use blind sorting.
                blind_sort(bucket, common_prefix_length);
            } else {
                // In this case, we use simple quicksort.
                simple_sort(bucket, common_prefix_length);
            }
        }
    }

    /// \brief Use Blind Sorting to sort the bucket.
    inline void blind_sort(const span<sa_index_type> bucket,
                           const size_t common_prefix_length) {
        blind::sort(input_text, bucket, common_prefix_length);
    }

    inline bool try_induced_sort(const span<sa_index_type> bucket,
                                 const sa_index_type common_prefix_length) {
        // Try every suffix index, which is to be sorted
        for (const sa_index_type& si : bucket) {

            // Check, if there is a suitable entry in anchor_data.
            const auto leftmost_suffix_opt = ad.get_leftmost_position(si);
            if (leftmost_suffix_opt.has_value()) {

                // Test, if the suffix is in the valid range, that means
                // entry_in_offset \in [leftmost_suffix, leftmost_suffix +
                // common_prefix_length]
                const auto leftmost_suffix = leftmost_suffix_opt.value();
                if (si < leftmost_suffix &&
                    leftmost_suffix < si + common_prefix_length) {

                    // This is the position the found bucket starts in si.
                    const auto relation = leftmost_suffix - si;

                    // This is the position of the known sorted suffix in its
                    // bucket.
                    const auto sorted_bucket =
                        ad.get_position_in_suffixarray(si);

                    // Get the bucket bounds for the already sorted suffix.
                    const auto left_bucket_bound = bd.start_of_bucket(
                        input_text[leftmost_suffix],
                        input_text[leftmost_suffix +
                                   static_cast<sa_index_type>(1)]);
                    const auto right_bucket_bound = bd.end_of_bucket(
                        input_text[leftmost_suffix],
                        input_text[leftmost_suffix +
                                   static_cast<sa_index_type>(1)]);

                    // Finde alle Elemente von sj zwischen
                    // left_bucket_bound und right_bucket_bound, beginnend mit
                    // der Suche um sorted_bucket.

                    // Allocate memory for ringbuffer, to store found suffixes.
                    auto rb_memory =
                        util::make_container<sa_index_type>(bucket.size());
                    util::ringbuffer<sa_index_type> rb(rb_memory);

                    // This function returns true, if `to_find` is a member of
                    // the bucket to be sorted.
                    const auto contains = [&](const sa_index_type to_find) {
                        for (const sa_index_type& bsi : bucket) {
                            if (to_find == bsi + relation) {
                                return true;
                            }
                        }
                        return false;
                    };

                    // This function checks the suffixes at a given distance
                    // from the pointer into the bucket. If the suffix is one
                    // we're looking for, then add it to the ringbuffer at the
                    // correct location.
                    const auto look_at = [&](const size_t dist) {
                        const size_t left =
                            static_cast<size_t>(sorted_bucket) - dist;
                        const size_t right =
                            static_cast<size_t>(sorted_bucket) + dist;

                        // Check if `left` overflowed.
                        if (static_cast<size_t>(sorted_bucket) >= dist) {
                            // Check, if `left` is still in the bucket we're
                            // searching.
                            if (left >= left_bucket_bound) {
                                if (contains(suffix_array[left])) {
                                    rb.push_front(suffix_array[left] -
                                                  relation);
                                }
                            }
                        }

                        if (right < right_bucket_bound) {
                            if (contains(suffix_array[right])) {
                                rb.push_back(suffix_array[right] - relation);
                            }
                        }
                    };

                    // We already found the first element, because it is stored
                    // in anchor_data.
                    rb.push_front(suffix_array[sorted_bucket] - relation);

                    // Look at increasing distance to `sorted_bucket`.
                    size_t i = 0;
                    while (!rb.is_full()) {
                        ++i;
                        look_at(i);
                    }

                    // Store contents of the ringbuffer to bucket.
                    rb.copy_into(bucket);

                    // The bucket has been sorted with induced sorting.
                    return true;
                }
            }
        }

        return false;
    }

    /// \brief Use ternary quicksort to sort the bucket.
    inline void simple_sort(span<sa_index_type> bucket,
                            const sa_index_type common_prefix_length) {
        const auto compare_suffix = [&](const sa_index_type a,
                                        const sa_index_type b) {
            DCHECK_LT(a + common_prefix_length, input_text.size());
            DCHECK_LT(b + common_prefix_length, input_text.size());

            const util::string_span as =
                input_text.slice(a + common_prefix_length);
            const util::string_span bs =
                input_text.slice(b + common_prefix_length);
            return as < bs;
        };
        util::sort::introsort(bucket, compare_suffix);
    }

    /// \brief Iteratively sort all buckets.
    inline void sort_all_buckets() {
        while (bd.are_buckets_left()) {
            // Find the smallest unsorted bucket.
            const auto unsorted_bucket = bd.get_smallest_bucket();
            const auto alpha = unsorted_bucket.first;
            const auto beta = unsorted_bucket.second;

            if (bd.size_of_bucket(alpha, beta) < 2) {
                // Buckets with a size of 0 or 1 are already sorted.
                // Do nothing.
            } else {
                // Get bucket bounds.
                const auto bucket_start = bd.start_of_bucket(alpha, beta);
                const auto bucket_end = bd.end_of_bucket(alpha, beta);

                DCHECK_LT(bucket_start, bucket_end);

                // Get slice of suffix array, which contains the elements of the
                // bucket.
                const span<sa_index_type> bucket =
                    suffix_array.slice(bucket_start, bucket_end);

                // Shallow sort it.
                shallow_sort(bucket);

                // Debug check: the bucket is correctly suffix sorted.
                // FIXME: get rid of the obnoxious debug warning.
                // DCHECK(is_partially_suffix_sorted(bucket, input_text));

                for (sa_index_type i = 0; i < bucket.size(); ++i) {
                    ad.update_anchor(bucket[i], bucket_start + i);
                }
            }

            // Mark this bucket as sorted.
            bd.mark_bucket_sorted(alpha, beta);
        }
    }

public:
    inline saca_run(util::string_span text, size_t _alphabet_size,
                    span<sa_index_type> sa)
        : input_text(text), alphabet_size(_alphabet_size), suffix_array(sa),
          bd(), ad(text.size()),
          max_blind_sort_size(text.size() / BLIND_SORT_RATIO) {

        // Fill sa with unsorted suffix array.
        for (size_t i = 0; i < sa.size(); ++i) {
            sa[i] = i;
        }

        // Catch corner cases, where input is smaller than bucket-prefix-size.
        if (text.size() < 3) {
            // Use Multikey-Quicksort.
            blind_sort(sa, 0);
        } else {
            // Use bucket sort to sort `sa` by the first two characters.
            bucket_sort();

            // Sort all buckets iteratively.
            sort_all_buckets();
        }
    }
};
} // namespace sacabench::deep_shallow
