/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>

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

    /// \brief Checks, if all buckets have been sorted.
    inline bool are_there_unsorted_buckets() {
        return get_smallest_unsorted_bucket().has_value();
    }

    /// \brief Returns either a character combination, for which the bucket has
    ///        has not yet been sorted, or std::nullopt.
    inline std::optional<std::pair<u_char, u_char>>
    get_smallest_unsorted_bucket() {
        // FIXME: this just tries every possible character combination, until
        //        one is not sorted.
        for (u_char i = 0; i <= alphabet_size; ++i) {
            for (u_char j = 0; j <= alphabet_size; ++j) {
                if (!bd.is_bucket_sorted(i, j)) {
                    return std::optional(std::make_pair(i, j));
                }
            }
        }

        // There are no unsorted buckets.
        return std::nullopt;
    }

    /// \brief Use Multikey-Quicksort to sort the bucket.
    inline void shallow_sort(const span<sa_index_type> bucket) {
        // We use multikey quicksort. Abort at depth DEEP_SORT_DEPTH.
        util::sort::multikey_quicksort::multikey_quicksort(
            bucket, input_text, DEEP_SORT_DEPTH,
            [&](const span<sa_index_type> equal_partition) {
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
            if (bucket.size() < BLIND_SORT_THRESHOLD) {
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
                                 const size_t common_prefix_length) {
        // FIXME: don't try every suffix index?
        for (const sa_index_type& si : bucket) {
            const auto leftmost_suffix_opt = ad.get_leftmost_position(si);

            if (leftmost_suffix_opt.has_value()) {
                const auto leftmost_suffix = leftmost_suffix_opt.value();

                if (si < leftmost_suffix &&
                    leftmost_suffix < si + common_prefix_length) {
                    const auto relation = leftmost_suffix - si;
                    const auto sorted_bucket =
                        ad.get_position_in_suffixarray(si);

                    std::cout << "Common Prefix: ";
                    print_text(input_text.slice(si, si + common_prefix_length));

                    std::cout << "gefundener Bucket: ";
                    print_text(
                        input_text.slice(leftmost_suffix, leftmost_suffix + 2));

                    const auto left_bucket_bound =
                        bd.start_of_bucket(input_text[leftmost_suffix],
                                           input_text[leftmost_suffix + 1]);
                    const auto right_bucket_bound =
                        bd.end_of_bucket(input_text[leftmost_suffix],
                                         input_text[leftmost_suffix + 1]);

                    std::cout << "sortierter Bucket:" << std::endl;
                    for (auto i = left_bucket_bound; i < right_bucket_bound;
                         ++i) {
                        print_text(input_text.slice(suffix_array[i]));
                    }

                    for (const sa_index_type& sj : bucket) {
                        std::cout << "Finde ";
                        print_text(input_text.slice(sj + relation));
                    }

                    return false;
                }
            }
        }

        return false;
    }

    /// \brief Use ternary quicksort to sort the bucket.
    inline void simple_sort(span<sa_index_type> bucket,
                            const size_t common_prefix_length) {
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
        while (are_there_unsorted_buckets()) {
            // Find the smallest unsorted bucket.
            const auto unsorted_bucket = get_smallest_unsorted_bucket().value();
            const auto alpha = unsorted_bucket.first;
            const auto beta = unsorted_bucket.second;

            if (bd.size_of_bucket(alpha, beta) < 2) {
                // Buckets with a size of 0 or 1 are already sorted.
                // Do nothing.
            } else {
                // std::cout << "Sorting bucket B_{" << (size_t)alpha << ", " <<
                // (size_t)beta << "}..." << std::endl;

                // Get bucket bounds.
                const auto bucket_start = bd.start_of_bucket(alpha, beta);
                const auto bucket_end = bd.end_of_bucket(alpha, beta);

                DCHECK_LT(bucket_start, bucket_end);

                // std::cout << "Sorting [" << bucket_start << ", " <<
                // bucket_end << ") with MKQS." << std::endl;

                // Get slice of suffix array, which contains the elements of the
                // bucket.
                const span<sa_index_type> bucket =
                    suffix_array.slice(bucket_start, bucket_end);

                // Shallow sort it.
                shallow_sort(bucket);

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
          bd(), ad(text.size()) {

        // Fill sa with unsorted suffix array.
        for (size_t i = 0; i < sa.size(); ++i) {
            sa[i] = i;
        }

        // Catch corner cases, where input is smaller than bucket-prefix-size.
        if (text.size() < 3) {
            // Use Multikey-Quicksort.
            shallow_sort(sa);
        } else {
            // Use bucket sort to sort `sa` by the first two characters.
            bucket_sort();

            // Sort all buckets iteratively.
            sort_all_buckets();
        }
    }
};
} // namespace sacabench::deep_shallow
