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

#include "blind/sort.hpp"
#include "bucket_data.hpp"

namespace sacabench::deep_shallow {

template <typename T>
using span = util::span<T>;
using u_char = util::character;

/// \brief Instead of continuing to sort with shallow_sort, we switch to
///        deep_sort as this depth.
constexpr auto DEEP_SORT_DEPTH = 50;

/// \brief We use blind sort on sets which are smaller than this threshold.
///        This has a direct effect on the memory footprint of the algorithm.
constexpr auto BLIND_SORT_THRESHOLD = 100;

/// \brief To speed up sorting, we store meta data for every one of the
///        `text_length/SEGMENT_LENGTH` segments. Each of the segments has
///        length `SEGMENT_LENGTH`.
constexpr auto SEGMENT_LENGTH = 200;

/// \brief Represents a run on a specific input of the Deep Shallow SACA.
///        After construction, the suffix array construction is completed.
template <typename sa_index_type>
class saca_run {
private:
    util::string_span input_text;
    size_t alphabet_size;
    span<sa_index_type> suffix_array;
    bucket_data_container<sa_index_type> bd;

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
                          const size_t /*common_prefix_length*/) {
        // Try induced sorting. This call returns false, if no ANCHOR and
        // OFFSET are suitable. We then use blind-/quicksort.
        bool induce_sorted_succeeded = try_induced_sort(bucket);

        if (!induce_sorted_succeeded) {
            if (bucket.size() < BLIND_SORT_THRESHOLD) {
                // If the bucket is small enough, we can use blind sorting.
                blind_sort(bucket);
            } else {
                // In this case, we use simple quicksort.
                simple_sort(bucket);
            }
        }
    }

    /// \brief Use Blind Sorting to sort the bucket.
    inline void blind_sort(const span<sa_index_type> bucket) {
        blind::sort(input_text, bucket);
    }

    inline bool try_induced_sort(const span<sa_index_type> bucket) {
        // TODO.
        return false;
    }

    /// \brief Use ternary quicksort to sort the bucket.
    inline void simple_sort(span<sa_index_type> bucket) {
        const auto compare_suffix = [&](const sa_index_type a,
                                        const sa_index_type b) {
            DCHECK_LT(a, input_text.size());
            DCHECK_LT(b, input_text.size());
            const util::string_span as = input_text.slice(a);
            const util::string_span bs = input_text.slice(b);
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
                auto bucket_start = bd.start_of_bucket(alpha, beta);
                auto bucket_end = bd.end_of_bucket(alpha, beta);

                DCHECK_LT(bucket_start, bucket_end);

                // std::cout << "Sorting [" << bucket_start << ", " <<
                // bucket_end << ") with MKQS." << std::endl;

                // Get slice of suffix array, which contains the elements of the
                // bucket.
                const span<sa_index_type> bucket =
                    suffix_array.slice(bucket_start, bucket_end);

                // Shallow sort it.
                shallow_sort(bucket);
            }

            // Mark this bucket as sorted.
            bd.mark_bucket_sorted(alpha, beta);
        }
    }

public:
    inline saca_run(util::string_span text, size_t _alphabet_size,
                    span<sa_index_type> sa)
        : input_text(text), alphabet_size(_alphabet_size), suffix_array(sa),
          bd() {

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
