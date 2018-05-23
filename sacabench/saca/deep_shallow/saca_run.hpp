/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/sort/bucketsort.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include "blind_trie.hpp"
#include "bucket_data.hpp"

namespace sacabench::deep_shallow {

using u_char = sacabench::util::character;

/// \brief We use blind sort on sets which are smaller than this threshold.
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

    /// \brief Checks, if all buckets have been sorted.
    inline bool are_there_unsorted_buckets() {
        for (u_char i = 0; i <= alphabet_size; ++i) {
            for (u_char j = 0; j <= alphabet_size; ++j) {
                if (!bd.is_bucket_sorted(i, j)) {
                    return true;
                }
            }
        }
        return false;
    }

    /// \brief Returns either a character combination, for which the bucket has
    ///        has not yet been sorted, or std::nullopt.
    inline std::optional<std::pair<u_char, u_char>>
    get_smallest_unsorted_bucket() {
        for (u_char i = 0; i <= alphabet_size; ++i) {
            for (u_char j = 0; j <= alphabet_size; ++j) {
                if (!bd.is_bucket_sorted(i, j)) {
                    return std::optional(std::make_pair(i, j));
                }
            }
        }
        return std::nullopt;
    }

    /// \brief Use Multikey-Quicksort to sort the bucket.
    inline void shallow_sort(const span<sa_index_type> bucket) {
        // We use multikey quicksort for now.
        // FIXME: Abort at depth L and continue with deep_sort();
        util::sort::multikey_quicksort::multikey_quicksort(bucket, input_text);
    }

    /// \brief Use Induce Sorting, Blind Sorting and Ternary Quicksort to sort
    ///        the bucket.
    /// \param common_prefix_length The number of characters every string in
    ///        bucket shares with each other.
    inline void deep_sort(const span<sa_index_type> bucket,
                          const size_t common_prefix_length) {
        // Try induced sorting. This call returns false, if no ANCHOR and
        // OFFSET are suitable.
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
    inline void blind_sort(const span<sa_index_type> bucket) {}

    /// \brief Use ternary quicksort to sort the bucket.
    inline void simple_sort(const span<sa_index_type> bucket) {}

    /// \brief Iteratively sort all buckets.
    inline void sort_all_buckets() {
        // Sort all buckets.
        while (are_there_unsorted_buckets()) {
            // Find the smallest unsorted bucket.
            const auto unsorted_bucket = get_smallest_unsorted_bucket().value();
            const auto alpha = unsorted_bucket.first;
            const auto beta = unsorted_bucket.second;

            // Get bucket bounds.
            auto bucket_start = bd.start_of_bucket(alpha, beta);
            auto bucket_end = bd.end_of_bucket(alpha, beta);
            const span<sa_index_type> bucket =
                suffix_array.slice(bucket_start, bucket_end);

            // Shallow sort it.
            shallow_sort(bucket);
            bd.mark_bucket_sorted(alpha, beta);
        }
    }

public:
    inline saca_run(util::string_span text, size_t _alphabet_size,
                    span<sa_index_type> sa)
        : input_text(text), alphabet_size(_alphabet_size) {
        // Fill sa with unsorted suffix array.
        for (size_t i = 0; i < sa.size(); ++i) {
            sa[i] = i;
        }

        // Use bucket sort to sort sa by the first two characters.
        // Then save the bucket bounds to a bucket_bounds object with name bb.
        bd = bucket_data_container<sa_index_type>(_alphabet_size);

        // Sort all buckets iteratively.
        sort_all_buckets();

        // FIXME: just use multikey quicksort, lol.
        util::sort::multikey_quicksort::multikey_quicksort(sa, text);
    }
};
} // namespace sacabench::deep_shallow
