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

// We use blind sort on sets which are smaller than this threshold.
constexpr auto BLIND_SORT_THRESHOLD = 100;

template <typename sa_index_type>
class saca_run {
private:
    util::string_span input_text;
    size_t alphabet_size;
    span<sa_index_type> suffix_array;
    bucket_data_container<sa_index_type> bd;

    inline bool are_there_unsorted_buckets() { return true; }

    inline std::pair<u_char, u_char> get_smallest_unsorted_bucket() {
        return std::make_pair('a', 'a');
    }

    inline void shallow_sort(const span<sa_index_type> bucket) {
        // We use multikey quicksort for now.
        // FIXME: Abort at depth L and continue with deep_sort();
        util::sort::multikey_quicksort::multikey_quicksort(bucket, input_text);
    }

    inline void deep_sort(const span<sa_index_type> bucket) {
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

    inline void blind_sort(const span<sa_index_type> bucket) {}

    inline void simple_sort(const span<sa_index_type> bucket) {}

    inline void sort_all_buckets() {
        // Sort all buckets.
        while (are_there_unsorted_buckets()) {
            // Find the smallest unsorted bucket.
            const auto unsorted_bucket = get_smallest_unsorted_bucket();
            const auto alpha = unsorted_bucket.first;
            const auto beta = unsorted_bucket.second;

            // Get bucket bounds.
            auto bucket_start = bd.start_of_bucket(alpha, beta);
            auto bucket_end = bd.end_of_bucket(alpha, beta);
            const span<sa_index_type> bucket =
                suffix_array.slice(bucket_start, bucket_end);

            // Shallow sort it.
            shallow_sort(bucket);
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
    }
};
} // namespace sacabench::deep_shallow
