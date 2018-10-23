/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>

#include <util/is_sorted.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/introsort.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/tagged_number.hpp>

#include "anchor_data.hpp"
#include "blind/sort.hpp"
#include "bucket_data.hpp"
#include "log.hpp"
#include "parameters.hpp"

namespace sacabench::deep_shallow {

template <typename T>
using span = util::span<T>;
using u_char = util::character;

/// \brief Represents a run on a specific input of the Deep Shallow SACA.
///        After construction, the suffix array construction is completed.
template <typename sa_index_type>
class parallel_saca_run {
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

        logger::get() << "bucket sorting phase done\n";
    }

    /// \brief Use Multikey-Quicksort to sort the bucket.
    inline void shallow_sort(const span<sa_index_type> bucket) {
        // We use multikey quicksort. Abort at depth DEEP_SORT_DEPTH.
        logger::get() << "using shallow sort on " << bucket.size()
                      << " elements.\n";

        size_t ns = duration([&]() {
            util::sort::multikey_quicksort::multikey_quicksort<DEEP_SORT_DEPTH>(
                bucket, input_text,
                [&](const span<sa_index_type> equal_partition) {
                    deep_sort(equal_partition, DEEP_SORT_DEPTH);
                });
        });

        DCHECK(is_partially_suffix_sorted(bucket, input_text));
        logger::get() << "Took " << ns << "ns.\n";
    }

    /// \brief Use Induce Sorting, Blind Sorting and Ternary Quicksort to sort
    ///        the bucket.
    /// \param common_prefix_length The number of characters every string in
    ///        bucket shares with each other.
    inline void deep_sort(const span<sa_index_type> bucket,
                          const size_t common_prefix_length) {
        DCHECK_GE(bucket.size(), 2);

        // Catch this common case and sort it efficiently.
        if (bucket.size() == 2) {
            if (input_text.slice(bucket[1]) < input_text.slice(bucket[0])) {
                std::swap(bucket[0], bucket[1]);
            }
            return;
        }

        // Try induced sorting. This call returns false, if no ANCHOR and
        // OFFSET are suitable. We then use blind-/quicksort.
        bool induce_sorted_succeeded;

        size_t induced_time = duration([&]() {
            induce_sorted_succeeded =
                try_induced_sort(bucket, common_prefix_length);
        });

        if (!induce_sorted_succeeded) {
            logger::get() << "induce-check on " << bucket.size()
                          << " elements took " << induced_time << "ns.\n";
            logger::get().time_spent_induction_testing(induced_time);

            if (blind::MIN_BLINDSORT_SIZE <= bucket.size() &&
                bucket.size() < max_blind_sort_size) {
                // If the bucket is small enough, we can use blind sorting.
                size_t blind_time = duration(
                    [&]() { blind_sort(bucket, common_prefix_length); });
                logger::get() << "using blind sort on " << bucket.size()
                              << " elements took " << blind_time << "ns.\n";
                logger::get().sorted_elements_blind(bucket.size());
                logger::get().time_spent_blind(blind_time);
            } else {
                // In this case, we use simple quicksort.
                size_t quick_time = duration(
                    [&]() { simple_sort(bucket, common_prefix_length); });
                logger::get() << "using quick sort on " << bucket.size()
                              << " elements took " << quick_time << ".\n";
                logger::get().sorted_elements_quick(bucket.size());
                logger::get().time_spent_quick(quick_time);
            }
        } else {
            logger::get() << "induce-sorted.\n";
            logger::get().sorted_elements_induction(bucket.size());
            logger::get().time_spent_induction_sorting(induced_time);
        }
    }

    /// \brief Use Blind Sorting to sort the bucket.
    inline void blind_sort(const span<sa_index_type> bucket,
                           const size_t common_prefix_length) {
        blind::sort(input_text, bucket, common_prefix_length);
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
        util::sort::binary_introsort::sort(bucket,
                                                         compare_suffix);
        DCHECK(is_partially_suffix_sorted(bucket, input_text));
    }

    inline bool try_induced_sort(const span<sa_index_type> bucket,
                                 const sa_index_type common_prefix_length) {
        // Try every suffix index, which is to be sorted
        for (const size_t& si : bucket) {

            // Check, if there is a suitable entry in anchor_data.
            const auto leftmost_suffix_opt = ad.get_leftmost_position(si);
            if (leftmost_suffix_opt.has_value()) {

                // Test, if the suffix is in the valid range, that means
                // entry_in_offset \in [leftmost_suffix, leftmost_suffix +
                // common_prefix_length]
                const size_t leftmost_suffix = leftmost_suffix_opt.value();
                if (si < leftmost_suffix &&
                    leftmost_suffix < si + common_prefix_length) {

                    // This is the position the found bucket starts in si.
                    const size_t relation = leftmost_suffix - si;

                    // Use suffix array with front bit as tag.
                    auto tagged_sa =
                        util::cast_to_tagged_numbers<sa_index_type, 1>(
                            suffix_array);

                    // This is the position of the known sorted suffix in its
                    // bucket.
                    const size_t sorted_bucket =
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

                    // This function returns true, if `to_find` is a member of
                    // the bucket to be sorted.
                    const auto contains = [&](const sa_index_type to_find) {
                        for (const size_t& bsi : bucket) {
                            if (to_find == bsi + relation) {
                                return true;
                            }
                        }
                        return false;
                    };

                    // We already found the first element, because it is stored
                    // in anchor_data.
                    tagged_sa[sorted_bucket].template set<0>(true);
                    size_t tagged = 1;

                    size_t leftmost = sorted_bucket;

                    // This function checks the suffixes at a given distance
                    // from the pointer into the bucket. If the suffix is one
                    // we're looking for, then add it to the ringbuffer at the
                    // correct location.
                    const auto look_at = [&](const size_t dist) {
                        const size_t left = sorted_bucket - dist;
                        const size_t right = sorted_bucket + dist;

                        // Check if `left` overflowed.
                        if (sorted_bucket >= dist) {
                            // Check, if `left` is still in the bucket we're
                            // searching.
                            if (left >= left_bucket_bound) {
                                if (contains(suffix_array[left])) {
                                    leftmost = left;
                                    tagged_sa[left].template set<0>(true);
                                    ++tagged;
                                }
                            }
                        }

                        if (right < right_bucket_bound) {
                            if (contains(suffix_array[right])) {
                                tagged_sa[right].template set<0>(true);
                                ++tagged;
                            }
                        }
                    };

                    // Look at increasing distance to `sorted_bucket`.
                    size_t i = 0;
                    while (tagged != bucket.size()) {
                        ++i;
                        look_at(i);
                    }

                    // Traverse marked elements in-order and remove tag.
                    for (size_t i = 0; i < tagged; ++i) {
                        // Ignore untagged entries in the sorted bucket.
                        while (tagged_sa[leftmost].template get<0>() == false) {
                            ++leftmost;
                        }

                        // Insert the correct index into the to-be-sorted
                        // bucket.
                        bucket[i] =
                            size_t(tagged_sa[leftmost].number()) - relation;

                        // Un-tag the number, so that the SA is valid again.
                        tagged_sa[leftmost].template set<0>(false);
                    }

                    DCHECK(is_partially_suffix_sorted(bucket, input_text));

                    // The bucket has been sorted with induced sorting.
                    return true;
                }
            }
        }

        return false;
    }

    inline void sort_bucket(const u_char alpha, const u_char beta) {
        // Get bucket bounds.
        const auto bucket_start = bd.start_of_bucket(alpha, beta);
        const auto bucket_end = bd.end_of_bucket(alpha, beta);

        DCHECK_LT(bucket_start, bucket_end);

        // Get slice of suffix array, which contains the elements of the
        // bucket.
        const span<sa_index_type> bucket =
            suffix_array.slice(bucket_start, bucket_end);

        if (bucket.size() <= 2) {
            simple_sort(bucket, 2);
        } else {
            // Shallow sort it.
            shallow_sort(bucket);
        }

        // Debug check: the bucket is correctly suffix sorted.
        DCHECK(is_partially_suffix_sorted(bucket, input_text));

        #pragma omp critical
        {
            for (sa_index_type i = 0; i < bucket.size(); ++i) {
                ad.update_anchor(bucket[i], bucket_start + i);
            }
            // Mark this bucket as sorted.
            bd.mark_bucket_sorted(alpha, beta);
        }
    }

    /// \brief Iteratively sort all buckets.
    inline void sort_all_buckets() {

        // Spawn a task pool to run the buckets in parallel
        #pragma omp parallel
        {
            // Schedule the buckets in serial
            #pragma omp single nowait
            while (bd.are_buckets_left()) {

                // Find the smallest unsorted bucket.
                const auto unsorted_bucket = bd.get_smallest_bucket();
                const auto alpha = unsorted_bucket.first;
                const auto beta = unsorted_bucket.second;
                const size_t size_of_bucket = bd.size_of_bucket(alpha, beta);

                if (size_of_bucket < 2) {
                    // Buckets with a size of 0 or 1 are already sorted.
                    // Do nothing.
                } else if (size_of_bucket > 10) {
                    // Sort big buckets in parallel
                    #pragma omp task
                    {
                        sort_bucket(alpha, beta);
                    }
                } else {
                    // Sort small buckets in serial
                    sort_bucket(alpha, beta);
                }
            }
        }

        // At this point, all tasks are synced.
        // We wait here until every bucket is sorted.
    }

public:
    inline parallel_saca_run(util::string_span text, size_t _alphabet_size,
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
            // Use Quicksort.
            simple_sort(sa, 0);
        } else {
            // Use bucket sort to sort `sa` by the first two characters.
            bucket_sort();

            // Sort all buckets iteratively.
            sort_all_buckets();
        }
    }
};
} // namespace sacabench::deep_shallow
