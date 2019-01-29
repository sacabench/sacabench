/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>
#include <omp.h>

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

    omp_lock_t writelock;
    // omp_lock_t inducing_lock;

    size_t tasks, tasks_done;

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

    /// \brief Sorts the suffix arrays in suffix_array by the first two
    ///        characters. Then saves the bucket bounds to `bd`.
    // inline void parallel_bucket_sort() {
    //     // Create bucket_data_container object with the right size.
    //     bd = bucket_data_container<sa_index_type>(alphabet_size);

    //     // size_t counts[256 * 256] = { 0 };
    //     size_t global_counts[256 * 256] = { 0 };

    //     #pragma omp parallel for schedule(static, 256)
    //     for(size_t i = 0; i < input_text.size() - 1; ++i) {
    //         const u_char alpha = input_text[i];
    //         const u_char beta = input_text[i + 1];

    //         // Index counts by [ alpha | beta ]
    //         #pragma omp critical
    //         ++global_counts[ uint16_t(alpha) << 8 | beta ];
    //     }
        
    //     // std::cout << span<size_t>(counts, 256*256) << std::endl;

    //     // #pragma omp critical
    //     // {
    //     //     for(size_t i = 0; i < 256 * 256; ++i) {
    //     //         global_counts[i] += counts[i];
    //     //     }
    //     // }

    //     // #pragma omp barrier

    //     // std::cout << span<size_t>(global_counts, 256*256) << std::endl;

    //     // for(uint16_t alpha = 0; alpha < (uint16_t) 256; ++alpha) {
    //     //     for(uint16_t beta = 0; beta < (uint16_t) 256; ++beta) {

    //     //         size_t count = 0;

    //     //         for(size_t i = 0; i < input_text.size() - 1; ++i) {
    //     //             if (input_text[i] == alpha && input_text[i + 1] == beta) {
    //     //                 ++count;
    //     //             }
    //     //         }

    //     //         std::cout << (size_t) alpha << "x" << (size_t) beta << std::endl;
    //     //         DCHECK_EQ(count, global_counts[ size_t(alpha) << 8 | size_t(beta) ]);
    //     //         std::cout << " ok" << std::endl;
    //     //     }
    //     // }

    //     // Overwrite global counts with prefix sum / bucket starts
    //     size_t sum = 0;
    //     for(size_t i = 0; i < 256 * 256; ++i) {
    //         size_t sum2 = sum;
    //         sum += global_counts[i];
    //         global_counts[i] = sum2;
    //     }

    //     // Print counts array
    //     util::kd_array<bucket_information<sa_index_type>, 2> bounds({alphabet_size + 1, alphabet_size + 1});

    //     for(u_char alpha = 0; alpha < alphabet_size + 1; ++alpha) {
    //         for(u_char beta = 0; beta < alphabet_size + 1; ++beta) {
    //             bounds[{alpha, beta}].starting_position = global_counts[ uint16_t(alpha) << 8 | beta ];
    //             bounds[{alpha, beta}].is_sorted = false;
    //         }
    //     }

    //     bd.set_bucket_bounds(std::move(bounds), input_text.size());

    //     for(size_t i = 0; i < input_text.size() - 1; ++i) {
    //         const u_char alpha = input_text[i];
    //         const u_char beta = input_text[i + 1];
    //         const size_t bucket = global_counts[ uint16_t(alpha) << 8 | beta ] ++;
    //         std::cout << "i = " << i << ", bucket = " << bucket << std::endl;
    //         suffix_array[ bucket + 1 ] = i;
    //     }

    //     suffix_array[0] = input_text.size() - 1;

    //     std::cout << "SA:" << suffix_array << std::endl;

    //     logger::get() << "parallel bucket sorting phase done\n";
    // }

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
        util::sort::binary_introsort::sort(bucket, compare_suffix);
        DCHECK(is_partially_suffix_sorted(bucket, input_text));
    }

    inline bool try_induced_sort(const span<sa_index_type> /*bucket*/,
                                 const sa_index_type /*common_prefix_length*/) {

        // omp_set_lock(&inducing_lock);

        // // Try every suffix index, which is to be sorted
        // for (const size_t& si : bucket) {

        //     // Check, if there is a suitable entry in anchor_data.
        //     omp_set_lock(&writelock);
        //     const auto leftmost_suffix_opt = ad.get_leftmost_position(si);
        //     omp_unset_lock(&writelock);

        //     if (leftmost_suffix_opt.has_value()) {

        //         // Test, if the suffix is in the valid range, that means
        //         // entry_in_offset \in [leftmost_suffix, leftmost_suffix +
        //         // common_prefix_length]
        //         const size_t leftmost_suffix = leftmost_suffix_opt.value();
        //         if (si < leftmost_suffix &&
        //             leftmost_suffix < si + common_prefix_length) {

        //             // This is the position the found bucket starts in si.
        //             const size_t relation = leftmost_suffix - si;

        //             // Use suffix array with front bit as tag.
        //             auto tagged_sa =
        //                 util::cast_to_tagged_numbers<sa_index_type, 1>(
        //                     suffix_array);

        //             // This is the position of the known sorted suffix in its
        //             // bucket.
        //             omp_set_lock(&writelock);

        //             const size_t sorted_bucket =
        //                 ad.get_position_in_suffixarray(si);

        //             // Get the bucket bounds for the already sorted suffix.
        //             const auto left_bucket_bound = bd.start_of_bucket(
        //                 input_text[leftmost_suffix],
        //                 input_text[leftmost_suffix +
        //                            static_cast<sa_index_type>(1)]);
        //             const auto right_bucket_bound = bd.end_of_bucket(
        //                 input_text[leftmost_suffix],
        //                 input_text[leftmost_suffix +
        //                            static_cast<sa_index_type>(1)]);
        //             omp_unset_lock(&writelock);

        //             // Finde alle Elemente von sj zwischen
        //             // left_bucket_bound und right_bucket_bound, beginnend mit
        //             // der Suche um sorted_bucket.

        //             // This function returns true, if `to_find` is a member of
        //             // the bucket to be sorted.
        //             const auto contains = [&](const sa_index_type to_find) {
        //                 for (const size_t& bsi : bucket) {
        //                     if (to_find == bsi + relation) {
        //                         return true;
        //                     }
        //                 }
        //                 return false;
        //             };

        //             // We already found the first element, because it is stored
        //             // in anchor_data.
        //             tagged_sa[sorted_bucket].template set<0>(true);
        //             size_t tagged = 1;

        //             size_t leftmost = sorted_bucket;

        //             // This function checks the suffixes at a given distance
        //             // from the pointer into the bucket. If the suffix is one
        //             // we're looking for, then add it to the ringbuffer at the
        //             // correct location.
        //             const auto look_at = [&](const size_t dist) {
        //                 const size_t left = sorted_bucket - dist;
        //                 const size_t right = sorted_bucket + dist;

        //                 // Check if `left` overflowed.
        //                 if (sorted_bucket >= dist) {
        //                     // Check, if `left` is still in the bucket we're
        //                     // searching.
        //                     if (left >= left_bucket_bound) {
        //                         if (contains(suffix_array[left])) {
        //                             leftmost = left;
        //                             tagged_sa[left].template set<0>(true);
        //                             ++tagged;
        //                         }
        //                     }
        //                 }

        //                 if (right < right_bucket_bound) {
        //                     if (contains(suffix_array[right])) {
        //                         tagged_sa[right].template set<0>(true);
        //                         ++tagged;
        //                     }
        //                 }
        //             };

        //             // Look at increasing distance to `sorted_bucket`.
        //             size_t i = 0;
        //             while (tagged != bucket.size()) {
        //                 ++i;
        //                 look_at(i);
        //             }

        //             // Traverse marked elements in-order and remove tag.
        //             for (size_t i = 0; i < tagged; ++i) {
        //                 // Ignore untagged entries in the sorted bucket.
        //                 while (tagged_sa[leftmost].template get<0>() == false) {
        //                     ++leftmost;
        //                 }

        //                 // Insert the correct index into the to-be-sorted
        //                 // bucket.
        //                 bucket[i] =
        //                     size_t(tagged_sa[leftmost].number()) - relation;

        //                 // Un-tag the number, so that the SA is valid again.
        //                 tagged_sa[leftmost].template set<0>(false);
        //             }

        //             omp_unset_lock(&inducing_lock);

        //             DCHECK(is_partially_suffix_sorted(bucket, input_text));

        //             // The bucket has been sorted with induced sorting.
        //             return true;
        //         }
        //     }
        // }

        // omp_unset_lock(&inducing_lock);

        return false;
    }

    inline void sort_bucket(const u_char alpha, const u_char beta) {
        // Get bucket bounds.
        omp_set_lock(&writelock);
        const auto bucket_start = bd.start_of_bucket(alpha, beta);
        const auto bucket_end = bd.end_of_bucket(alpha, beta);
        omp_unset_lock(&writelock);

        DCHECK_LT(bucket_start, bucket_end);

        // Get slice of suffix array, which contains the elements of the
        // bucket.
        const span<sa_index_type> bucket =
            suffix_array.slice(bucket_start, bucket_end);

        if (bucket.size() <= 2) {
            // Special case: use quicksort for small buckets
            simple_sort(bucket, 2);
        } else {
            // Shallow sort it.
            shallow_sort(bucket);
        }

        // Debug check: the bucket is correctly suffix sorted.
        DCHECK(is_partially_suffix_sorted(bucket, input_text));

        omp_set_lock(&writelock);
        for (sa_index_type i = 0; i < bucket.size(); ++i) {
            ad.update_anchor(bucket[i], bucket_start + i);
        }
        // Mark this bucket as sorted.
        // bd.mark_bucket_sorted(alpha, beta);
        omp_unset_lock(&writelock);

        // #pragma omp critical
        // std::cout << ++tasks_done << "/" << tasks << ": " << bucket.size() << " elements, thread id was " << omp_get_thread_num() << std::endl;
    }

    /// \brief Iteratively sort all buckets.
    inline void sort_all_buckets() {
        // Schedule the buckets in serial
        while (bd.are_buckets_left()) {

            // Find the smallest unsorted bucket.
            const auto unsorted_bucket = bd.get_smallest_bucket();
            const auto alpha = unsorted_bucket.first;
            const auto beta = unsorted_bucket.second;
            const size_t size_of_bucket = bd.size_of_bucket(alpha, beta);

            if (size_of_bucket < 2) {
                // Buckets with a size of 0 or 1 are already sorted.
                // Do nothing.
            } else {
                // tasks++;

                // Sort big buckets in parallel
                #pragma omp task
                {
                    sort_bucket(alpha, beta);
                }
            }
        }

        // Start execution of buckets.
        omp_unset_lock(&writelock);

        // At this point, all tasks are synced.
        // We wait here until every bucket is sorted.
    }

public:
    inline parallel_saca_run(util::string_span text, size_t _alphabet_size,
                    span<sa_index_type> sa)
        : input_text(text), alphabet_size(_alphabet_size), suffix_array(sa),
          bd(_alphabet_size), ad(text.size()),
          max_blind_sort_size(text.size() / BLIND_SORT_RATIO), tasks(0), tasks_done(0) {

        omp_init_lock(&writelock);
        // omp_init_lock(&inducing_lock);

        // std::cout << "start" << std::endl;
        // // Fill sa with unsorted suffix array.
        // for (size_t i = 0; i < sa.size(); ++i) {
        //     sa[i] = 999;
        // }
        // std::cout << "mid" << std::endl;

        for (size_t i = 0; i < sa.size(); ++i) {
            sa[i] = i;
        }

        // Catch corner cases, where input is smaller than bucket-prefix-size.
        if (text.size() < 3) {
            // Use Quicksort.
            simple_sort(sa, 0);
        } else {
            // Spawn a task pool to run the buckets in parallel
            // #pragma omp parallel
            // {
            //     #pragma omp master
            //     {
                    // Use bucket sort to sort `sa` by the first two characters.
                    bucket_sort();

                    // std::cout << "end" << std::endl;

                    // Sort all buckets iteratively.
                    sort_all_buckets();
            //     }
            // }
        }

        omp_destroy_lock(&writelock);
        // omp_destroy_lock(&inducing_lock);
    }
};
} // namespace sacabench::deep_shallow
