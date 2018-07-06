/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>

#include <util/alphabet.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/sort/insertionsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::bucket_pointer_refinement {

class bucket_pointer_refinement {
public:
    static constexpr size_t EXTRA_SENTINELS = 7;
    static constexpr char const* NAME = "BPR";
    static constexpr char const* DESCRIPTION =
        "Bucket-Pointer Refinement according to Klaus-Bernd Schürmann";

    static constexpr size_t INSSORT_THRESHOLD = 200;

    /**\brief Performs a simplified version of the bucket pointer refinement
     * algorithm described by Klaus-Bernd Schürmann and Jens Stoye in "An
     * incomplex algorithm for fast suffix array construction"
     */
    template <typename sa_index>
    static void construct_sa(util::string_span input,
                             util::alphabet const& alphabet,
                             util::span<sa_index> sa) {
        tdc::StatPhase bpr("Phase 1.1");

        size_t alphabet_size = alphabet.size_with_sentinel();

        size_t const n = input.size();
        if (n == 0) { // there's nothing to do
            return;
        }

        // TODO: choose appropiate value
        size_t bucketsort_depth = 2;

        if (alphabet_size <= 9) {
            bucketsort_depth = 7;
        }
        if (9 < alphabet_size && alphabet_size <= 13) {
            bucketsort_depth = 6;
        }
        if (13 < alphabet_size && alphabet_size <= 21) {
            bucketsort_depth = 5;
        }
        if (13 < alphabet_size && alphabet_size <= 21) {
            bucketsort_depth = 5;
        }
        if (21 < alphabet_size && alphabet_size <= 46) {
            bucketsort_depth = 4;
        }
        if (46 < alphabet_size) {
            bucketsort_depth = 3;
        }

        if (bucketsort_depth > n) {
            bucketsort_depth = n;
        }

        // Phase 1.1
        // determine initial buckets with bucketsort
        auto buckets = util::sort::bucketsort_presort(input, alphabet_size,
                bucketsort_depth, sa);

        // Phase 1.2
        // initialize bucket pointers such that each suffix is mapped to the
        // bucket it's currenly in, indexed by right inclusive bound
        bpr.split("Phase 1.2");
        util::container<sa_index> bptr =
            initialize_bucket_pointers<sa_index>(input, alphabet_size,
                    bucketsort_depth, sa);

        // Phase 2
        bpr.split("Phase 2");
        refine_all_buckets<sa_index>(buckets, sa, bptr, bucketsort_depth);
    }

private:
    /**\brief Creates a bptr container which maps suffixes to buckets
     * \param input Input string containing the suffixes
     * \param alphabet_size Number of distinct symbols in input
     * \param bucketsort_depth Length of the prefix to use for code
     *  determination in phase 1
     * \param sa Complete suffix array as available after bucketsort step
     * \return Computed bucket pointer container
     */
    template <typename sa_index>
    static util::container<sa_index>
    initialize_bucket_pointers(util::string_span input, size_t alphabet_size,
                               size_t bucketsort_depth,
                               util::span<sa_index> sa) {
        const size_t n = sa.size();

        // create bucket pointer container
        util::container<sa_index> bptr = util::make_container<sa_index>(n);

        // the rightmost bucket is identified by the rightmost index: n-1
        size_t current_bucket = n - 1;
        size_t current_sa_position = n;

        size_t current_prefix_code = 0;
        size_t recent_prefix_code = current_prefix_code;

        // sequentially scan the sa from right to left and calculate prefix
        // codes of suffixes in order to determine borders between buckets
        do {
            --current_sa_position;
            // find current prefix code by inspecting
            // sa[current_sa_position]
            current_prefix_code =
                code_d<sa_index>(input, alphabet_size, bucketsort_depth,
                                 sa[current_sa_position]);

            if (current_prefix_code != recent_prefix_code) {
                // If the prefix code has changed, we have passed a border
                // between two buckets. The current index is the new
                // bucket's identifier.
                current_bucket = current_sa_position;
                recent_prefix_code = current_prefix_code;
            }
            bptr[sa[current_sa_position]] = current_bucket;
        } while (current_sa_position > 0);

        return bptr;
    }

    /**\brief Calculates the prefix code for a given suffix index
     * \param input Input string containing the suffix
     * \param alphabet_size Number of distinct symbols in input
     * \param depth Length of the prefix to use for code determination
     */
    template <typename sa_index>
    static size_t code_d(util::string_span input, size_t alphabet_size,
                         size_t depth, size_t start_index) {
        size_t const n = input.size();
        size_t code = 0;
        const size_t stop_index = start_index + depth;

        while (start_index < stop_index && start_index < n) {
            // for each symbol of the prefix: extend the code by one symbol
            code *= alphabet_size;
            code += input[start_index++];
        }

        // TODO: This *might* be useless
        while (start_index < stop_index) {
            // for out-of-bound indices (sentinel) fill code with zeros
            code *= alphabet_size;
            ++start_index;
        }

        return code;
    }

    /**\brief Refines given buckets in a suffix array one by one
     * \param buckets Set of buckets each containing starting position and
     * size
     * \param sa Suffix array which will be manipulated
     * \param bptr Bucket pointer array
     * \param offset Length of common prefixes inside pre sorted buckets
     */
    template <typename sa_index>
    static void refine_all_buckets(util::span<util::sort::bucket> buckets,
                                   util::span<sa_index> sa,
                                   util::span<sa_index> bptr, size_t offset) {
        // set sentinel pointers in bptr --> buckets[0] already sorted
        for (size_t sentinel_idx = 0; sentinel_idx < EXTRA_SENTINELS; ++sentinel_idx) {
            bptr[bptr.size() - sentinel_idx - 1] = sentinel_idx;
        }

        // sort remaining buckets in naive order
        for (size_t bucket_idx = 1; bucket_idx < buckets.size(); ++bucket_idx) {
            auto& b = buckets[bucket_idx];
            if (b.count > 0) {
                size_t bucket_end_exclusive = b.position + b.count;
                refine_single_bucket<sa_index>(
                    offset, offset, bptr, b.position,
                    sa.slice(b.position, bucket_end_exclusive));
            }
        }
    }

    /**\brief Refines a single given buckets inside the suffix array
     * \param offset Length of common prefixes inside pre sorted buckets
     * \param step_size Minimum length of common prefixes in other buckets
     * \param bptr Bucket pointers for the complete sa
     * \param bucket_start Starting index of the bucket
     * \param bucket Slice of the suffix array containing only elements of
     * the current bucket.
     */
    template <typename sa_index>
    static void refine_single_bucket(size_t offset, size_t step_size,
                                     util::span<sa_index> bptr,
                                     size_t bucket_start,
                                     util::span<sa_index> bucket) {
        // this bucket is already sorted
        if (bucket.size() < 2) {
            return;
        }

        /* right_bounds indicates jumps between buckets:
         * right_bounds[i] = false: suffix i and i+1 are in the same bucket
         * right_bounds[i] = true: suffix i and i+1 are in different buckets
         * TODO: Find a more memory efficient solution
         */
        util::container<uint8_t> right_bounds =
            util::make_container<uint8_t>(bucket.size());
        std::fill(right_bounds.begin(), right_bounds.end(), false);

        // sort_key maps a suffix s_i to the bucket identifier of suffix
        // s_{i+offset}. If no such suffix exists, it's assumed to be $.
        auto sort_key = [&bptr,&offset](size_t suffix) {
            DCHECK_LT(suffix+offset, bptr.size());
            return static_cast<size_t>(bptr[suffix + offset]);
        };

        bool sortable = false;
        while (true) {
            // check if bucket is sortable
            size_t bucket_code = sort_key(bucket[0]);
            for (size_t idx = 1; idx < bucket.size(); ++idx) {
                if (sort_key(bucket[idx]) != bucket_code) {
                    sortable = true;
                    break;
                }
            }

            if (sortable) {
                break;
            } else {
                // higher offset is needed in order to refine bucket
                offset += step_size;
            }
        }

        // sort the given bucket by using sort_key for each suffix
        if (bucket.size() < INSSORT_THRESHOLD) {
            util::sort::insertion_sort(bucket, util::compare_key(sort_key));
        } else {
            util::sort::ternary_quicksort::ternary_quicksort(
                bucket, util::compare_key(sort_key));
        }

        /* As a consequence of sorting, bucket pointers might have changed.
         * We have to update the bucket pointers for further use.
         */
        size_t current_bucket_position = bucket.size();
        size_t current_code = 0;
        size_t recent_code = current_code;

        // sequentially scan the sa from right to left and calculate prefix
        // codes of suffixes in order to determine borders between buckets
        do {
            --current_bucket_position;

            // find current prefix code by inspecting
            // bucket[current_bucket_position]
            current_code = sort_key(bucket[current_bucket_position]);

            if (current_code != recent_code) {
                // If the prefix code has changed, we have passed a border
                // between two buckets. Remember the border for later use.
                // As the determination of codes is based on the old bucket
                // pointers, bptr can not be updated instantly.
                right_bounds[current_bucket_position] = true;
                recent_code = current_code;
            }
        } while (current_bucket_position > 0);

        current_bucket_position = bucket.size();
        // the key of the rightmost sub bucket is the rightmost index
        size_t current_sub_bucket_key = bucket_start + bucket.size() - 1;

        // do another sequential scan and update the bucket pointers for all
        // elements
        do {
            --current_bucket_position;

            if (right_bounds[current_bucket_position]) {
                // we passed a border between two buckets, so we have a new
                // sub bucket key
                current_sub_bucket_key = bucket_start + current_bucket_position;
            }

            // finally update bptr
            bptr[bucket[current_bucket_position]] = current_sub_bucket_key;
        } while (current_bucket_position > 0);

        /* Refine all sub buckets */

        // increase offset for next level
        offset += step_size;

        size_t start_of_bucket = 0;
        size_t end_of_bucket;

        // from left to right: refine all buckets
        while (start_of_bucket < bucket.size()) {
            end_of_bucket = static_cast<size_t>(bptr[bucket[start_of_bucket]]) -
                            bucket_start;
            // Sort sub-buckets recursively
            refine_single_bucket<sa_index>(
                offset, step_size, bptr, start_of_bucket + bucket_start,
                bucket.slice(start_of_bucket, ++end_of_bucket));
            // jump to the first index of the following sub bucket
            start_of_bucket = end_of_bucket;
        }
    }

}; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement
