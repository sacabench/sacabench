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
#include <saca/bucket_pointer_refinement/bucketsort.hpp>
#include <saca/bucket_pointer_refinement/insertionsort.hpp>
#include <saca/bucket_pointer_refinement/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::bucket_pointer_refinement {

using namespace sacabench::bucket_pointer_refinement::sort;

class bucket_pointer_refinement {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;
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
        tdc::StatPhase bpr("Phase 1");

        size_t alph_size = alphabet.size_with_sentinel();

        size_t const n = input.size();
        if (n == 0) { // there's nothing to do
            return;
        }

        // TODO: choose appropiate value
        size_t bucketsort_depth = 2;

        if (alph_size <= 9) {
            bucketsort_depth = 7;
        }
        if (9 < alph_size && alph_size <= 13) {
            bucketsort_depth = 6;
        }
        if (13 < alph_size && alph_size <= 21) {
            bucketsort_depth = 5;
        }
        if (13 < alph_size && alph_size <= 21) {
            bucketsort_depth = 5;
        }
        if (21 < alph_size && alph_size <= 46) {
            bucketsort_depth = 4;
        }
        if (46 < alph_size) {
            bucketsort_depth = 3;
        }

        if (bucketsort_depth > n) {
            bucketsort_depth = n;
        }

        /**
         * Phase 1
         * determine initial buckets with bucketsort
         */

        util::container<sa_index> bptr =
            util::make_container<sa_index>(n + 2 * bucketsort_depth);
        auto buckets = bucketsort_presort_lightweight(
            input, alphabet.max_character_value(), bucketsort_depth, sa, bptr);

        /**
         * Phase 2
         * Perform comparison based sorting
         */

        bpr.split("Phase 2");
        // set sentinel pointers in bptr --> buckets[0] already sorted
        for (size_t sentinel_idx = 0; sentinel_idx < 2 * bucketsort_depth;
             ++sentinel_idx) {
            // not sure if there are cases where -1 is needed instead of 0
            bptr[bptr.size() - sentinel_idx - 1] = 0;
        }

        // (buckets.size() - 1) because of trailing pseudo bucket
        const size_t in_l1_bucket = (buckets.size() - 1) / alph_size;
        const size_t in_l2_bucket = in_l1_bucket / alph_size;
        const size_t in_l3_bucket = in_l2_bucket / alph_size;

        util::character c_cur, c_suc, c_suc_suc;

        for (c_cur = 0; c_cur < alph_size; ++c_cur) {
            for (c_suc = c_cur + 1; c_suc < alph_size; ++c_suc) {
                for (c_suc_suc = c_cur; c_suc_suc < alph_size; ++c_suc_suc) {
                    const size_t bucket_idx_begin = c_cur * in_l1_bucket +
                                                    c_suc * in_l2_bucket +
                                                    c_suc_suc * in_l3_bucket;
                    const size_t bucket_idx_end =
                        bucket_idx_begin + in_l3_bucket;
                    for (size_t bucket_idx = bucket_idx_begin;
                         bucket_idx < bucket_idx_end; ++bucket_idx) {
                        if (buckets[bucket_idx + 1] > buckets[bucket_idx]) {
                            // if the bucket has at least 1 element
                            refine_single_bucket<sa_index>(
                                bucketsort_depth, bucketsort_depth, bptr,
                                buckets[bucket_idx],
                                sa.slice(buckets[bucket_idx],
                                         buckets[bucket_idx + 1]));
                        }
                    }
                }
            }
        }

        /**
         * Phase 3
         * Perform copy step by Seward
         */

        bpr.split("Phase 3");

        // insert positions on next left scan (inclusive index)
        // "leftmost_undetermined"
        auto lmu = util::make_container<size_t>(alph_size);
        // insert positions on next right scan (exclusive index)
        // "leftmost_undetermined"
        auto rmu = util::make_container<size_t>(alph_size);

        // 2nd level insert positions on next left scan (inclusive index)
        auto sub_lmu = util::make_container<size_t>(alph_size * alph_size);
        // 2nd level insert positions on next right scan (exclusive index)
        auto sub_rmu = util::make_container<size_t>(alph_size * alph_size);

        size_t left_scan_idx, right_scan_idx;

        sa_index suffix_idx;

        size_t bucket_idx;

        // predecessor and pre-predecessor characters of
        util::character c_pre, c_pre_pre;

        for (c_cur = 0; c_cur < alph_size; ++c_cur) {

            /*
             * initialize undetermined pointers
             */

            for (c_pre = c_cur; c_pre < alph_size; ++c_pre) {
                lmu[c_pre] =
                    buckets[c_pre * in_l1_bucket + c_cur * in_l2_bucket];
                rmu[c_pre] =
                    buckets[c_pre * in_l1_bucket + (c_cur + 1) * in_l2_bucket];
                for (c_pre_pre = c_cur + 1; c_pre_pre < alph_size;
                     ++c_pre_pre) {
                    bucket_idx = c_pre_pre * alph_size + c_pre;
                    sub_lmu[bucket_idx] =
                        buckets[c_pre_pre * in_l1_bucket +
                                c_pre * in_l2_bucket + c_cur * in_l3_bucket];
                    sub_rmu[bucket_idx] = buckets[c_pre_pre * in_l1_bucket +
                                                  c_pre * in_l2_bucket +
                                                  (c_cur + 1) * in_l3_bucket];
                }
            }

            /*
             * use copy technique for left buckets
             */

            left_scan_idx = buckets[c_cur * in_l1_bucket];
            while (left_scan_idx < lmu[c_cur]) {
                if (suffix_idx = sa[left_scan_idx]) {
                    c_pre = input[--suffix_idx];
                    if (c_pre >= c_cur) { // bucket of c_pre not yet sorted
                        sa[lmu[c_pre]++] = suffix_idx;
                    }
                    // second level copy
                    if (suffix_idx) {
                        c_pre_pre = input[--suffix_idx];
                        if (c_cur < c_pre_pre && c_pre_pre < c_pre) {
                            // bucket of c_pre_pre is not yet sorted
                            bucket_idx = c_pre_pre * alph_size + c_pre;
                            if (sub_lmu[bucket_idx] < sub_rmu[bucket_idx]) {
                                sa[sub_lmu[bucket_idx]++] = suffix_idx;
                            }
                        }
                    }
                }
                ++left_scan_idx;
            }

            /*
             * use copy technique for right buckets
             */

            right_scan_idx = buckets[(c_cur + 1) * in_l1_bucket];
            while (left_scan_idx < right_scan_idx) {
                --right_scan_idx;
                if (suffix_idx = sa[right_scan_idx]) {
                    c_pre = input[--suffix_idx];
                    if (c_pre >= c_cur) { // bucket of c_pre not yet sorted
                        sa[--rmu[c_pre]] = suffix_idx;
                    }
                    // second level copy
                    if (suffix_idx) {
                        c_pre_pre = input[--suffix_idx];
                        if (c_cur < c_pre_pre && c_pre_pre < c_pre) {
                            // bucket of c_pre_pre not yet sorted
                            bucket_idx = c_pre_pre * alph_size + c_pre;
                            if (sub_rmu[bucket_idx] > sub_lmu[bucket_idx]) {
                                sa[--sub_rmu[bucket_idx]] = suffix_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    /**\brief Increases offset by step_size until sort_key returns different
     *  values for at least two elements inside the bucket
     * \param offset Length of common prefixes inside pre sorted buckets
     * \param step_size Minimum length of common prefixes in other buckets
     * \param bptr Bucket pointers for the complete sa
     * \param bucket_start Starting index of the bucket
     * \param sort_key Key function which uses offset and bptr to compare
     *  suffixes
     */
    template <typename sa_index, typename key_func_type>
    inline static void
    find_offset(size_t& offset, size_t step_size, util::span<sa_index> bptr,
                util::span<sa_index> bucket, key_func_type& sort_key) {
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
    }

    /**\brief Refines a size-2-bucket inside the suffix array
     * \param offset Length of common prefixes inside pre sorted buckets
     * \param step_size Minimum length of common prefixes in other buckets
     * \param bptr Bucket pointers for the complete sa
     * \param bucket_start Starting index of the bucket
     * \param bucket Slice of the suffix array containing only elements of
     * the current bucket.
     */
    template <typename sa_index>
    inline static void refine_size_2_bucket(size_t offset, size_t step_size,
                                            util::span<sa_index> bptr,
                                            size_t bucket_start,
                                            util::span<sa_index> bucket) {
        // sort_key maps a suffix s_i to the bucket identifier of suffix
        // s_{i+offset}. If no such suffix exists, it's assumed to be $.
        auto sort_key = [bptr, &offset](size_t suffix) {
            DCHECK_LT(suffix + offset, bptr.size());
            return static_cast<size_t>(bptr[suffix + offset]);
        };

        find_offset(offset, step_size, bptr, bucket, sort_key);

        if (sort_key(bucket[0]) > sort_key(bucket[1])) {
            sa_index tmp = bucket[0];
            bucket[0] = bucket[1];
            bucket[1] = tmp;
        }

        bptr[bucket[0]] = bucket_start;
        bptr[bucket[1]] = bucket_start + 1;
    }

    /**\brief Refines a single given bucket inside the suffix array
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
        } else if (bucket.size() == 2) {
            refine_size_2_bucket(offset, step_size, bptr, bucket_start, bucket);
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
        auto sort_key = [bptr, &offset](size_t suffix) {
            DCHECK_LT(suffix + offset, bptr.size());
            return static_cast<size_t>(bptr[suffix + offset]);
        };

        find_offset(offset, step_size, bptr, bucket, sort_key);

        // sort the given bucket by using sort_key for each suffix
        if (bucket.size() < INSSORT_THRESHOLD) {
            insertion_sort(bucket, util::compare_key(sort_key));
        } else {
            ternary_quicksort(
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
