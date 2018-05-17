/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/std_sort.hpp>

namespace sacabench::bucket_pointer_refinement {

class bucket_pointer_refinement {
    public:
        template<typename sa_index>
        static void construct_sa(util::string_span input,
                size_t alphabet_size, util::span<sa_index> sa) {
            size_t const n = input.size();
            if (n == 0) { // There's nothing to do
                return;
            }

            // TODO: choose appropiate value
            size_t bucketsort_depth = 2;
            if (bucketsort_depth > input.size()) {
                bucketsort_depth = input.size();
            }

            auto buckets =
                util::sort::bucketsort_presort(input, alphabet_size,
                        bucketsort_depth, sa);

            // initialize bucket pointers
            util::container<sa_index> bptr;
            initialize_bucket_pointers<sa_index>(input, alphabet_size,
                    bucketsort_depth, sa, bptr);

            refine_all_buckets<sa_index>(buckets, sa, bptr, bucketsort_depth);
        }

    private:
        template<typename sa_index>
        static void initialize_bucket_pointers(util::string_span input,
                size_t alphabet_size, size_t bucketsort_depth,
                util::span<sa_index> sa, util::container<sa_index>& bptr) {
            const size_t n = sa.size();

            // create bucket pointer array
            bptr = util::make_container<sa_index>(n);

            size_t current_bucket = n - 1;
            size_t current_sa_position = n;

            size_t current_code = 0;
            size_t recent_code = current_code;

            // from right to left: Calculate codes in order to determine
            // bucket borders
            do {
                --current_sa_position;
                // find current code by inspecting sa[current_sa_position]
                current_code = code_d<sa_index>(input, alphabet_size,
                        bucketsort_depth, sa[current_sa_position]);

                if (current_code != recent_code) {
                    // we passed a border between two buckets
                    current_bucket = current_sa_position;
                    recent_code = current_code;
                }
                bptr[sa[current_sa_position]] = current_bucket;
            } while (current_sa_position > 0);
        }

        template<typename sa_index>
        static size_t code_d (util::string_span input, size_t alphabet_size,
                size_t depth, size_t start_index) {
            size_t code = 0;
            const size_t stop_index = start_index + depth;
            const size_t real_alphabet_size = alphabet_size + 1; // incl '$'

            while (start_index < stop_index && start_index < input.size()) {
                code *= real_alphabet_size;
                code += input[start_index++];
            }

            return code;
        }

        template<typename sa_index>
        static void refine_all_buckets (
                util::span<util::sort::bucket> buckets, util::span<sa_index> sa,
                util::span<sa_index> bptr, size_t offset) {
            // sort each bucket in naive order
            for (auto& b : buckets) {
                if (b.count > 0) {
                    size_t bucket_end_exclusive = b.position + b.count;
                    refine_single_bucket<sa_index>(offset, offset, bptr,
                            b.position,
                            sa.slice(b.position, bucket_end_exclusive));
                }
            }
        }

        template<typename sa_index>
        static void refine_single_bucket (size_t offset, size_t step_size,
                util::span<sa_index> bptr, size_t bucket_start,
                util::span<sa_index> bucket) {

            if (bucket.size() < 2) {
                return;
            }

            auto sort_key = [=] (sa_index suffix) {
                if (suffix >= bptr.size() - offset) {
                    return (sa_index) 0;
                } else {
                    // Add 1 to sort key in order to prevent collision with
                    // sentinel.
                    return bptr[suffix + offset] + 1;
                }
            };

            // TODO: use ternary quicksort
            util::sort::std_sort(bucket,
                [&sort_key](const auto& lhs, const auto& rhs) {
                    return sort_key(lhs) < sort_key(rhs);
                });

            // Refine bucket pointers
            size_t current_bucket_position = bucket.size();

            size_t current_code = 0;
            size_t recent_code = current_code;

            util::container<bool> right_bounds =
                util::make_container<bool>(bucket.size());
            // from right to left: Calculate codes in order to determine
            // bucket borders and store them in temp_bptr
            do {
                --current_bucket_position;

                // find current code by inspecting sa[current_sa_position]
                current_code = sort_key(bucket[current_bucket_position]);

                if (current_code != recent_code) {
                    // we passed a border between two buckets
                    right_bounds[current_bucket_position] = true;
                    recent_code = current_code;
                }
            } while (current_bucket_position > 0);

            current_bucket_position = bucket.size();
            size_t current_bucket_key = bucket_start + bucket.size() - 1;

            do {
                --current_bucket_position;

                if (right_bounds[current_bucket_position]) {
                    // we passed a border between two buckets
                    current_bucket_key = bucket_start + current_bucket_position;
                }

                bptr[bucket[current_bucket_position]] = current_bucket_key;
            } while (current_bucket_position > 0);

            // Determine sub-buckets
            size_t start_of_bucket = 0;
            size_t end_of_bucket;
            // from right to left: Calculate codes in order to determine
            // bucket borders
            while (start_of_bucket < bucket.size()) {
                end_of_bucket = bptr[bucket[start_of_bucket]] - bucket_start;
                // Sort sub-buckets recursively
                refine_single_bucket<sa_index>(offset + step_size, step_size,
                        bptr, start_of_bucket + bucket_start,
                        bucket.slice(start_of_bucket, ++end_of_bucket));
                start_of_bucket = end_of_bucket;
            }
        }

}; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement

