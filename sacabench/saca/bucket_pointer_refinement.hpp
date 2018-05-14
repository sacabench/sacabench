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

            auto buckets =
                util::sort::bucketsort_presort(input, alphabet_size,
                        bucketsort_depth, sa);

            // TODO: initialize bucket pointers
            static util::container<sa_index> bptr;
            initialize_bucket_pointers<sa_index>(input, alphabet_size,
                    bucketsort_depth, sa, bptr);

            refine_all_buckets<sa_index>(buckets, sa, bptr, bucketsort_depth);
        }

    private:
        template<typename sa_index>
        static void initialize_bucket_pointers(util::string_span input,
                size_t alphabet_size, size_t bucketsort_depth,
                util::span<sa_index> sa, util::container<sa_index> bptr) {
            const size_t n = sa.size();

            // create bucket pointer array
            bptr = util::make_container<sa_index>(n);

            size_t current_bucket = n - 1;
            size_t current_sa_position = n;

            // TODO: find current code
            size_t current_code = 0;
            size_t recent_code = current_code;

            // from right to left: Calculate codes in order to determine
            // bucket borders
            do {
                --current_sa_position;
                // TODO: find current code by inspecting
                // sa[current_sa_position]
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
                    refine_single_bucket<sa_index>(offset, offset, bptr,
                            sa.slice(b.position - b.count + 1, b.position + 1));
                }
            }
        }

        template<typename sa_index>
        static void refine_single_bucket (size_t offset, size_t step_size,
                util::span<sa_index> bptr, util::span<sa_index> sa) {

            if (sa.size() < 2) {
                return;
            }

            std::cout << "Sorting { ";
            for (auto s : sa) {
                std::cout << (int) s << " ";
            }
            std::cout << "}" << std::endl;

            auto sort_key = [=] (sa_index suffix) {
                if (suffix >= sa.size() - offset) {
                    return (sa_index) 0;
                } else {
                    return bptr[suffix + offset];
                }
            };

            // TODO: Sort sa
            // TODO: Refine bucket pointers
            // TODO: Determine sub-buckets
            // TODO: Sort recursively
        }

}; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement

