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

                util::sort::bucketsort_presort(input, alphabet_size,
                        bucketsort_depth, sa);
                // TODO: initialize bucket pointers
                initialize_bucket_pointers(input, alphabet_size,
                        bucketsort_depth, sa);
            }
        private:
            template<typename sa_index>
            static util::container<sa_index> bptr;

            template<typename sa_index>
            static void initialize_bucket_pointers(util::string_span input,
                    size_t alphabet_size, size_t bucketsort_depth,
                    util::span<sa_index> sa) {
                const size_t n = sa.size();

                // create bucket pointer array
                bptr<sa_index> = util::make_container<sa_index>(n);

                const std::size_t real_alphabet_size = alphabet_size + 1;
                const std::size_t code_modulo = pow(real_alphabet_size,
                        bucketsort_depth);
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
                    
                    if (current_code != recent_code) {
                        // we passed a border between two buckets
                        current_bucket = current_sa_position;
                    }
                    bptr<sa_index>[sa[current_sa_position]] = current_bucket;
                } while (current_sa_position > 0);
            }

    }; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement

