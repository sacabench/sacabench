/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>

#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>

namespace sacabench::example1 {

    class example1 {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                // Suppress unused variable warnings:
                (void) text;
                (void) alphabet_size;
                (void) out_sa;

                std::cout << "Running example1" << std::endl;
            }
    }; // class example1

} // namespace sacabench::example1
