/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>

#include <util/string.hpp>
#include <util/container.hpp>

namespace sacabench::example1 {

    class example1 {
        public:
            template<typename sa_index>
            static void construct_saca(util::string_span test_input,
                                       size_t alphabet_size,
                                       util::container<sa_index>& output) {
                std::cout << "Running example1" << std::endl;
            }
    }; // class example1

} // namespace sacabench::example1
