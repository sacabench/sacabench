/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include <util/sort/std_sort.hpp>
#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>

namespace sacabench::example2 {

    class example2 {
        public:
            template<typename sa_index>
            static void construct_saca(util::string_span test_input,
                                       size_t alphabet_size,
                                       util::span<sa_index> output) {
                std::vector<std::size_t> data = { 39, 3192, 29, 1923, 29, 0, 19238, 2, 4 };
                sacabench::util::sort::std_sort(data, [](const std::size_t a, const std::size_t b) {
                        return a < b;
                        });
                std::cout << "Running example2: ";
                for (const auto d : data) { std::cout << d << " "; };
                std::cout << std::endl;
            }

    }; // class example2

} // namespace sacabench::example2
