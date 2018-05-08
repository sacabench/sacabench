/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "util/sort.hpp"

namespace sacabench::example2 {

    class example2 {
        public:
            static void run_example() {
                std::vector<std::size_t> data = { 39, 3192, 29, 1923, 29, 0, 19238, 2, 4 };
                sacabench::util::sort(data, [](const std::size_t a, const std::size_t b) {
                        return a < b;
                        });
                std::cout << "Running example2: ";
                for (const auto d : data) { std::cout << d << " "; };
                std::cout << std::endl;
            }

    }; // class example2

} // namespace sacabench::example2
