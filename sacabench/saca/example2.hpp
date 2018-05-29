/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::example2 {

class example2 {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet_info const& alphabet,
                             util::span<sa_index> out_sa) {
        // Suppress unused variable warnings:
        (void)text;
        (void)alphabet;
        (void)out_sa;

        util::container<std::size_t> data = {39, 3192,  29, 1923, 29,
                                             0,  19238, 2,  4};
        sacabench::util::sort::std_sort(
            data,
            [](const std::size_t a, const std::size_t b) { return a < b; });
        for (const auto d : data) {
            std::cout << d << " ";
        };
        std::cout << std::endl;
    }

}; // class example2

} // namespace sacabench::example2
