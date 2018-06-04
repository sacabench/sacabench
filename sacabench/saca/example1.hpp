/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>

#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::example1 {

class example1 {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        // Suppress unused variable warnings:
        (void)text;
        (void)alphabet;
        (void)out_sa;
    }
}; // class example1

} // namespace sacabench::example1
