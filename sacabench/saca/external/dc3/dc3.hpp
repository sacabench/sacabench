/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>

#include <util/alphabet.hpp>
#include "drittel.h"
#include "../external_saca.hpp"

namespace sacabench::reference_sacas {

class dc3 {
public:
    static constexpr size_t EXTRA_SENTINELS = 3;
    static constexpr char const* NAME = "DC3_ref";
    static constexpr char const* DESCRIPTION =
        "Difference Cover Modulo 3 Reference implementation";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet& alphabet,
                                    util::span<sa_index> out_sa) {
        //alphabet size
        int K = alphabet.max_character_value();
        size_t n = text.size() - EXTRA_SENTINELS;

        auto saca_fn = [K](auto text_ptr, auto sa_ptr, size_t n) {
            ::suffixArray(text_ptr, sa_ptr, n, K);
        };
        external_saca_with_writable_text_one_size_only<sa_index, int, int>(text, out_sa, n,
                                     saca_fn);
    }

}; // class dc3
} // namespace sacabench::reference_sacas
