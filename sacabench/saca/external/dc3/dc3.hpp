/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include <cstdint>

namespace sacabench::reference_sacas {

    class dc3 {
    public:
        static constexpr size_t EXTRA_SENTINELS = 3;
        static constexpr char const* NAME = "DC3";
        static constexpr char const* DESCRIPTION = "Difference Cover Modulo 3 SACA";

        template <typename sa_index>
        inline static void construct_sa(util::string_span text,
                                        const util::alphabet&,
                                        util::span<sa_index> out_sa) {

            // void suffixArray(int* s, int* SA, int n, int K)
            // external_saca<sa_index>(text, out_sa, text.size(), ::suffixArray);
        }

    }; // class dc3
} // namespace sacabench::reference_sacas
