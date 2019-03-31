//
//  gsaca.hpp
//  gsaca
//
//  Created by David Piper on 09.05.18.
//  Copyright © 2018 David Piper. All rights reserved.
//

#pragma once

#include "external_saca.hpp"
#include <util/span.hpp>
#include <util/string.hpp>
#include <limits.h>
#include <stdlib.h>

#include <gsaca.h>

namespace sacabench::reference_sacas {

    class gsaca {

    public:
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const *NAME = "GSACA_ref";
        static constexpr char const *DESCRIPTION =
                "Computes a suffix array with the algorithm gsaca by Uwe Baier.";

        template<typename sa_index>
        inline static void construct_sa(sacabench::util::string_span text_with_sentinels,
                                        util::alphabet const& /* alphabet */,
                                        sacabench::util::span<sa_index> out_sa) {

            size_t n = text_with_sentinels.size();
            external_saca_32bit_only<sa_index>(text_with_sentinels, out_sa, n, ::gsaca);
        }
    }; // class gsaca
} // namespace sacabench::reference_sacas
