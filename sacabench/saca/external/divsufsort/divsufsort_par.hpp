/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include <util/span.hpp>
#include <cstdint>

extern "C" int32_t divsufsort_par(const uint8_t* T, int32_t* SA, int32_t n);

namespace sacabench::reference_sacas {
struct div_suf_sort_par {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "DivSufSort_par_ref";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of DivSufSort with enabled parallelization.";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        external_saca<sa_index>(text, out_sa, text.size(), ::divsufsort_par);
    }
};
} // namespace sacabench::reference_sacas
