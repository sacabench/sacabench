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
extern "C" int32_t divsufsort_par64(const uint8_t* T, int64_t* SA, int64_t n);

namespace sacabench::reference_sacas {
struct div_suf_sort_par {
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "DivSufSort_par_ref";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of DivSufSort with enabled parallelization.";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        size_t n = text.size() - EXTRA_SENTINELS;
        DCHECK_EQ(text[n], 0);
        external_saca<sa_index>(text.slice(0, n),
                                out_sa.slice(EXTRA_SENTINELS, n + EXTRA_SENTINELS),
                                n,
                                ::divsufsort_par,
                                ::divsufsort_par64);
    }
};
} // namespace sacabench::reference_sacas
