/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include <util/span.hpp>
#include <cstdint>

extern "C" int32_t divsufsort(const uint8_t* T, int32_t* SA, int32_t n);
extern "C" int32_t divsufsort64(const uint8_t* T, int64_t* SA, int64_t n);

namespace sacabench::reference_sacas {
struct div_suf_sort {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "DivSufSort_ref";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of DivSufSort with disabled parallelization.";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        external_saca_32bit_only<sa_index>(text, out_sa, text.size(), ::divsufsort);
    }
};
} // namespace sacabench::reference_sacas
