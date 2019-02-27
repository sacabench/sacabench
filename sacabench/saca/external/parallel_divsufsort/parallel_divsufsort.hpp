/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include <util/span.hpp>
#include <cstdint>

extern int32_t pdivsufsort(const uint8_t* T, int64_t* SA, int64_t n);

namespace sacabench::reference_sacas {
struct parallel_div_suf_sort {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "DivSufSort_PARALLEL_ref";
    static constexpr char const* DESCRIPTION =
        "Parallel Reference implementation of DivSufSort by Shun and Labeit";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        sadslike<sa_index, int64_t>(text, out_sa, text.size(), pdivsufsort);
    }
};
} // namespace sacabench::reference_sacas
