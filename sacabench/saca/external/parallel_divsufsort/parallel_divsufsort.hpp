/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 * Copyright (C) 2019 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include <util/span.hpp>
#include <cstdint>

extern int32_t pdivsufsort(const uint8_t* T, int32_t* SA, int32_t n);
extern int32_t pdivsufsort(const uint8_t* T, int64_t* SA, int64_t n);

namespace sacabench::reference_sacas {
inline int32_t pdivsufsort32(const uint8_t* T, int32_t* SA, int32_t n) {
    return ::pdivsufsort(T, SA, n);
}
inline int32_t pdivsufsort64(const uint8_t* T, int64_t* SA, int64_t n) {
    return ::pdivsufsort(T, SA, n);
}
struct parallel_div_suf_sort {
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "DivSufSort_PARALLEL_ref";
    static constexpr char const* DESCRIPTION =
        "Parallel Reference implementation of DivSufSort by Shun and Labeit";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        size_t n = text.size() - EXTRA_SENTINELS;
        DCHECK_EQ(text[n], 0);
        external_saca<sa_index>(text,
                                out_sa,
                                n,
                                pdivsufsort32,
                                pdivsufsort64);
    }
};
} // namespace sacabench::reference_sacas
