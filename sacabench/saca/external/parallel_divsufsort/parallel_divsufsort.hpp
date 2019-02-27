/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 * Copyright (C) 2019 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>
#include <cstdint>

extern int32_t pdivsufsort(const uint8_t* T, int32_t* SA, int32_t n);
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

        const size_t n = text.size();

        if constexpr (sizeof(sa_index) == 64) {  
            if (n < 2) {
                return;
            }
            pdivsufsort(text.data(), out_sa.data(), n);
        } else if constexpr (sizeof(sa_index) == 32) {
            if (n < 2) {
                return;
            }
            pdivsufsort(text.data(), out_sa.data(), n);
        } else {
            tdc::StatPhase::pause_tracking();
            auto sa_correct_size = util::make_container<int64_t>(n);
            tdc::StatPhase::resume_tracking();

            if (n < 2) {
                return;
            }

            {
                pdivsufsort(text.data(), sa_correct_size.data(), n);
            }

            tdc::StatPhase::pause_tracking();
            for (size_t i = 0; i < n; ++i) {
                out_sa[i] = sa_correct_size[i];
            }
            tdc::StatPhase::resume_tracking();
        }
    }
};
} // namespace sacabench::reference_sacas
