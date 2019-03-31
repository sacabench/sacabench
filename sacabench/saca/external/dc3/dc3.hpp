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
    inline static void construct_sa(util::string_span text_with_sentinels,
                                    const util::alphabet& alphabet,
                                    util::span<sa_index> out_sa) {

        tdc::StatPhase::pause_tracking();
        {
            // TODO: This might still measure the allocation and copy time,
            // just not its memory usage

            //length of text with extra sentinals
            size_t n = text_with_sentinels.size();
            if (n <= 4) {
                tdc::StatPhase::resume_tracking();
                return;
            }

            // adjust out_sa for extra sentinels
            out_sa = out_sa.slice(EXTRA_SENTINELS);

            //creates arrays of type int for input text and out_sa
            auto s = std::make_unique<int[]>(n);
            std::unique_ptr<int[]> optional_sa_alloc;
            int* SA = nullptr;
            if constexpr (INDEX_BITS<sa_index> == INDEX_BITS<int>) {
                SA = (int*) out_sa.data();
            } else {
                optional_sa_alloc = std::make_unique<int[]>(n);
                SA = optional_sa_alloc.get();
            }

            //cast input text in ints
            for (size_t index = 0; index < n; index++) {
                s[index] = static_cast<int>(text_with_sentinels[index]);
            }

            //alphabet size
            int K = alphabet.max_character_value();

            {
                tdc::StatPhase::resume_tracking();

                //run algorithm with input text (incl. 3 Sentinals),
                //empty SuffixArray, length of text without sentinals
                //and alphabet size for radix sort
                ::suffixArray(s.get(), SA, n - EXTRA_SENTINELS, K);

                tdc::StatPhase::pause_tracking();
            }

            if constexpr (INDEX_BITS<sa_index> != INDEX_BITS<int>) {
                //copy SA into correct positions of out_sa
                for (size_t index = 0; index < (n - EXTRA_SENTINELS); index++) {
                    out_sa[index] = static_cast<sa_index>(SA[index]);
                }
            } else {
                // already in out_sa
            }
        }
        tdc::StatPhase::resume_tracking();
    }

}; // class dc3
} // namespace sacabench::reference_sacas
