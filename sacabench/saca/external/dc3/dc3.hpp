/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../../../../external/reference_impls/dc3/drittel.C"
#include "../external_saca.hpp"
#include <cstdint>
#include <util/alphabet.hpp>

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
        
        //length of text with extra sentinals
        int n = text_with_sentinels.size();
        if (n <= 4) {
            return;
        }

        //creates arrays of type int for input text and out_sa 
        auto s = std::make_unique<int[]>(n);
        auto SA = std::make_unique<int[]>(n);

        //cast input text in ints
        for (size_t index = 0; index < n; index++) {
            s[index] = static_cast<int>(text_with_sentinels[index]);
        }
        
        //alphabet size
        int K = alphabet.max_character_value();

        tdc::StatPhase::resume_tracking();

        //run algorithm with input text (incl. 3 Sentinals),
        //empty SuffixArray, length of text without sentinals
        //and alphabet size for radix sort
        ::suffixArray(s.get(), SA.get(), n - 3, K);

        tdc::StatPhase::pause_tracking();

        //copy SA into correct positions of out_sa
        for (size_t index = 3; index < n; index++) {
            out_sa[index] = static_cast<sa_index>(SA[index - 3]);
        }

        tdc::StatPhase::resume_tracking();
    }

}; // class dc3
} // namespace sacabench::reference_sacas
