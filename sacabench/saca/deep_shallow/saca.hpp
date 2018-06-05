/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

//#include <util/sa_check.hpp> // FIXME: uncomment, if DCHECK is used.
#include <util/alphabet.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include "saca_run.hpp"

namespace sacabench::deep_shallow {
class saca {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;

    /// \brief Use Deep Shallow Sorting to construct the suffix array.
    template <typename sa_index_type>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet& alphabet,
                                    span<sa_index_type> sa) {

        // Construct an object of type `saca_run`, which contains the algorithm.
        // This will construct the suffix array in `sa` using deep-shallow.
        saca_run<sa_index_type> r(text, alphabet.size_without_sentinel(), sa);

        // auto result = util::sa_check(sa, text);
        // DCHECK_MSG(result,
        //            "Das Suffixarray, welches von Deep-Shallow berechnet
        //            wurde, " "war nicht korrekt: " << result);
    }
};
} // namespace sacabench::deep_shallow
