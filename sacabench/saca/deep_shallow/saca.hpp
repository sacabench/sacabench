/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>
#include <util/string.hpp>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>

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

        // Check if `sa_index_type` is suitable.
        DCHECK(util::assert_text_length<sa_index_type>(text.size(), 0));

        // Construct an object of type `saca_run`, which contains the algorithm.
        // This will construct the suffix array in `sa` using deep-shallow.
        saca_run<sa_index_type> r(text, alphabet.size_without_sentinel(), sa);
    }
};
} // namespace sacabench::deep_shallow
