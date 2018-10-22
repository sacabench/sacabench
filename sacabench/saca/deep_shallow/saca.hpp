/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include "saca_run.hpp"
#include "parallel_saca_run.hpp"

namespace sacabench::deep_shallow {
class serial {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Deep-Shallow";
    static constexpr char const* DESCRIPTION =
        "Deep Shallow SACA by Manzini and Ferragina";

    /// \brief Use Deep Shallow Sorting to construct the suffix array.
    template <typename sa_index_type>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet& alphabet,
                                    span<sa_index_type> sa) {

        // Check if `sa_index_type` is suitable.
        DCHECK(util::assert_text_length<sa_index_type>(text.size(), 1));

        // Construct an object of type `saca_run`, which contains the algorithm.
        // This will construct the suffix array in `sa` using deep-shallow.
        saca_run<sa_index_type> r(text, alphabet.size_without_sentinel(), sa);

        logger::get().flush();
    }
};

class parallel {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Deep-Shallow_par";
    static constexpr char const* DESCRIPTION =
        "Parallel Deep Shallow SACA by Manzini and Ferragina";

    /// \brief Use Deep Shallow Sorting to construct the suffix array.
    template <typename sa_index_type>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet& alphabet,
                                    span<sa_index_type> sa) {

        // Check if `sa_index_type` is suitable.
        DCHECK(util::assert_text_length<sa_index_type>(text.size(), 1));

        // Construct an object of type `saca_run`, which contains the algorithm.
        // This will construct the suffix array in `sa` using deep-shallow.
        parallel_saca_run<sa_index_type> r(text, alphabet.size_without_sentinel(), sa);

        logger::get().flush();
    }
};

} // namespace sacabench::deep_shallow
