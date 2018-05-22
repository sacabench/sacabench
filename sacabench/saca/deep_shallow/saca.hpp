/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/sa_check.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include "saca_run.hpp"

namespace sacabench::deep_shallow {
class saca {
public:
    template <typename sa_index_type>
    static void construct_sa(util::string_span text, size_t alphabet_size,
                             span<sa_index_type> sa) {
        saca_run<sa_index_type> r(text, alphabet_size, sa);
        DCHECK_MSG(util::sa_check(sa, text),
                   "Das Suffixarray, welches von Deep-Shallow berechnet wurde, "
                   "war nicht korrekt.");
    }
};
} // namespace sacabench::deep_shallow
