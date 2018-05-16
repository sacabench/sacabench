/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow {
class saca {
public:
    template<typename sa_index_type>
    static void construct_sa(util::string_span text, size_t alphabet_size,
                             span<sa_index_type> sa) {}
};
} // namespace sacabench::deep_shallow
