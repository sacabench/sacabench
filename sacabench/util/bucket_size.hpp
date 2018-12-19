/*******************************************************************************
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "alphabet.hpp"
#include "container.hpp"
#include "string.hpp"
#include "alphabet.hpp"

namespace sacabench::util {
/// \brief Returns a container with bucket sizes in size of the
///        effective alphabet of the input string.
inline container<size_t> get_bucket_sizes(const string_span input, util::alphabet const& alpha) {
    container<size_t> bucket_sizes =
        make_container<size_t>(alpha.size_with_sentinel());
    for (const character c : input) {
        ++bucket_sizes[c];
    }
    return bucket_sizes;
}
} // namespace sacabench::util
