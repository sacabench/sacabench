/*******************************************************************************
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
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
inline container<size_t> get_bucket_sizes(const string_span input) {
    string temp_input = make_string(input);
    const alphabet input_alphabet = apply_effective_alphabet(temp_input);
    container<size_t> bucket_sizes =
        make_container<size_t>(input_alphabet.size + 1);
    for (const character c : temp_input) {
        ++bucket_sizes[c];
    }
    return bucket_sizes;
}
} // namespace sacabench::util
