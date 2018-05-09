/*******************************************************************************
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "alphabet.hpp"
#include "string.hpp"
#include "container.hpp"

inline container get_bucket_sizes(const string& input){
    alphabet input_alphabet = alphabet(input);
    apply_effective_alphabet(input, input_alphabet);
    auto bucket_sizes = make_container<int>(input_alphabet.size);
    for(const char c: input){
        ++bucket_sizes[c];
    }
    return bucket_sizes;
}
