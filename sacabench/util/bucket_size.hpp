/*******************************************************************************
 * sacabench/util/bucket_size.hpp
 * 
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "container.hpp"
#include "string.hpp"
#include "alphabet.hpp"

namespace sacabench::util {
/*
* Returns a container with bucket sizes in size of the effective alphabet of the input string
*/
    inline container<int> get_bucket_sizes(const string &input) {
        const alphabet input_alphabet = alphabet(input);
        string temp_input = input;
        apply_effective_alphabet(temp_input, input_alphabet);
        container<int> bucket_sizes = make_container<int>(input_alphabet.size);
        for (const char c: input) {
            ++bucket_sizes[c];
        }
        return bucket_sizes;
    }
}