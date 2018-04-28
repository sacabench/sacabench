/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 * Copyright (C) 2018 Florian Kurpicz   <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "util/string.hpp"
#include "util/alphabet.hpp"

/**\brief Transforms a text such that it uses the effective alphabet
 * \param input Input text that is going to be transformed
 * \param alphabet_map Alphabet to be used for transformation
 *
 * This method takes the input text and replaces it's symbols according to the
 * specified mapping in the given effective alphabet. Note that the
 * transformation relies on the correctness of the alphabet map.
 */
void apply_effective_alphabet(string& input, const alphabet& alphabet_map) {
    // Map the characters using the new effective alphabet
    for (character& c : input) {
        c = alphabet_map.effective[c];
    }
}

/******************************************************************************/
