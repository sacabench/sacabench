/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 * Copyright (C) 2018 Florian Kurpicz   <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "util/string.hpp"

namespace sacabench::util {

class alphabet {
private:
    size_t m_max_character;
    bool m_is_effective;
    std::array<character, std::numeric_limits<character>::max() + 1> effective;

public:
    /**\brief Generates an effective alphabet by searching an input text
     *  for occurring characters
     * \param input Input text that is going to be scanned
     *
     * This constructor takes the input text and creates an effective
     * alphabet with characters {1, ..., n} where n is the number of
     * distinct characters in the input text. If the text relies on all
     * characters (0...max_char), the resulting map will be invalid.
     */
    inline alphabet(string_span input) {
        // Find all existing characters (mark with 1)
        std::fill(effective.begin(), effective.end(), 0);
        for (const character c : input) {
            effective[c] = 1;
        }

        // Create a mapping for the new (effective) alphabet (zero not
        // included)
        //
        // (Maybe use std::inclusive_scan)
        for (std::size_t i = 1; i < effective.size(); ++i) {
            effective[i] += effective[i - 1];
        }

        // The number of effective characters equals the effective value
        // of the greatest character.
        m_max_character = effective[effective.size() - 1];
        m_is_effective = true;
    }

    /**\brief Generates a dummy alphabet which maps each character to
     *  itself
     * \param max_char Maximum value of characters in input text.
     */
    inline alphabet(size_t max_char, bool is_effective)
        : m_max_character(max_char), m_is_effective(is_effective) {
        for (size_t index = 0; index < effective.size(); ++index) {
            effective[index] = index;
        }
    }

    inline size_t size_with_sentinel() const { return m_max_character + 1; }

    inline size_t size_without_sentinel() const { return m_max_character; }

    inline size_t max_character_value() const { return m_max_character; }

    inline bool is_effective() const { return m_is_effective; }

    inline character effective_value(const character original_value) const {
        return effective[original_value];
    }
}; // end class

/**\brief Transforms a text such that it uses the effective alphabet
 * \param input Input text that is going to be transformed
 * \param alphabet_map Alphabet to be used for transformation
 *
 * This method takes the input text and replaces it's symbols according to
 * the specified mapping in the given effective alphabet. Note that the
 * transformation relies on the correctness of the alphabet map.
 */
inline void apply_effective_alphabet(span<character> input,
                                     const alphabet& alphabet_map) {
    // Map the characters using the new effective alphabet
    for (character& c : input) {
        c = alphabet_map.effective_value(c);
    }
}

/**\brief Transforms a text such that it uses the effective alphabet
 * \param input Input text that is going to be transformed
 *
 * This method takes the input text and replaces its symbols by consecutive
 * numbers in the `character` range, holding onto the relative ordering.
 */
inline alphabet apply_effective_alphabet(span<character> input) {
    const alphabet input_alphabet(input);
    apply_effective_alphabet(input, input_alphabet);
    return input_alphabet;
}
} // namespace sacabench::util

/******************************************************************************/
