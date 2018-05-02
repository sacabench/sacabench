/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "util/assertions.hpp"
#include <math.h>

namespace sacabench::util::sort {

    /**\brief Holds information about a bucket inside the suffix array.
     *
     * A bucket has an attribute `count` for it's size and `position` to store
     * position of the leftmost index inside the bucket. The rightmost index
     * can be derived by `position + count - 1`.
     */
    template <typename index_type>
        struct bucket {
            std::size_t count = 0;
            index_type position = 0;
        };

    /**\brief Divides the suffix array into buckets with each bucket containing
     *  suffixes which are equal to a given offset.
     * \param input Input text (of length n) whose suffixes are to be sorted.
     *  The input text has to use an effective alphabet with characters
     *  {1, ..., m}. 
     * \param alphabet_size The size of the alphabet which is used by the input
     *  text.
     * \param depth The offset which is used to match suffixes into buckets.
     *  Suffixes with equal length-depth-prefix are matched to the same bucket.
     * \param sa A container (of length n) for the suffix array.
     *
     * Bucketsort sorts all suffixes of the input text into buckets within the
     * suffix array. After a call of the function, the suffix array contains all
     * buckets in sorted ascending order. The suffixes within each bucket are
     * not necessarily sorted.
     */
    template <typename index_type>
        void bucketsort(const string& input, const std::size_t alphabet_size,
                const std::size_t depth, container<index_type>& sa) {
            DCHECK_EQ(input.size(), sa.size());
            DCHECK_LE(depth, sa.size());

            const std::size_t length = input.size();
            // the real alphabet includes $, so it has one more character
            const std::size_t real_alphabet_size = alphabet_size + 1;
            const std::size_t bucket_count = pow(real_alphabet_size, depth);
            container<bucket<index_type>> buckets =
                make_container<bucket<index_type>>(bucket_count);

            // calculate code for an (imaginary) 0-th suffix
            std::size_t initial_code = 0;
            for (index_type index = 0; index < depth - 1; ++index) {
                initial_code *= (real_alphabet_size);
                initial_code += input[index];
            }

            // calculate modulo for code computation
            const std::size_t code_modulo = pow(real_alphabet_size, depth - 1);

            // count occurrences of each possible length-d-substring
            std::size_t code = initial_code;
            for (index_type index = 0; index < length - depth + 1; ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                code += input[index + depth - 1];
                ++buckets[code].count;
            }

            // same as above, but for substrings containing at least one $
            for (index_type index = length - depth + 1; index < length;
                    ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                // the index overlaps input bounds, therefore we assume to add
                // a 0.
                ++buckets[code].count;
            }

            // calculate positions for all buckets
            for (size_t index = 1; index < bucket_count; ++index) {
                buckets[index].position =
                    buckets[index - 1].position + buckets[index-1].count;
            }

            // insert entries in suffix array
            code = initial_code;
            for (index_type index = 0; index < length - depth + 1; ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                code += input[index + depth - 1];
                bucket<index_type>& current_bucket = buckets[code];
                sa[current_bucket.position] = index;
                ++current_bucket.position;
            }

            // same as above, but for substrings containing at least one $
            for (index_type index = length - depth + 1; index < length;
                    ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                bucket<index_type>& current_bucket = buckets[code];
                sa[current_bucket.position] = index;
                ++current_bucket.position;
            }
        }

}
