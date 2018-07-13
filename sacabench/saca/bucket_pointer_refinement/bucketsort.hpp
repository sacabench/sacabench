/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "util/assertions.hpp"
#include <math.h>

namespace sacabench::bucket_pointer_refinement::sort {

using namespace util;

/**\brief Holds information about a bucket inside the suffix array.
 *
 * A bucket has an attribute `count` for it's size and `position` to store
 * position of the leftmost index inside the bucket. The rightmost index
 * can be derived by `position + count - 1`.
 */
struct bucket {
    std::size_t count = 0;
    std::size_t position = 0;
};

/**
 * A lightweight_bucket does only store the position of the leftmost index
 * inside the bucket. The rightmost index can be derived looking at the
 * successor bucket's position.
 */
using lightweight_bucket = std::size_t;

/**\brief Determines sizes and positions of buckets containing suffixes
 *  which are equal to a given offset.
 * \param input Input text (of length n) whose suffixes are to be sorted.
 *  The input text has to use an effective alphabet with characters
 *  {1, ..., m}.
 * \param max_character_code Maximum possible value of characters in input
 *  text.
 * \param depth The offset which is used to match suffixes into buckets.
 *  Suffixes with equal length-depth-prefix are matched to the same bucket.
 *
 * \return Size and starting position for each bucket in the suffix array.
 */
template <typename span_type>
inline container<bucket> get_buckets(const span_type input,
                                     const std::size_t max_character_code,
                                     const std::size_t depth) {
    DCHECK_GE(depth, 1);

    const std::size_t length = input.size();
    const std::size_t alphabet_size = max_character_code + 1;
    const std::size_t bucket_count = pow(alphabet_size, depth);

    auto buckets = make_container<bucket>(bucket_count);

    /*
     * first step: count bucket sizes
     */

    // calculate code for an (imaginary) 0-th suffix
    std::size_t initial_code = 0;
    for (std::size_t index = 0; index < depth - 1; ++index) {
        initial_code *= (alphabet_size);
        initial_code += input[index];
    }

    // calculate modulo for code computation
    const std::size_t code_modulo = pow(alphabet_size, depth - 1);

    // count occurrences of each possible length-d-substring
    std::size_t code = initial_code;
    for (std::size_t index = 0; index < length - depth + 1; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        code += input[index + depth - 1];
        ++buckets[code].count;
    }

    // same as above, but for substrings containing at least one $
    for (std::size_t index = length - depth + 1; index < length; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        // the index overlaps input bounds, therefore we assume to add
        // a 0.
        ++buckets[code].count;
    }

    /*
     * second step: determine starting positions for buckets
     */

    // calculate positions for all buckets
    for (size_t index = 1; index < bucket_count; ++index) {
        buckets[index].position =
            buckets[index - 1].position + buckets[index - 1].count;
    }

    return buckets;
}

/**\brief Determines positions of buckets containing suffixes
 *  which are equal to a given offset.
 * \param input Input text (of length n) whose suffixes are to be sorted.
 *  The input text has to use an effective alphabet with characters
 *  {1, ..., m}.
 * \param max_character_code Maximum possible value of characters in input
 *  text.
 * \param depth The offset which is used to match suffixes into buckets.
 *  Suffixes with equal length-depth-prefix are matched to the same bucket.
 *
 * \return Starting position for each bucket in the suffix array as well as
 *  a pseudo starting position for one extra bucket at the end of the SA.
 */
template <typename span_type>
inline container<lightweight_bucket>
get_lightweight_buckets(const span_type input,
                        const std::size_t max_character_code,
                        const std::size_t depth) {
    DCHECK_GE(depth, 1);

    const std::size_t length = input.size();
    const std::size_t alphabet_size = max_character_code + 1;
    const std::size_t bucket_count = pow(alphabet_size, depth) + 1;

    auto buckets = make_container<lightweight_bucket>(bucket_count);

    /*
     * first step: count bucket sizes
     */

    // calculate code for an (imaginary) 0-th suffix
    std::size_t initial_code = 0;
    for (std::size_t index = 0; index < depth - 1; ++index) {
        initial_code *= (alphabet_size);
        initial_code += input[index];
    }

    // calculate modulo for code computation
    const std::size_t code_modulo = pow(alphabet_size, depth - 1);

    // count occurrences of each possible length-d-substring
    std::size_t code = initial_code;
    for (std::size_t index = 0; index < length - depth + 1; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        code += input[index + depth - 1];
        ++buckets[code];
    }

    // same as above, but for substrings containing at least one $
    for (std::size_t index = length - depth + 1; index < length; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        // the index overlaps input bounds, therefore we assume to add
        // a 0.
        ++buckets[code];
    }

    /*
     * second step: determine starting positions for buckets
     */

    // calculate positions for all buckets
    size_t next_bucket_start = buckets[0];
    size_t current_bucket_size = 0;
    buckets[0] = 0;
    for (size_t index = 1; index < bucket_count; ++index) {
        current_bucket_size = buckets[index];
        buckets[index] = next_bucket_start;
        next_bucket_start += current_bucket_size;
    }

    return buckets;
}

/**\brief Divides the suffix array into buckets with each bucket containing
 *  suffixes which are equal to a given offset.
 * \param input Input text (of length n) whose suffixes are to be sorted.
 *  The input text has to use an effective alphabet with characters
 *  {1, ..., m}.
 * \param max_character_code Maximum possible value of characters in input
 *  text.
 * \param depth The offset which is used to match suffixes into buckets.
 *  Suffixes with equal length-depth-prefix are matched to the same bucket.
 * \param sa A container (of length n) for the suffix array.
 *
 * Bucketsort sorts all suffixes of the input text into buckets within the
 * suffix array. After a call of the function, the suffix array contains all
 * buckets in sorted ascending order. The suffixes within each bucket are
 * not necessarily sorted.
 *
 * \return Size and starting position for each bucket in the suffix array.
 */
template <typename index_type>
__attribute__((noinline)) container<bucket>
bucketsort_presort(const string_span input,
                   const std::size_t max_character_code,
                   const std::size_t depth, span<index_type> sa) {
    DCHECK_EQ(input.size(), sa.size());
    DCHECK_LE(depth, sa.size());

    auto buckets = get_buckets(input, max_character_code, depth);

    const std::size_t length = input.size();
    // the real alphabet includes $, so it has one more character
    const std::size_t alphabet_size = max_character_code + 1;

    // calculate modulo for code computation
    const std::size_t code_modulo = pow(alphabet_size, depth - 1);

    // calculate code for an (imaginary) 0-th suffix
    std::size_t initial_code = 0;
    for (std::size_t index = 0; index < depth - 1; ++index) {
        initial_code *= (alphabet_size);
        initial_code += input[index];
    }

    std::size_t code = initial_code;

    // insert entries in suffix array
    for (index_type index = 0; index < length - depth + 1; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        code += input[static_cast<size_t>(index) + depth - 1];
        bucket& current_bucket = buckets[code];
        sa[current_bucket.position] = index;
        ++current_bucket.position;
    }

    // same as above, but for substrings containing at least one $
    for (index_type index = length - depth + 1; index < length; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        bucket& current_bucket = buckets[code];
        sa[current_bucket.position] = index;
        ++current_bucket.position;
    }

    // determine leftmost index of each bucket
    for (auto& bucket : buckets) {
        bucket.position -= bucket.count;
    }

    return buckets;
}

/**\brief Divides the suffix array into buckets with each bucket containing
 *  suffixes which are equal to a given offset.
 * \param input Input text (of length n) whose suffixes are to be sorted.
 *  The input text has to use an effective alphabet with characters
 *  {1, ..., m}.
 * \param max_character_code Maximum possible value of characters in input
 *  text.
 * \param depth The offset which is used to match suffixes into buckets.
 *  Suffixes with equal length-depth-prefix are matched to the same bucket.
 * \param sa A container (of length n) for the suffix array.
 * \param bptr A container (of length n) for suffix to bucket mapping
 *
 * Bucketsort sorts all suffixes of the input text into buckets within the
 * suffix array. After a call of the function, the suffix array contains all
 * buckets in sorted ascending order. The suffixes within each bucket are
 * not necessarily sorted.
 * Additionally fills the bptr with determined suffix positions.
 *
 * \return Starting position for each bucket in the suffix array and
 *  starting position of a pseudo bucket at the end of the SA.
 */
template <typename index_type>
__attribute__((noinline)) container<lightweight_bucket>
bucketsort_presort_lightweight(const string_span input,
                               const std::size_t max_character_code,
                               const std::size_t depth, span<index_type> sa,
                               container<index_type>& bptr) {
    DCHECK_EQ(input.size(), sa.size());
    DCHECK_LE(depth, sa.size());

    auto buckets = get_lightweight_buckets(input, max_character_code, depth);
    auto buckets_tmp = buckets.make_copy();

    const std::size_t length = input.size();
    // the real alphabet includes $, so it has one more character
    const std::size_t alphabet_size = max_character_code + 1;

    // calculate modulo for code computation
    const std::size_t code_modulo = pow(alphabet_size, depth - 1);

    // calculate code for an (imaginary) 0-th suffix
    std::size_t initial_code = 0;
    for (std::size_t index = 0; index < depth - 1; ++index) {
        initial_code *= (alphabet_size);
        initial_code += input[index];
    }

    std::size_t code = initial_code;

    // insert entries in suffix array
    for (size_t index = 0; index < length - depth + 1; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        code += input[index + depth - 1];
        if ((index + 2 >= length) || (input[index] < input[index + 1] &&
                                      input[index] <= input[index + 2])) {
            // DCHECK_LT(index + 2, length);
            sa[--buckets_tmp[code + 1]] = index;
        }
        bptr[index] = buckets[code + 1] - 1;
        //++buckets_tmp[code];
    }

    // same as above, but for substrings containing at least one $
    for (size_t index = length - depth + 1; index < length; ++index) {
        // induce code for nth suffix from (n-1)th suffix
        code %= code_modulo;
        code *= alphabet_size;
        sa[--buckets_tmp[code + 1]] = index;
        bptr[index] = buckets[code + 1] - 1;
        //++buckets_tmp[code];
    }

    return buckets;
}

} // namespace sacabench::util::sort
