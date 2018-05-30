/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#pragma once
namespace sacabench::util {
/**\brief Merge two suffix array with the difference cover idea.
 * \tparam T input string
 * \tparam C input characters
 * \tparam I ISA
 * \tparam S SA
 * \param t input text
 * \param sa_0 calculated SA for triplets beginning in i mod 3 = 0
 * \param sa_12 calculated SA for triplets beginning in i mod 3 != 0
 * \param isa_12 calculated ISA for triplets beginning in i mod 3 != 0
 * \param sa memory block for merged SA
 * \param comp function which compares for strings a and b if a is
 *        lexicographically smaller than b
 * \param get_substring function which expects a string t, an index i and an
 *        integer n and returns a substring of t beginning in i where n
 *        equally calculated substrings are concatenated
 *
 * This method merges the suffix arrays s_0, which contains the
 * lexicographical ranks of positions i mod 3 = 0, and s_12, which
 * contains the lexicographical ranks of positions i mod 3 != 0.
 * This method works correct because of the difference cover idea.
 */
template <typename C, typename T, typename I, typename S, typename Compare,
          typename Substring>
void merge_sa_dc(const T& t, const S& sa_0, const S& sa_12, const I& isa_12,
                 S& sa, const Compare comp, const Substring get_substring) {

    DCHECK_MSG(sa.size() == t.size(),
               "sa must be initialised and must have the same length as t.");
    DCHECK_MSG(
        sa.size() == (sa_0.size() + sa_12.size()),
        "the length of sa must be the sum of the length of sa_0 and sa_12");
    DCHECK_MSG(sa_12.size() == isa_12.size(),
               "the length of sa_12 must be equal to isa_12");

    size_t i = 0;
    size_t j = 0;
    size_t counter = 0;

    while (counter < sa.size()) {
        if (i < sa_0.size() && j < sa_12.size()) {
            sacabench::util::span<C> t_0;
            sacabench::util::span<C> t_12;
            if (sa_12[j] % 3 == 1) {
                t_0 = get_substring(t, &t[sa_0[i]], 1);
                t_12 = get_substring(t, &t[sa_12[j]], 1);
            } else {
                t_0 = get_substring(t, &t[sa_0[i]], 2);
                t_12 = get_substring(t, &t[sa_12[j]], 2);
            }

            const bool less_than = comp(t_0, t_12);
            const bool eq = as_equal(comp)(t_0, t_12);
            // NB: This is a closure so that we can evaluate it later, because
            // evaluating it if `eq` is not true causes out-of-bounds errors.
            auto lesser_suf = [&]() {
                return !((2 * (sa_12[j] + t_12.size())) / 3 >=
                         isa_12.size()) && // if index to compare for t_12 is
                                           // out of bounds of isa then sa_0[i]
                                           // is never lexicographically smaller
                                           // than sa_12[j]
                       ((2 * (sa_0[i] + t_0.size())) / 3 >=
                            isa_12
                                .size() || // if index to compare for t_0 is out
                                           // of bounds of isa then sa_0[i] is
                                           // lexicographically smaller
                        isa_12[(2 * (sa_0[i] + t_0.size())) / 3] <
                            isa_12[2 * ((sa_12[j] + t_12.size())) / 3]);
            };
            if (less_than || (eq && lesser_suf())) {
                sa[counter] = sa_0[i++];
            } else {
                sa[counter] = sa_12[j++];
            }
        }

        else if (i >= sa_0.size()) {
            sa[counter] = sa_12[j++];
        }

        else {
            sa[counter] = sa_0[i++];
        }

        ++counter;
    }
}
} // namespace sacabench::util