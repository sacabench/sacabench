/*******************************************************************************
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include "../util/insert_lms.hpp"
#include "../util/signed_size_type.hpp"
#include "../util/span.hpp"
#include "../util/string.hpp"
#include "../util/type_extraction.hpp"
#include <array>
#include <tuple>

#include "../util/alphabet.hpp"
#include "../util/container.hpp"
#include "../util/sort/bucketsort.hpp"
#include "../util/string.hpp"

namespace sacabench::sacak {

inline void calculate_deep_sa(util::span<size_t>& t_0,
                              util::container<size_t>& sa, size_t max_char) {

    // TODO: almost the same as the usual method, but without the use of bkt
    // array. Needs negative size_t variables!!
}

template <typename sa_index>
inline void induced_sort(util::string_span t, util::span<sa_index> sa,
                         size_t max_char) {

    // Iterate SA RTL and distribute the LMS suffixes into the S-Buckets

    // Initialize bkt so that the pointers point to the S-Buckets (end of
    // buckets)
    util::container<util::sort::bucket> bkt =
        util::sort::get_buckets(t, max_char, 1);

    for (size_t i = 0; i < bkt.size(); i++) {
        bkt[i].position = bkt[i].position + bkt[i].count - 1;
    }

    // At this state the LMS substrings are supposed to be right at the
    // beginning of SA and all other entries are 0 We distribute them to their
    // respective S-Buckets now

    for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
    {
        if (sa[i - 1] != 0 && sa[i - 1] != -1) {
            size_t c = t[sa[i - 1]]; // The character c itself is already his
                                     // bkt index (-1)
            sa[bkt[c].position] = sa[i - 1];
            if (bkt[c].position !=
                i - 1) // We need to overwrite all unused slots with 0 but we
                       // dont want to overwrite the slot that was just written
                       // to
                sa[i - 1] = -1;
            bkt[c].position--;
        }
    }

    // Iterate SA LTR and distribute all the indices into L-Buckets which are
    // lefthand next to already sorted indices

    // Initialize bkt so that the pointers point to the L-Buckets (start of
    // buckets)

    bkt = util::sort::get_buckets(t, max_char, 1);

    for (size_t i = 0; i < t.size(); i++) // LTR Iteration of SA
    {

        if (sa[i] != 0 && sa[i] != -1 &&
            t[sa[i] - 1] >= t[sa[i]]) // check if t[sa[i]-1] is L-Type
        {
            size_t c = t[sa[i] - 1];
            sa[bkt[c].position] = sa[i] - 1;
            bkt[c].position++;
        }
    }

    // Iterate SA RTL and distribute all the indices into S-Buckets which are
    // righthand next to already sorted indices

    // Initialize bkt so that the pointers point to the S-Buckets (end of
    // buckets)
    bkt = util::sort::get_buckets(t, max_char, 1);

    for (size_t i = 0; i < bkt.size(); i++) {
        bkt[i].position = bkt[i].position + bkt[i].count - 1;
    }

    for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
    {
        size_t ind = sa[i - 1];

        if (ind != -1 && ind != 0 && ind != t.size() - 1 &&
            !std::get<0>(util::get_type_ltr_dynamic(
                t, ind - 1))) // check if t[ind - 1] is S-Type (inefficient!)
        {

            size_t c =
                t[ind -
                  1]; // The character c itself is already his bkt index (-1)
            sa[bkt[c].position] = sa[i - 1] - 1;
            bkt[c].position--;
        }
    }
}

/*
    Given an effective string t_0 with one sentinel symbol it calculates the
   suffix array sa
*/
template <typename sa_index>
inline void calculate_sa(util::string_span t_0, util::span<sa_index> sa,
                         size_t max_char) {

    // Initialize SA so that all items are -1 at the beginning

    for (size_t i = 0; i < t_0.size(); i++) {
        sa[i] = -1;
    }

    // First get the alphabet container of t_0
    // The alphabet is given to saca-k normalized and effective, so that it's
    // always in the form of {0, ..., n} That means you only need the n to know
    // the whole effective alphabet

    size_t lms_amount = util::insert_lms_rtl<sa_index>(t_0, sa);
    induced_sort(t_0, sa, max_char);

    // Allocate t_1 in the last bits of space of SA by slicing it

    util::span<size_t> t_1 = sa.slice(sa.size() - lms_amount - 1, sa.size());

    // After this operation the sorted LMS Strings will be in the first half of
    // the SA
    util::extract_sorted_lms(t_0, sa);

    // TODO: "Create" t_1 by renaming the sorted LMS Substrings in SA as indices
    // to their own buckets and put them into t_1 How exactly?!

    size_t k_1 = 0;

    // if new alphabetsize k_1 != |t_1|, we need to call the recursion and
    // calculate SA_1 of t_1 else t_1 is already ordered by SA and we can
    // immediately calculate the full SA of t_0

    if (t_1.size() != k_1) {

        // util::container<size_t> sa_sliced = util::make_container(sa.slice(0,
        // t_1.size())); calculate_deep_sa(t_1, sa_sliced, max_char);

        // TODO(?): remapping the pointers of SA_1 to pointers of SA
    }

    // calculate the first round of SA from the now sorted SA_1

    induced_sort(t_0, sa, max_char);
}

struct sacak {
    static constexpr size_t EXTRA_SENTINELS = 0;

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        calculate_sa<sa_index>(text, out_sa, alphabet.max_character_value());
    }
}; // struct sacak

} // namespace sacabench::sacak

/******************************************************************************/
