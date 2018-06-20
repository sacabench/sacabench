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

    class sacak {
    public:
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const* NAME = "SACA-K";
        static constexpr char const* DESCRIPTION =
            "Constant-Space SA-Algorithm based on SAIS";

        template <typename string_type, typename sa_index>
        inline static void induced_sort(const string_type t, util::span<sa_index> sa,
            size_t max_char) {

            // Iterate SA RTL and distribute the LMS suffixes into the S-Buckets

            // Initialize bkt so that the pointers point to the S-Buckets (end of
            // buckets)
            util::container<util::sort::bucket> bkt = util::sort::get_buckets<string_type>(t, max_char, 1);

            for (size_t i = 0; i < bkt.size(); i++) {
                bkt[i].position = bkt[i].position + bkt[i].count - 1;
            }

            // At this state the LMS substrings are supposed to be right at the
            // beginning of SA and all other entries are 0 We distribute them to their
            // respective S-Buckets now

            for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
            {
                if (static_cast<size_t>(sa[i - 1]) != 0 && sa[i - 1] != static_cast<sa_index>(-1)) {
                    size_t c = t[sa[i - 1]]; // The character c itself is already his
                                             // bkt index (-1)
                    sa[bkt[c].position] = sa[i - 1];
                    if (bkt[c].position !=  i - 1) // We need to overwrite all unused slots with 0 but we  dont want to overwrite the slot that was just written to
                        sa[i - 1] = -1;
                    bkt[c].position--;
                }
            }

            // Iterate SA LTR and distribute all the indices into L-Buckets which are
            // lefthand next to already sorted indices

            // Initialize bkt so that the pointers point to the L-Buckets (start of
            // buckets)
            bkt = util::sort::get_buckets<string_type>(t, max_char, 1);

            for (size_t i = 0; i < t.size(); i++) // LTR Iteration of SA
            {

                if (static_cast<size_t>(sa[i]) != 0 && sa[i] != static_cast<sa_index>(-1) &&
                    t[static_cast<size_t>(sa[i]) - 1] >= t[sa[i]]) // check if t[sa[i]-1] is L-Type
                {
                    size_t c = t[static_cast<size_t>(sa[i]) - 1];
                    sa[bkt[c].position] = sa[i] - static_cast<sa_index>(1);
                    bkt[c].position++;
                }
            }

            // Iterate SA RTL and distribute all the indices into S-Buckets which are
            // righthand next to already sorted indices

            // Initialize bkt so that the pointers point to the S-Buckets (end of
            // buckets)
            bkt = util::sort::get_buckets<string_type>(t, max_char, 1);

            for (size_t i = 0; i < bkt.size(); i++) {
                bkt[i].position = bkt[i].position + bkt[i].count - 1;
            }

            for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
            {
                sa_index ind = sa[i - 1];

                if (ind != static_cast<sa_index>(-1) && ind != static_cast<sa_index>(0) && ind != static_cast<sa_index>(t.size() - 1) &&
                    !std::get<0>(util::get_type_ltr_dynamic<string_type>(t, ind - static_cast<sa_index>(1)))) // check if t[ind - 1] is S-Type (inefficient!)
                {

                    size_t c = t[ind - static_cast<sa_index>(1)]; // The character c itself is already his bkt index (-1)
                    sa[bkt[c].position] = static_cast<size_t>(sa[i - 1]) - 1;
                    bkt[c].position--;
                }
            }
        }


        template <typename sa_index>
        inline static void calculate_deep_sa(util::span<sa_index> t,
            util::span<sa_index> sa, size_t max_char) {

            for (size_t i = 0; i < sa.size(); i++) {
                sa[i] = -1;
            }

            size_t lms_amount = util::insert_lms_rtl<util::span<sa_index>, sa_index>(t, sa);

            induced_sort<util::span<sa_index>, sa_index>(t, sa, max_char);

            if (lms_amount <= 2) { // breaking point for special cases so we dont need to do more calculation
                for (size_t i = 0; i < t.size(); i++)
                {
                    t[i] = -1; // finishing up the recursion before returning
                }
                return;
            }

            util::extract_sorted_lms<util::span<sa_index>, sa_index>(t, sa);

            util::ssize name = 0;
            util::ssize previous_LMS = -1;
            for (util::ssize i = 0; i < static_cast<util::ssize>(lms_amount); i++) {
                bool diff = false;

                util::ssize current_LMS = sa[i];
                for (size_t j = 0; j < t.size(); j++) {

                    if (previous_LMS == -1)
                    {
                        diff = true;
                        break;
                    }

                    util::character c_old = t[previous_LMS + j];
                    util::character c_new = t[current_LMS + j];

                    bool type_old = std::get<0>(util::get_type_ltr_dynamic(t, previous_LMS + j));
                    bool type_new = std::get<0>(util::get_type_ltr_dynamic(t, current_LMS + j));

                    if (c_old != c_new || type_new != type_old) {
                        diff = true;
                        break;
                    }
                    else if (j > 0 && (util::is_LMS<util::span<sa_index>, size_t>(t, current_LMS + j) || util::is_LMS<util::span<sa_index>, size_t>(t, previous_LMS + j))) {
                        break;
                    }

                }

                if (diff) {
                    name++;
                    previous_LMS = current_LMS;
                }

                current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2 : (current_LMS - 1) / 2;
                sa[lms_amount + current_LMS] = name - 1;
            }

            size_t null_counter = 0;
            for (size_t i = sa.size() - 1; i >= lms_amount; i--)
            {
                if (sa[i] != static_cast<sa_index>(-1))
                {
                    sa[i + null_counter] = sa[i];
                    sa[i] = -1;
                }
                else
                    null_counter++;
            }

            size_t k_1 = name;

            if (lms_amount > k_1) {

                calculate_deep_sa<sa_index>(sa.slice(sa.size() - lms_amount, sa.size()), sa.slice(0, lms_amount), k_1);
                util::recover_lms_after_recursion<util::span<sa_index>, sa_index>(t, sa);

            }
            else
            {
                for (size_t i = sa.size() - 1; i >= lms_amount; i--)
                {
                    sa[i] = -1;
                }
            }

            induced_sort<util::span<sa_index>, sa_index>(t, sa, max_char);

            for (size_t i = 0; i < t.size(); i++)
            {
                t[i] = -1; // finishing up the recursion before returning
            }
        }

        /*
            Given an effective string t_0 with one sentinel symbol it calculates the
           suffix array sa
        */
        template <typename sa_index>
        inline static void calculate_sa(util::string_span t_0, util::span<sa_index> sa,
            size_t max_char) {

            // Initialize SA so that all items are -1 at the beginning

            for (size_t i = 0; i < sa.size(); i++) {
                sa[i] = -1;
            }

            // First get the alphabet container of t_0
            // The alphabet is given to saca-k normalized and effective, so that it's
            // always in the form of {0, ..., n} That means you only need the n to know
            // the whole effective alphabet

            size_t lms_amount = util::insert_lms_rtl<util::string_span, sa_index>(t_0, sa);
            induced_sort(t_0, sa, max_char);

            if (lms_amount <= 2) {
                // for special corner cases and optimization
                return;
            }

            // After this operation the sorted LMS Strings will be in the first half of the SA
            util::extract_sorted_lms<util::string_span>(t_0, sa);

            // TODO: "Create" t_1 by renaming the sorted LMS Substrings in SA as indices

            // for (size_t i = lms_amount; i < t_0.size(); i++) { sa[i] = -1; }

            util::ssize name = 0;
            util::ssize previous_LMS = -1;
            for (util::ssize i = 0; i < static_cast<util::ssize>(lms_amount); i++) { // max n/2 iterations
                bool diff = false;

                util::ssize current_LMS = sa[i];
                if (sa[i] < static_cast<sa_index>(0)) continue;

                for (size_t j = 0; j < t_0.size(); j++) {

                    if (previous_LMS == -1)
                    {
                        diff = true;
                        break;
                    }

                    util::character c_old = t_0[previous_LMS + j];
                    util::character c_new = t_0[current_LMS + j];

                    bool type_old = std::get<0>(util::get_type_ltr_dynamic(t_0, previous_LMS + j));
                    bool type_new = std::get<0>(util::get_type_ltr_dynamic(t_0, current_LMS + j));

                    if (c_old != c_new || type_new != type_old) {
                        diff = true;
                        break;
                    }
                    else if (j > 0 && (util::is_LMS<util::string_span, size_t>(t_0, current_LMS + j) || util::is_LMS<util::string_span, size_t>(t_0, previous_LMS + j))) { // check if next LMS is reached
                       break; // no diff was found
                    }

                }

                if (diff) { // if diff was found, adjust the name and continue with current LMS as previous
                    name++;
                    previous_LMS = current_LMS;
                }

                current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2 : (current_LMS - 1) / 2;
                sa[lms_amount + current_LMS] = name - 1;
            }

            size_t null_counter = 0;
            for (size_t i = sa.size() - 1; i >= lms_amount; i--)
            {
                if (sa[i] != static_cast<sa_index>(-1))
                {
                    sa[i + null_counter] = sa[i];
                    sa[i] = -1;
                }
                else
                    null_counter++;
            }

            // Allocate t_1 in the last bits of space of SA by slicing it

            // util::span<sa_index>t_1 = sa.slice(sa.size() - lms_amount, sa.size());
            size_t k_1 = name;

                // if new alphabetsize k_1 != |t_1|, we need to call the recursion and
                // calculate SA_1 of t_1 else t_1 is already ordered by SA and we can
                // immediately calculate the full SA of t_0

                if (lms_amount > k_1) {

                    calculate_deep_sa<sa_index>(sa.slice(sa.size() - lms_amount, sa.size()), sa.slice(0, lms_amount), k_1);
                    util::recover_lms_after_recursion<util::string_span, sa_index>(t_0, sa);

                }
                else
                {
                    // Delete the (now) unneccessary, previously calculated names here
                    for (size_t i = sa.size() - 1; i >= lms_amount; i--)
                    {
                        sa[i] = -1;
                    }
                }


                // calculate the first round of SA from the now sorted SA_1

                induced_sort(t_0, sa, max_char);
            }

            template<typename sa_index>
            static void construct_sa(util::string_span text_with_sentinels,
                util::alphabet const& alphabet,
                util::span<sa_index> out_sa) {

                calculate_sa<sa_index>(text_with_sentinels, out_sa, alphabet.size_without_sentinel());
            }

    }; // class sacak
} // namespace sacabench::sacak

/******************************************************************************/
