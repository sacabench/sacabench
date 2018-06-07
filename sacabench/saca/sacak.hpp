/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include <tuple>
#include "../util/string.hpp"
#include "../util/signed_size_type.hpp"
#include "../util/span.hpp"
#include "../util/type_extraction.hpp"
#include "../util/insert_lms.hpp"

#include "../util/container.hpp"
#include "../util/string.hpp"
#include "../util/alphabet.hpp"
#include "../util/sort/bucketsort.hpp"

#pragma once


namespace sacabench::saca {

    template<typename my_character_type>

    void calculate_deep_sa(util::span<my_character_type> t_0, util::span<size_t> sa) {

        // TODO: almost the same as the usual method, but without the use of bkt array. Needs negative size_t variables!!

    }

    template<typename my_character_type>

    void induced_sort(util::string_span t, util::container<size_t> sa, size_t max_char) {

    // Iterate SA RTL and distribute the LMS suffixes into the S-Buckets

        // Initialize bkt so that the pointers point to the S-Buckets (end of buckets)
        util::container<util::sort::bucket> bkt = util::sort::get_buckets(t, max_char, 1);

        for (size_t i = 0; i < bkt.size(); i++)
        {
            bkt[i].position = bkt[i].position + bkt[i].count - 1;
        }

        // At this state the LMS substrings are supposed to be right at the beginning of SA and all other entries are 0
        // We iterate through SA until we find a 0 (because there cant be a LMS substring at position 0 in t_0)

        for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
        {
            if (sa[i - 1] != 0)
            {  
                size_t c = t[sa[i - 1]]; // The character c itself is already his bkt index (-1)
                sa[bkt[c-1].position] = sa[i - 1];
                bkt[c].position--;
                sa[i - 1] = 0;
            }
        }





    // Iterate SA LTR and distribute all the indices into L-Buckets which are lefthand next to already sorted indices

        // Initialize bkt so that the pointers point to the L-Buckets (start of buckets)

        bkt = util::sort::get_buckets(t, max_char, 1);


        for (size_t i = 0; i < t.size(); i--) // LTR Iteration of SA
        {
            if (sa[i] != 0 && t[sa[i]-1] >= t[sa[i]]) // check if t[sa[i]-1] is L-Type
            {
                size_t c = t[sa[i]-1]; // The character c itself is already his bkt index (-1)
                sa[bkt[c].position] = sa[i]-1;
                bkt[c].position++;
            }
        }





    // Iterate SA RTL and distribute all the indices into S-Buckets which are righthand next to already sorted indices
        
        // Initialize bkt so that the pointers point to the S-Buckets (end of buckets)
        bkt = util::sort::get_buckets(t, max_char, 1);

        for (size_t i = 0; i < bkt.size(); i++)
        {
            bkt[i].position = bkt[i].position + bkt[i].count - 1;
        }


        for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
        {
            size_t ind = sa[i - 1];

            if (ind != 0 && 
                ind != t.size()-1 && 
                (t[ind] < t[ind+1] || (t[ind] == t[ind+1] && t[ind] == t[sa[i]]))) // check if t[sa[i-1] + 1] is S-Type
            {
                size_t c = t[sa[i - 1]]; // The character c itself is already his bkt index (-1)
                sa[bkt[c - 1].position] = sa[i - 1];
                bkt[c].position--;
            }
        }

    }
    
    template<typename my_character_type>

    void calculate_sa(my_character_type t_0, util::container<size_t> sa) {

        // Initialize SA so that all items are 0 at the beginning

        for (size_t i = 0; i < t_0.size(); i++)
        {
            sa[i] = 0;
        }

        // Calculate Bucketlist bkt for L-bucket-pointers

        size_t max_char = util::alphabet(t_0).size_without_sentinel();

        util::container<util::sort::bucket> bkt = util::sort::get_buckets(t_0, max_char, 1);

        // Insert and sort LMS substrings into SA

        // First get the alphabet container of t_0
        // The alphabet is given to saca-k normalized and effective, so that it's always in the form of {0, ..., n}
        // That means you only need the n to know the whole effective alphabet

        size_t lms_amount = util::insert_lms_rtl(t_0, sa);
        induced_sort(t_0, sa, max_char);

        // Allocate t_1 in the last bits of space of SA by slicing it

        util::span<my_character_type> t_1 = sa.slice(sa.size() - lms_amount - 1, sa.size());

        // After this operation the sorted LMS Strings will be in the first half of the SA
        util::extract_sorted_lms(t_0, sa);

        // TODO: "Create" t_1 by renaming the sorted LMS Substrings in SA as indices to their own buckets and put them into t_1
        // How exactly?!

        size_t k_1;

        // if new alphabetsize k_1 != |t_1|, we need to call recursion and calculate SA_1 of t_1
        // else t_1 is already ordered by SA and we can immediately calculate the full SA of t_0

        if (t_1.size() != k_1) {

            calculate_deep_sa(t_1, sa.slice(0, t_1.size));

            //TODO(?): remapping the pointers of SA_1 to pointers of SA

        }

        // calculate the first round of SA from the now sorted SA_1

        induced_sort(t_0, sa, max_char);
    }

}

/******************************************************************************/

