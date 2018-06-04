/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include <tuple>
#include "util/string.hpp"
#include "util/signed_size_type.hpp"
#include "util/span.hpp"
#include "util/type_extraction.hpp"
#include "util/insert_lms.hpp"

#include "util/container.hpp"
#include "util/string.hpp"
#include "util/alphabet.hpp"
#include "util/bucket_size.hpp"

#pragma once


namespace sacabench::saca {

    void calculate_deep_sa(util::string_span t_0, util::container<size_t> sa) {

        // TODO: almost the same as the usual method, but without the use of bkt array. Needs negative size_t variables!!

    }

    void induced_sort(util::string_span t, util::container<size_t> sa, util::container<util::character> alph) {

    // Iterate SA RTL and distribute the LMS suffixes into the S-Buckets

        // Initialize bkt so that the pointers point to the S-Buckets (end of buckets)
        util::container<size_t> sizes = sacabench::util::get_bucket_sizes(t);
        util::container<size_t> bkt = util::container<size_t>(sizes.size);

        int current_sum = 0;

        for (size_t i = 0; i < bkt.size; i++)
        {
            current_sum += sizes[i];
            bkt[i] = current_sum-1;
        }


        for (size_t i = t.size; i > 0; i--) // RTL Iteration of SA
        {
            if (sa[i - 1] != 0)
            {
                for (size_t j = 0; j < alph.size(); j++) // Iteration of alphabet to find correct bkt entry
                {
                    if (alph[j] == t[sa[i - 1]])
                    {
                        sa[bkt[j]] = sa[i - 1];
                        bkt[j]--;
                    }
                }
            }
        }





    // Iterate SA LTR and distribute all the indices into L-Buckets which are lefthand next to already sorted indices

        // Initialize bkt so that the pointers point to the L-Buckets (start of buckets)
        sizes = sacabench::util::get_bucket_sizes(t);
        bkt = util::container<size_t>(sizes.size);

        current_sum = 0;

        for (size_t i = 0; i < bkt.size; i++)
        {
            bkt[i] = current_sum;
            current_sum += sizes[i];
        }


        for (size_t i = 0; i < t.size; i--) // LTR Iteration of SA
        {
            if (sa[i] != 0 && t[sa[i]-1] >= t[sa[i]]) // check if t[sa[i]-1] is L-Type
            {
                for (size_t j = 0; j < alph.size(); j++) // Iteration of alphabet to find correct bkt entry
                {
                    if (alph[j] == t[sa[i]-1])
                    {
                        sa[bkt[j]] = sa[i]-1;
                        bkt[j]++;
                    }
                }
            }
        }





    // Iterate SA RTL and distribute all the indices into S-Buckets which are righthand next to already sorted indices

        // Initialize bkt so that the pointers point to the S-Buckets (end of buckets)
        sizes = sacabench::util::get_bucket_sizes(t);
        bkt = util::container<size_t>(sizes.size);

        current_sum = 0;

        for (size_t i = 0; i < bkt.size; i++)
        {
            current_sum += sizes[i];
            bkt[i] = current_sum - 1;
        }


        for (size_t i = t.size; i > 0; i--) // RTL Iteration of SA
        {
            if (sa[i - 1] != 0) // TODO: check if t[sa[i-1] + 1] is S-Type
            {
                for (size_t j = 0; j < alph.size(); j++) // Iteration of alphabet to find correct bkt entry
                {
                    if (alph[j] == t[sa[i - 1]+1])
                    {
                        sa[bkt[j]] = sa[i - 1]+1;
                        bkt[j]--;
                    }
                }
            }
        }

    }


    void calculate_sa(util::string_span t_0, util::container<size_t> sa) {

        // Initialize SA so that all items are 0 at the beginning

        for (size_t i = 0; i < t_0.size; i++)
        {
            sa[i] = 0;
        }

        // Calculate Bucketlist bkt for bucket-pointers

        util::container<size_t> sizes = sacabench::util::get_bucket_sizes(t_0);
        util::container<size_t> bkt = util::container<size_t>(sizes.size);

        int current_sum = 0;

        for (size_t i = 0; i < bkt.size; i++)
        {
            bkt[i] = current_sum;
            current_sum += sizes[i];
        }

        // Insert and sort LMS substrings into SA

        // TODO: First get the alphabet container of t_0
        util::alphabet((util::string)t_0);
        util::container<util::character> alph;

        size_t lms_amount = util::insert_lms_rtl(t_0, sa, bkt, alph);
        induced_sort(t_0, sa, alph);

        // Allocate t_1 in the last bits of space of SA

        util::string_span t_1 = sa[sa.size - lms_amount - 1];

        // TODO "Create" t_1 by renaming the sorted LMS Substrings in SA as indices to their own buckets and put them into t_1

        size_t k_1 = util::alphabet(t_1).real_size;

        // if new alphabetsize k_1 != |t_1|, we need to call recursion and calculate SA_1 of t_1
        // else t_1 is already ordered by SA and we can immediately calculate the full SA of t_0

        if (t_1.size != k_1) {

            calculate_deep_sa(t_1, sa);

            //TODO(?): remapping the pointers of SA_1 to pointers of SA

        }

        // calculate the first round of SA from the now sorted SA_1

        size_t sa_pointer = 0;

        for (size_t i = 0; i < t_0.size; i++)
        {
            if (sa[i] != 0)
            {
                sa[sa_pointer] = sa[i];
            }
        }

        induced_sort(t_0, sa, alph);
    }

}

/******************************************************************************/

