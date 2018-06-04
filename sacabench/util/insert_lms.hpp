/*******************************************************************************
* util/type_extraction.hpp
*
* Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
*
* All rights reserved. Published under the BSD-3 license in the LICENSE file.
******************************************************************************/
#include <array>
#include "string.hpp"
#include "util/type_extraction.hpp"
#include <tuple>

#pragma once


namespace sacabench::util {


    /* Inserts LMS Substrings LTR into the Suffix-Array (L-Buckets)
        returns the amount of LMS substrings indices found
    */
    size_t insert_lms_ltr(string_span t_0, container<size_t> sa, span<size_t> bucket_array, span<character> alph) {

        std::tuple<bool, size_t> last_type = get_type_ltr_dynamic(t_0[0], 0);
        size_t amount = 0;

        // Iterate whole string LTR and compare types of symbols with each other

        for (size_t i = std::get<1>(last_type); i < t_0.size(); i++)
        {
            std::tuple<bool, size_t> current_type = get_type_ltr_dynamic(t_0[0], i);

            // Check if last type was L and current one is S, then this one is LMS

            if (std::get<0>(last_type) && !std::get<0>(current_type))
            {
                // Search for the right index of the bucket pointer through the place of the character in the alphabet

                for (size_t j = 0; j < alph.size(); j++)
                {
                    if (alph[j] == t_0[i])
                    {
                        // Insert LMS Index into SA and increase respective Bucket-Array Entry by 1

                        sa[bucket_array[j]] = i;
                        bucket_array[j]++;
                        amount++;
                    }
                }

            }

            i += std::get<1>(current_type) - 1;
            last_type = current_type;
        }

        return amount;

    }

    /* Inserts LMS Substrings RTL into the Suffix-Array (S-Buckets)
    returns the amount of LMS substrings indices found
    */
    size_t insert_lms_rtl(string_span t_0, container<size_t> sa, span<size_t> bucket_array, span<character> alph) {

        bool last_type = true;
        size_t amount = 0;

        // Iterate whole string RTL and compare types of symbols with each other

        for (size_t i = t_0.size-1; i < 0; i--) //shift by -1 when you use i
        {
            bool current_type = get_type_rtl_dynamic(t_0, i - 1, last_type);

            // Check if last type was S and current one is L, then last one was LMS

            if (!last_type && current_type)
            {
                // Search for the right index of the bucket pointer through the place of the character in the alphabet

                for (size_t j = 0; j < alph.size(); j++)
                {
                    if (alph[j] == t_0[i-1])
                    {
                        // Insert LMS Index of last_type (i, NOT i-1) into SA and decrease respective Bucket-Array Entry by 1

                        sa[bucket_array[j]] = i;
                        bucket_array[j]--;
                        amount++;
                    }
                }

            }

            last_type = current_type;
        }

        return amount;

    }
}

/******************************************************************************/
