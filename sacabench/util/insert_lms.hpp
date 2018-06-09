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


    /*
        Tries to find entry j in SA after index start  
    */
    bool entry_comes_after(container<size_t> &sa, size_t start, size_t j)
    {
        for (size_t i = start+1; i < sa.size(); i++)
        {
            if (sa[i] == j)
                return true;
        }

        return false;
    }

    /* Inserts LMS Substrings into the beginning of the Suffix-Array by iterating LTR
        returns the amount of LMS substrings indices found.
    */
    size_t insert_lms_ltr(string_span t_0, container<size_t> &sa) {

        std::tuple<bool, size_t> last_type = get_type_ltr_dynamic(t_0, 0);
        size_t amount = 0;
        size_t sa_pointer = 0;

        // Iterate whole string LTR and compare types of symbols with each other

        for (size_t i = std::get<1>(last_type); i < t_0.size(); i++)
        {
            std::tuple<bool, size_t> current_type = get_type_ltr_dynamic(t_0, i);

            // Check if last type was L and current one is S, then this one is LMS

            if (std::get<0>(last_type) && !std::get<0>(current_type))
            {
                // The index of the bucket array for the character is already given by the character itself

                sa[sa_pointer] = i;
                sa_pointer++;

            }

            i += std::get<1>(current_type) - 1;
            last_type = current_type;
        }

        return amount;

    }

    /* Inserts LMS Substrings into the beginning of the Suffix-Array by iterating RTL
    returns the amount of LMS substrings indices found
    */
    size_t insert_lms_rtl(string_span t_0, container<size_t> &sa) {

        bool last_type = true;
        size_t amount = 1;
        size_t sa_pointer = 1;
        
        sa[0] = t_0.size()-1;

        // Iterate whole string RTL and compare types of symbols with each other

        for (size_t i = t_0.size()-1; i > 0; i--) //shift by -1 when you use i
        {
            bool current_type = get_type_rtl_dynamic(t_0, i - 1, last_type);

            // Check if last type was S and current one is L, then last one was LMS

            if (!last_type && current_type)
            {
                sa[sa_pointer] = i;
                sa_pointer++;

                // Now, for keeping the LMS Substrings in order (which is important for induced sort) we shift our current entry left 
                // as long as the char it represents is smaller than the char of the entry on his left

                size_t j = sa_pointer - 1;
                while (j > 0 && t_0[sa[j]] < t_0[sa[j - 1]])
                {
                    sa[j] = sa[j - 1];
                    sa[j - 1] = i;
                    j--;
                }
            }

            last_type = current_type;
        }

        return amount;

    }

    /**
        Call this function to get the sorted LMS-Strings into the beginning of your SA after you have induced sorted your SA.
    */
    size_t extract_sorted_lms(string_span t, container<size_t> &sa)
    {
        size_t null_counter = 0;
        size_t amount = 0;

        for (size_t i = 0; i < sa.size(); i++)
        {

            if (sa[i] != 0 && sa[i] != -1 &&
                t[sa[i]] < t[sa[i] - 1] &&                               // character before is L-Type
                    (t[sa[i]] == util::SENTINEL || 
                        (!std::get<0>(get_type_ltr_dynamic(t, sa[i]))))) // character itself is S-Type (inefficient!)
            {
                size_t lms_entry = sa[i];
                sa[i] = -1;
                sa[i - null_counter] = lms_entry;
                amount++;

            }
            else
            {
                null_counter++;
                sa[i] = -1;
            }
        }

        return amount;
    }
}

/******************************************************************************/
