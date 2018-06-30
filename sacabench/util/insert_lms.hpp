/*******************************************************************************
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include "string.hpp"
#include "util/type_extraction.hpp"
#include <array>
#include <tuple>
#include "sort/bucketsort.hpp"

#pragma once

namespace sacabench::util {

/*
    Tries to find entry j in SA after index start
*/
template <typename sa_index>
inline bool entry_comes_after(span<sa_index> sa, size_t start, size_t j) {
    for (size_t i = start + 1; i < sa.size(); i++) {
        if (sa[i] == j)
            return true;
    }

    return false;
}



/* Inserts LMS Substrings into the beginning of the Suffix-Array buckets by iterating
RTL returns the amount of LMS substrings indices found
*/
template <typename string_type, typename sa_index>
inline static size_t insert_lms_ltr(const string_type t, span<sa_index> sa, size_t max_char) {

    container<sort::bucket> bkt = util::sort::get_buckets<string_type>(t, max_char, 1); //Reverse the bucket array because we are searching for S-Types
    for (size_t i = 0; i < bkt.size(); i++) {
        bkt[i].position = bkt[i].position + bkt[i].count - 1;
    }

    bool last_type = false; 
    size_t amount = 0;

    DCHECK_GE(sa.size(), 1);

    // Iterate whole string LTR and compare types of symbols with each othernt

    for (size_t i = 0; i < t.size(); i++)
    {
        std::tuple<bool, size_t> current_type = get_type_ltr_dynamic(t, i);
        size_t c = t[i];

        // Check if current type (i) was S and last one (i-1) is L, then current one (i) was LMS

        if (last_type && !std::get<0>(current_type)) {
            sa[bkt[c].position] = i;
            bkt[c].position--;
            amount++;
            

        }

        last_type = std::get<0>(current_type);
        i += std::get<1>(current_type) - 1;
    }

    return amount;
}

template <typename string_type, typename sa_index>
inline bool is_LMS(const string_type& t, sa_index index) {

    if (index <= 0 || t.size() <= index || t[index] >= t[index - 1] || std::get<0>(util::get_type_ltr_dynamic(t,index)))
    {
        return false;
    }

    return true;

}

/**
    Call this function to get the sorted LMS-Strings into the beginning of your
   SA after you have induced sorted your SA.
*/
template <typename string_type, typename sa_index>
inline size_t extract_sorted_lms(string_type &t, span<sa_index> sa) {
    size_t null_counter = 0;
    size_t amount = 0;

    for (size_t i = 0; i < sa.size(); i++) {

        if (static_cast<size_t>(sa[i]) != 0 && sa[i] != static_cast<sa_index>(-1) &&
            t[sa[i]] < t[static_cast<size_t>(sa[i]) - 1] && // character before is L-Type
            (static_cast<size_t>(t[sa[i]]) == util::SENTINEL ||
             (!std::get<0>(get_type_ltr_dynamic(t, sa[i]))))) // character itself is S-Type (inefficient!)
        {
            size_t lms_entry = sa[i];
            sa[i] = -1;
            sa[i - null_counter] = lms_entry;
            amount++;

        } else {
            null_counter++;
            sa[i] = -1;
        }
    }
    return amount;
}

// At the end of this method t_1 will have the original LMS substrings again instead of the renamed ones
template <typename string_type, typename old_string_type>
inline static void overwrite_lms_after_recursion(string_type &t_0, old_string_type &t_1) {

    bool last_type = false;
    size_t t_1_pointer = 0;


    // Iterate whole string LTR and compare types of symbols with each othernt

    for (size_t i = 0; i < t_0.size(); i++)
    {
        std::tuple<bool, size_t> current_type = get_type_ltr_dynamic(t_0, i);

        // Check if current type (i) was S and last one (i-1) is L, then current one (i) was LMS

        if (last_type && !std::get<0>(current_type)) {
            
            t_1[t_1_pointer] = i;
            t_1_pointer++;
        }

        last_type = std::get<0>(current_type);
        i += std::get<1>(current_type) - 1;
    }
}


template <typename string_type, typename sa_index>
inline static void buckesort_lms_positions(string_type &t , span<sa_index> sa, size_t max_char) {
    util::container<util::sort::bucket> bkt = util::sort::get_buckets<string_type>(t, max_char, 1);

    for (size_t i = 0; i < bkt.size(); i++) {
        bkt[i].position = bkt[i].position + bkt[i].count - 1;
    }

    for (size_t i = t.size(); i > 0; i--) // RTL Iteration of SA
    {

        if (static_cast<size_t>(sa[i - 1]) != 0 && sa[i - 1] != static_cast<sa_index>(-1)) {

            // std::cout << "now observing sa[" << (i - 1) << "] = " << sa[i - 1] << std::endl;

            size_t c = t[sa[i - 1]]; // The character c itself is already his
                                     // bkt index (-1)
            // std::cout << "writing this value to index " << bkt[c].position << std::endl;

            sa[bkt[c].position] = sa[i - 1];
            if (bkt[c].position != i - 1) // We need to overwrite all unused slots with -1 but we  dont want to overwrite the slot that was just written to
                sa[i - 1] = -1;
            bkt[c].position--;
        }
    }

}

} // namespace sacabench::util

/******************************************************************************/
