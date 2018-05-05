/*******************************************************************************
 * util/induce_sa_dc.hpp
 *
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#inlcude "span.hpp"

#pragma once
namespace sacabench::util {
    template<typename C, typename T, typename I, typename S, typename Compare, typename Substring>
    //template T for input string
    //template C for input characters
    //template I for ISA
    //template S for SA

    /**\brief Merge two suffix array with the difference cover idea.
    * \param t input text
    * \param sa_0 calculated SA for triplets beginning in i mod 3 = 0
    * \param sa_12 calculated SA for triplets beginning in i mod 3 != 0
    * \param isa_12 calculated ISA for triplets beginning in i mod 3 != 0
    * \param sa memory block for merged SA
    * \param comp function which compares for strings a and b if a is lexicographically smaller than b
    * \param get_substring function which expects a string t, an index i and an integer n and
    *       returns a substring of t beginning in i where n equally calculated substrings are concatenated
    *
    * This method merges the suffix arrays s_0, which contains the 
    * lexicographical ranks of positions i mod 3 = 0, and s_12, which
    * contains the lexicographical ranks of positions i mod 3 != 0.
    * This method works correct because of the difference cover idea.
    */
    void merge_sa_dc(T& t, S& sa_0, S& sa_12 I& isa_12, S& sa, Compare comp, Substring get_substring) {

        DCHECK(sa.size() > 0);

        int i = 0;
        int j = 0;
        int counter = 0;
        
        while (counter < sa.size()) {
            if (i < sa_0.size() && j < sa_12.size()) {
                if (sa_12[j] % 3 == 1) {
                    span<T> t_0 = get_substring(t, sa_0[i], 1);    
                    span<T> t_12 = get_substring(t, sa_12[j], 1); 
                    
                    if (comp(t_0, t_12) || (!comp(t_0, t_12) && !comp(t_12, t_0) && isa_12[i+t_0.size()] < isa_12[j+t_12.size())) { 
                        sa[counter] = sa_0[i++];
                    }
                    else {
                        sa[counter] = sa_12[j++];
                    }
                }
                else {
                    span<T> t_0 = get_substring(t, sa_0[i], 2);     
                    span<T> t_12 = get_substring(t, sa_12[j], 2);  
                    
                    if (comp(t_0, t_12) || (!comp(t_0, t_12) && !comp(t_12, t_0) && isa_12[i+t_0.size()] < isa_12[j+t_12.size())) { 
                        sa[counter] = sa_0[i++];
                    }
                    else {
                        sa[counter] = sa_12[j++];
                    }
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
}  // namespace sacabench::util