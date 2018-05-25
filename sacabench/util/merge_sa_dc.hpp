/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <util/span.hpp>
#include <util/assertions.hpp>
#include <util/string.hpp>


#pragma once
namespace sacabench::util {
    template<typename C, typename T, typename Compare, typename Substring, typename S, 
            typename SA, typename S0, typename I>
    //template T for input string
    //template C for input characters
    //template I for ISA
    //template S for SA
    //template Compare for comp function
    //template Substring for get_substring function

    /**\brief Merge two suffix array with the difference cover idea.
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
    static void merge_sa_dc(const T& t, const S0& sa_0, const SA& sa_12, const I& isa_12, S& sa, Compare comp, 
            const Substring get_substring) {

        std::cout << "sa_0 groesse: " << sa_0.size() << " sa_12 groesse: " << sa_12.size() << " sa groesse: " << sa.size() << std::endl;
        DCHECK_MSG(sa.size() == t.size(), 
                "sa must be initialised and must have the same length as t.");
        DCHECK_MSG(sa.size() == (sa_0.size() + sa_12.size()), 
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
                    t_0 = get_substring(t, &t[sa_0[i]], 1, sa_0[i]);    
                    t_12 = get_substring(t, &t[sa_12[j]], 1, sa_12[j]); 
                }
                else {
                    t_0 = get_substring(t, &t[sa_0[i]], 2, sa_0[i]);    
                    t_12 = get_substring(t, &t[sa_12[j]], 2, sa_12[j]); 
                }
                
                const bool less_than = comp(t_0, t_12);
                const bool eq = !comp(t_0, t_12) && !comp(t_12, t_0);
                const bool lesser_suf = isa_12[(2*(sa_0[i]+t_0.size()))/3] 
                    < isa_12[2*((sa_12[j]+t_12.size()))/3];
                if (less_than || (eq && lesser_suf)) { 
                    sa[counter] = sa_0[i++];
                }
                else { sa[counter] = sa_12[j++]; }
            }
            
            else if (i >= sa_0.size()) { sa[counter] = sa_12[j++]; }
            
            else { sa[counter] = sa_0[i++]; }
            
            ++counter;
        }
    }
}  // namespace sacabench::util