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

#include <omp.h>

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

        for(size_t index = 0; index < sa.size(); ++index) {
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
                    sa[index] = sa_0[i++];
                } else {
                    sa[index] = sa_12[j++];
                }
            }

            else if (i >= sa_0.size()) {
                sa[index] = sa_12[j++];
            }

            else {
                sa[index] = sa_0[i++];
            }
        }
    }
    
    template <typename sa_index, typename Compare>
    void merge(util::span<sa_index> a_1, util::span<sa_index> a_2,
                     util::span<sa_index> b, bool swapped, const Compare comp) {
        size_t i = 0;
        size_t j = 0;

        for(size_t index = 0; index < b.size(); ++index) {
            if (i < a_1.size() && j < a_2.size()) {
                sa_index elem_a_1 = a_1[i];
                sa_index elem_a_2 = a_2[j];
                
                bool less = false;
                if (!swapped) { less = comp(elem_a_1, elem_a_2); }
                else { less = !comp(elem_a_2, elem_a_1); }
                
                if (less) {
                    b[index] = elem_a_1;
                    ++i;
                }
                else {
                    b[index] = elem_a_2;
                    ++j;
                }
            }

            else if (i >= a_1.size()) {
                b[index] = a_2[j++];
            }
            else {
                b[index] = a_1[i++];
            }
        }
    }

    template <typename sa_index, typename Compare>
    void merge_sa_dc_parallel(const util::span<sa_index> sa_0, const util::span<sa_index> sa_12,
                     util::span<sa_index> sa, const Compare comp) {
        //std::cout << "sa_0: " << sa_0 << ", sa_12: " << sa_12 << std::endl; 
        DCHECK_MSG(
            sa.size() == (sa_0.size() + sa_12.size()),
            "the length of sa must be the sum of the length of sa_0 and sa_12");

        merge_parallel(sa_0, sa_12, sa, false, comp, 1);
        //std::cout << "sa: " << sa << std::endl;
    }

    template <typename sa_index, typename Compare>
    void merge_parallel(util::span<sa_index> a_1, util::span<sa_index> a_2,
                     util::span<sa_index> b, bool swapped, const Compare comp, size_t num_threads) {                 
        if (a_1.size() < a_2.size()) {
            auto tmp = a_1.slice();
            a_1 = a_2.slice();
            a_2 = tmp.slice();
            swapped = !swapped;
        }
        if (a_1.size() == 0) { return; }
        else {
            size_t q_1 = a_1.size()/2;
            size_t q_2 = binarysearch_parallel(a_2, 0, a_2.size(), a_1[q_1], swapped, comp);
            size_t q_out = q_1 + q_2;
            b[q_out] = a_1[q_1];
            
            util::span<sa_index> a_1_left = a_1.slice(0, q_1);
            util::span<sa_index> a_2_left = a_2.slice(0, q_2);
            util::span<sa_index> b_left = b.slice(0, q_out);
              
            util::span<sa_index> a_1_right = a_1.slice(std::min(q_1+1, a_1.size()));
            util::span<sa_index> a_2_right = a_2.slice(q_2);
            util::span<sa_index> b_right = b.slice(std::min(q_out+1, b.size()));
            
            size_t max_threads = omp_get_max_threads();
            
            if (num_threads < max_threads) {
                #pragma omp parallel num_threads(2)
                {
                    #pragma omp single nowait
                    {
                        #pragma omp task
                        merge_parallel(a_1_left, a_2_left, b_left, swapped, comp, 2*num_threads);
                        #pragma omp task
                        merge_parallel(a_1_right, a_2_right, b_right, swapped, comp, 2*num_threads);
                    }
                }
            }
            else {
                merge(a_1_left, a_2_left, b_left, swapped, comp);
                merge(a_1_right, a_2_right, b_right, swapped, comp);
            }
        }
    }
    
    template <typename sa_index, typename Compare>
    size_t binarysearch(const util::span<sa_index> array, const size_t left, 
                const size_t right, sa_index key, const bool swapped, const Compare comp) {
        if (left+1 > right) { return left; }
        if (left+1 == right) {
            bool less = false;
            if (!swapped) { less = comp(key, array[left]); }
            else { less = !comp(array[left], key); }
            
            if (less) { return left; }
            else { return left+1; }
        }
        
        int middle = (left+right)/2;
        
        bool less = false;
        if (!swapped) { less = comp(key, array[middle]); }
        else { less = !comp(array[middle], key); }
        
        if (less) {
            return binarysearch(array, left, middle, key, swapped, comp);
        }    
        else {
            return binarysearch(array, middle, right, key, swapped, comp);
        }
    }
    
    template <typename sa_index, typename Compare>
    size_t binarysearch_parallel(const util::span<sa_index> array, const size_t left, 
                const size_t right, sa_index key, const bool swapped, const Compare comp) {
        size_t n = right-left;
        
        if (n < 10000) {
            return binarysearch(array, left, right, key, swapped, comp);
        }
        
        if (left+1 > right) { return left; }
        if (left+1 == right) {
            bool less = false;
            if (!swapped) { less = comp(key, array[left]); }
            else { less = !comp(array[left], key); }
            
            if (less) { return left; }
            else { return left+1; }
        }
        
        size_t new_left = 0;
        size_t new_right = 0;
        size_t p = 0;
        util::container<bool> comp_results;
        #pragma omp parallel
        {
            #pragma omp single
            {
                p = std::min((size_t)omp_get_num_threads(),n);
                comp_results = util::container<bool>(p);
            }
            
            #pragma omp for
            for (size_t i = 0; i < p; ++i) {
                sa_index elem = array[left + n/(p+1)*(i+1)];
                
                if (!swapped) { comp_results[i] = comp(key, elem); }
                else { comp_results[i] = !comp(elem, key); }
            }
            
            #pragma omp for
            for (size_t i = 0; i < p; ++i) {
                if (i == 0 && comp_results[0] == true) {
                    new_left = left;
                    new_right = left + n/(p+1)*1;
                }
                else if (i == p-1 && comp_results[p-1] == false) {
                    new_left = left + n/(p+1)*(p);
                    new_right = right;
                }
                else if (comp_results[i] == false && comp_results[i+1] == true) {
                    new_left = left + n/(p+1)*(i+1);
                    new_right = left + n/(p+1)*(i+2);
                }
            }
        }
        
        return binarysearch(array, new_left, new_right, key, swapped, comp);
    }
} // namespace sacabench::util