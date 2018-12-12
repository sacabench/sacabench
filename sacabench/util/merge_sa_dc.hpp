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
        merge_from_left(a_1, a_2, b, swapped, comp, b.size());
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
    void merge_sa_dc_parallel_opt(const util::span<sa_index> sa_0, const util::span<sa_index> sa_12,
                     util::span<sa_index> sa, const Compare comp) {
        //std::cout << "sa_0: " << sa_0 << ", sa_12: " << sa_12 << std::endl; 
        DCHECK_MSG(
            sa.size() == (sa_0.size() + sa_12.size()),
            "the length of sa must be the sum of the length of sa_0 and sa_12");

        omp_set_num_threads(1);
        merge_parallel_opt(sa_0, sa_12, sa, false, comp);
        //std::cout << "sa: " << sa << std::endl;
    }

    template <typename sa_index, typename Compare>
    void merge_parallel(util::span<sa_index> a_1, util::span<sa_index> a_2, util::span<sa_index> b, 
                bool swapped, const Compare comp, size_t num_threads) {                 
        if (a_1.size() < a_2.size()) {
            auto tmp = a_1.slice();
            a_1 = a_2.slice();
            a_2 = tmp.slice();
            swapped = !swapped;
        }
        if (a_1.size() == 0) { return; }
        else {
            size_t q_1 = a_1.size()/2;
            size_t q_2 = binarysearch(a_2, 0, a_2.size(), a_1[q_1], swapped, comp);
            size_t q_out = q_1 + q_2;
            b[q_out] = a_1[q_1];
            
            auto a_1_left = a_1.slice(0, q_1);
            auto a_2_left = a_2.slice(0, q_2);
            auto b_left = b.slice(0, q_out);
              
            auto a_1_right = a_1.slice(std::min(q_1+1, a_1.size()));
            auto a_2_right = a_2.slice(q_2);
            auto b_right = b.slice(std::min(q_out+1, b.size()));
            
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
    
    //TODO: Anzahl Prozessoren als Eingabe
    template <typename sa_index, typename Compare>
    void merge_parallel_opt(util::span<sa_index> a_1, util::span<sa_index> a_2,
                     util::span<sa_index> b, bool swapped, const Compare comp) {
        using marked_element = std::tuple<size_t, bool, size_t>;
                     
        size_t n = a_1.size();
        size_t m = a_2.size();   
        size_t p = 0;
        size_t stepsize = 0;
        util::container<marked_element> marked_elements_1;
        util::container<marked_element> marked_elements_2;
        util::container<marked_element> marked_elements_out;
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                p = omp_get_num_threads();
                stepsize = (m+n)/p + ((m+n) % p > 0);
                
                size_t marked_elements_1_size = n/stepsize;
                size_t marked_elements_2_size = m/stepsize;
                size_t marked_elements_out_size = marked_elements_1_size+marked_elements_2_size;
                
                marked_elements_1 = 
                        util::container<marked_element>(marked_elements_1_size);
                marked_elements_2 = 
                        util::container<marked_element>(marked_elements_2_size);
                marked_elements_out = 
                        util::container<marked_element>(marked_elements_out_size);
            }
            
            #pragma omp for
            for (size_t i = 1; i <= marked_elements_1.size(); ++i) {
                marked_elements_1[i-1] = marked_element(i*stepsize-1, 0, i-1);
            }
        
            #pragma omp for
            for (size_t i = 1; i <= marked_elements_2.size(); ++i) {
                marked_elements_2[i-1] = marked_element(i*stepsize-1, 1, i-1);
            }
        }
        
        /*std::cout << "marked_elements_1: " << std::endl;
        for (size_t i = 0; i < marked_elements_1.size(); ++i) {
            auto elem_1 = std::get<0>(marked_elements_1[i]);
            auto elem_2 = std::get<1>(marked_elements_1[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        /*std::cout << "marked_elements_2: " << std::endl;
        for (size_t i = 0; i < marked_elements_2.size(); ++i) {
            auto elem_1 = std::get<0>(marked_elements_2[i]);
            auto elem_2 = std::get<1>(marked_elements_2[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        auto comp_marked_elements = [&](marked_element a, 
                marked_element b) {
            return comp(a_1[std::get<0>(a)], a_2[std::get<0>(b)]);
        };
        //TODO: Implement Valiants merge
        util::span<marked_element> marked_elements_1_span = marked_elements_1;
        util::span<marked_element> marked_elements_2_span = marked_elements_2;
        util::span<marked_element> marked_elements_out_span = marked_elements_out;
        merge_parallel(marked_elements_1_span, marked_elements_2_span, marked_elements_out_span, swapped, 
                comp_marked_elements, 1);
                
        /*std::cout << "marked_elements_out: " << std::endl;
        for (size_t i = 0; i < marked_elements_out.size(); ++i) {
            auto elem_1 = std::get<0>(marked_elements_out[i]);
            auto elem_2 = std::get<1>(marked_elements_out[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
                
        util::container<std::tuple<size_t, size_t>> segments = 
                util::container<std::tuple<size_t, size_t>>(marked_elements_out.size());       
        util::span<std::tuple<size_t, size_t>> segments_span = segments;
        determine_segments(a_1, a_2, marked_elements_out_span, segments_span, marked_elements_1.size(), 
                marked_elements_2.size(), stepsize);
                
        /*std::cout << "segments: " << std::endl;
        for (size_t i = 0; i < segments.size(); ++i) {
            auto elem_1 = std::get<0>(segments[i]);
            auto elem_2 = std::get<1>(segments[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        auto pos_1 = util::make_container<std::tuple<size_t, bool>>(marked_elements_2.size());
        auto pos_2 = util::make_container<std::tuple<size_t, bool>>(marked_elements_1.size());
                
        /* Binary Search of marked elements in segments and write them to correct 
           positions in b */
        #pragma omp parallel for
        for (size_t i = 0; i < marked_elements_out.size(); ++i) {
            size_t pos = std::get<0>(marked_elements_out[i]); 
            sa_index elem = 0;
            bool is_in_2 = std::get<1>(marked_elements_out[i]);
            size_t pos_in_pos = std::get<2>(marked_elements_out[i]);
            size_t left = std::get<0>(segments[i]);
            size_t right = std::get<1>(segments[i]);
            
            size_t pos_in_other = 0;
            if (!is_in_2) {
                elem = a_1[pos];
                pos_in_other = binarysearch(a_2, left, right, elem, false, comp);
                pos_2[pos_in_pos] = std::tuple<size_t, bool>(pos_in_other, 0);
            }
            else {
                elem = a_2[pos];
                pos_in_other = binarysearch(a_1, left, right, elem, true, comp);
                pos_1[pos_in_pos] = std::tuple<size_t, bool>(pos_in_other, 0);
            }
            
            b[pos+pos_in_other] = elem;
        }
        
        /*std::cout << "pos_1: " << std::endl;
        for (size_t i = 0; i < pos_1.size(); ++i) {
            auto elem_1 = std::get<0>(pos_1[i]);
            auto elem_2 = std::get<1>(pos_1[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        // determine subsegments
        using scheduled_pair = std::tuple<util::span<sa_index>, util::span<sa_index>, 
                util::span<sa_index>, bool, size_t>;
        using processor_list = std::vector<scheduled_pair>;  
        auto subsegments_1_cont = util::make_container<std::tuple<size_t, size_t>>(
                pos_1.size()+marked_elements_1.size()+1);
        util::span<std::tuple<size_t, size_t>> subsegments_1 = subsegments_1_cont;
        auto subsegments_2_cont = util::make_container<std::tuple<size_t, size_t>>(
                pos_2.size()+marked_elements_2.size()+1);
        util::span<std::tuple<size_t, size_t>> subsegments_2 = subsegments_2_cont;
        util::span<std::tuple<size_t, bool>> pos_1_span = pos_1;
        util::span<std::tuple<size_t, bool>> pos_2_span = pos_2;
        determine_subsegments(a_1, a_2, pos_1_span, pos_2_span, stepsize, subsegments_1, subsegments_2);
        
        /*std::cout << "subsegments_1: " << std::endl;
        for (size_t i = 0; i < subsegments_1.size(); ++i) {
            auto elem_1 = std::get<0>(subsegments_1[i]);
            auto elem_2 = std::get<1>(subsegments_1[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        /*std::cout << "subsegments_2: " << std::endl;
        for (size_t i = 0; i < subsegments_2.size(); ++i) {
            auto elem_1 = std::get<0>(subsegments_2[i]);
            auto elem_2 = std::get<1>(subsegments_2[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl; */
        
        // schedule subsegments to processors
        auto schedule = util::make_container<processor_list>(p);
        #pragma omp parallel for 
        for (size_t i = 0; i < schedule.size(); ++i) {
            schedule[i] = std::vector<scheduled_pair>();
        }
        #pragma omp parallel for
        for (size_t i = 0; i < subsegments_1.size(); ++i) {
            auto subsegment_1 = subsegments_1[i];
            auto left_1 = std::get<0>(subsegment_1);
            auto right_1 = std::get<1>(subsegment_1);
            auto span_1 = a_1.slice(left_1, right_1);
            
            auto subsegment_2 = subsegments_2[i];
            auto left_2 = std::get<0>(subsegment_2);
            auto right_2 = std::get<1>(subsegment_2);
            auto span_2 = a_2.slice(left_2, right_2);
            
            auto left_out = left_1+left_2;
            auto right_out = left_out+span_1.size()+span_2.size();
            auto span_out = b.slice(left_out, right_out);
            
            auto processor_1 = 0;
            auto processor_2 = 0;
            auto mapped_pos_1 = right_1/stepsize;
            auto mapped_pos_2 = right_2/stepsize;
            scheduled_pair pair_1 = std::tuple(span_1, span_2, span_out, 0, span_1.size());
            scheduled_pair pair_2 = std::tuple(span_1, span_2, span_out, 1, span_2.size());
            if (mapped_pos_1 < marked_elements_1.size()) {
                processor_1 = mapped_pos_1;
            }
            else {
                processor_1 = p-1;
            }
            if (mapped_pos_2 < marked_elements_2.size()) {
                processor_2 = mapped_pos_2 + marked_elements_1.size();
            }
            else {
                processor_2 = p-1;
            }
            #pragma omp critical
            {
                schedule[processor_1].push_back(pair_1);
                schedule[processor_2].push_back(pair_2);
            }
        }
        
        /*for (size_t i = 0; i < schedule.size(); ++i) {
            processor_list list = schedule[i];
            for (size_t j = 0; j < list.size(); ++j) {
                auto elem_1 = std::get<0>(list[j]);
                auto elem_2 = std::get<1>(list[j]);
                auto elem_3 = std::get<2>(list[j]);
                auto elem_4 = std::get<3>(list[j]);
                std::cout << "<" << elem_1 << ", " << elem_2 << ", " << elem_3 << ", " << elem_4 << ">";
            }
            std::cout << std::endl;
        }*/
        
        // merge subsegments by calculated schedule
        #pragma omp parallel for
        for (size_t i = 0; i < schedule.size(); ++i) {
            processor_list list = schedule[i];
            for (scheduled_pair pair : list) {
                auto first_span = std::get<0>(pair);
                auto second_span = std::get<1>(pair);
                auto out_span = std::get<2>(pair);
                auto from_right = std::get<3>(pair);
                auto steps = std::get<4>(pair);
                
                if (!from_right) {
                    merge_from_left(first_span, second_span, out_span, swapped, comp, steps);
                }
                else {
                    merge_from_right(first_span, second_span, out_span, swapped, comp, steps);
                }
            }
        }
    }
    
    template <typename sa_index, typename Compare>
    void merge_from_left(util::span<sa_index> a_1, util::span<sa_index> a_2,
                     util::span<sa_index> b, bool swapped, const Compare comp, size_t steps) { 
        size_t i = 0;
        size_t j = 0;

        for(size_t index = 0; index < steps; ++index) {
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
    void merge_from_right(util::span<sa_index> a_1, util::span<sa_index> a_2,
                    util::span<sa_index> b, bool swapped, const Compare comp, size_t steps) { 
        size_t i = a_1.size();
        size_t j = a_2.size();

        for(size_t index = b.size(); index > b.size()-steps; --index) {
            if (i > 0 && j > 0) {
                sa_index elem_a_1 = a_1[i-1];
                sa_index elem_a_2 = a_2[j-1];
                
                bool greater = false;
                if (!swapped) { greater = !comp(elem_a_1, elem_a_2); }
                else { greater = comp(elem_a_2, elem_a_1); }
                
                if (greater) {
                    b[index-1] = elem_a_1;
                    --i;
                }
                else {
                    b[index-1] = elem_a_2;
                    --j;
                }
            }

            else if (i == 0) {
                b[index-1] = a_2[j-1];
                --j;
            }
            else {
                b[index-1] = a_1[i-1];
                --i;
            }
        }           
    }
    
    template <typename sa_index>
    void determine_segments(util::span<sa_index> a_1, util::span<sa_index> a_2,
            util::span<std::tuple<size_t, bool, size_t>> marked_elements,
            util::span<std::tuple<size_t, size_t>> segments, size_t marked_elements_1_size,
            size_t marked_elements_2_size, size_t stepsize) {
        // Positions of marked elements of a_1 and a_2 in merged array        
        util::container<size_t> pos_1 = util::container<size_t>(marked_elements_1_size); 
        util::container<size_t> pos_2 = util::container<size_t>(marked_elements_2_size);

        #pragma omp parallel 
        {
            /* Fill pos_1 and pos_2. We simply have to divide by
               stepsize because the merged array contains positions 
               of marked elements */
            #pragma omp for
            for (size_t i = 0; i < marked_elements.size(); ++i) {
                size_t elem = std::get<0>(marked_elements[i]);
                bool is_in_2 = std::get<1>(marked_elements[i]);
                if (!is_in_2) { pos_1[elem/stepsize] = i; }
                else { pos_2[elem/stepsize] = i; }
            }
            
            /*#pragma omp single 
            {
                std::cout << "pos_1: " << pos_1 << std::endl;
                std::cout << "pos_2: " << pos_2 << std::endl;
            }*/
            
            //Determine segments of marked elements of a_2 in a_1
            #pragma omp for
            for (size_t i = 0; i <= pos_1.size(); ++i) {
                size_t l = 0;
                size_t r = 0;
                size_t pos_l = 0;
                size_t pos_r = 0;
                if (pos_1.size() == 0) {
                    l = 0;
                    r = marked_elements.size();
                    pos_l = 0;
                    pos_r = a_1.size();
                }
                else if (i == 0) {
                    l = 0;
                    r = pos_1[i];
                    pos_l = 0;
                    pos_r = std::get<0>(marked_elements[r]);
                }
                else if (i < pos_1.size()) {
                    l = pos_1[i-1]+1;
                    r = pos_1[i];
                    pos_l = std::get<0>(marked_elements[l-1])+1;
                    pos_r = std::get<0>(marked_elements[r]);
                }
                else {
                    l = pos_1[i-1]+1;
                    r = marked_elements.size();
                    pos_l = std::get<0>(marked_elements[l-1])+1;
                    pos_r = a_1.size();
                }
                
                //TODO: #pragma omp for
                for (size_t j = l; j < r; ++j) {
                    segments[j] = std::tuple<size_t, size_t>(pos_l, pos_r);
                }
            }
            
            //Determine segments of marked elements of a_1 in a_2
            #pragma omp for
            for (size_t i = 0; i <= pos_2.size(); ++i) {
                size_t l = 0;
                size_t r = 0;
                size_t pos_l = 0;
                size_t pos_r = 0;
                if (pos_2.size() == 0) {
                    l = 0;
                    r = marked_elements.size();
                    pos_l = 0;
                    pos_r = a_2.size();
                }
                else if (i == 0) {
                    l = 0;
                    r = pos_2[i];
                    pos_l = 0;
                    pos_r = std::get<0>(marked_elements[r]);
                }
                else if (i < pos_2.size()) {
                    l = pos_2[i-1]+1;
                    r = pos_2[i];
                    pos_l = std::get<0>(marked_elements[l-1])+1;
                    pos_r = std::get<0>(marked_elements[r]);
                }
                else {
                    l = pos_2[i-1]+1;
                    r = marked_elements.size();
                    pos_l = std::get<0>(marked_elements[l-1])+1;
                    pos_r = a_2.size();
                }
                
                //TODO: #pragma omp for
                for (size_t j = l; j < r; ++j) {
                    segments[j] = std::tuple<size_t, size_t>(pos_l, pos_r);
                }
            }
        }
    }
    
    template <typename sa_index>
    void determine_subsegments(util::span<sa_index> a_1, util::span<sa_index> a_2,
            util::span<std::tuple<size_t, bool>> pos_1, util::span<std::tuple<size_t, bool>> pos_2, 
            size_t stepsize, util::span<std::tuple<size_t, size_t>>& subsegments_1, 
            util::span<std::tuple<size_t, size_t>>& subsegments_2) {
        auto pos_marked_elements_1 = util::make_container<std::tuple<size_t, bool>>(a_1.size()/stepsize);
        auto pos_marked_elements_2 = util::make_container<std::tuple<size_t, bool>>(a_2.size()/stepsize);

        #pragma omp parallel for
        for (size_t i = 0; i < pos_marked_elements_1.size()+pos_marked_elements_2.size(); ++i) {
            if (i < pos_marked_elements_1.size()) {
                pos_marked_elements_1[i] = std::tuple<size_t, bool>((i+1)*stepsize-1, 1);
            }
            else {
                size_t j = i-pos_marked_elements_1.size();
                pos_marked_elements_2[j] = std::tuple<size_t, bool>((j+1)*stepsize-1, 1);
            }
        }
        
        /*std::cout << "pos_marked_elements_1: " << std::endl;
        for (size_t i = 0; i < pos_marked_elements_1.size(); ++i) {
            auto elem_1 = std::get<0>(pos_marked_elements_1[i]);
            auto elem_2 = std::get<1>(pos_marked_elements_1[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl;*/
        
        /*std::cout << "pos_1: " << std::endl;
        for (size_t i = 0; i < pos_1.size(); ++i) {
            auto elem_1 = std::get<0>(pos_1[i]);
            auto elem_2 = std::get<1>(pos_1[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">"; 
        }
        std::cout << std::endl;*/
        
        /*std::cout << "pos_marked_elements_2: " << std::endl;
        for (size_t i = 0; i < pos_marked_elements_2.size(); ++i) {
            auto elem_1 = std::get<0>(pos_marked_elements_2[i]);
            auto elem_2 = std::get<1>(pos_marked_elements_2[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl; */
        
        auto borders_1 = util::make_container<std::tuple<size_t, bool>>(pos_1.size()+pos_marked_elements_1.size());
        auto borders_2 = util::make_container<std::tuple<size_t, bool>>(pos_2.size()+pos_marked_elements_2.size());
        util::span<std::tuple<size_t, bool>> pos_marked_elements_1_span = pos_marked_elements_1;
        util::span<std::tuple<size_t, bool>> pos_marked_elements_2_span = pos_marked_elements_2;
        util::span<std::tuple<size_t, bool>> borders_1_span = borders_1;
        util::span<std::tuple<size_t, bool>> borders_2_span = borders_2;
        
        auto comp = [&](std::tuple<size_t, bool> a, std::tuple<size_t, bool> b) {
            return std::get<0>(a) < std::get<0>(b);
        };
        merge_parallel(pos_marked_elements_1_span, pos_1, borders_1_span, false, comp, 1);
        merge_parallel(pos_marked_elements_2_span, pos_2, borders_2_span, false, comp, 1);
        
        /*std::cout << "borders_1_span: " << std::endl;
        for (size_t i = 0; i < borders_1_span.size(); ++i) {
            auto elem_1 = std::get<0>(borders_1_span[i]);
            auto elem_2 = std::get<1>(borders_1_span[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl; */
        
        /*std::cout << "borders_2_span: " << std::endl;
        for (size_t i = 0; i < borders_2_span.size(); ++i) {
            auto elem_1 = std::get<0>(borders_2_span[i]); 
            auto elem_2 = std::get<1>(borders_2_span[i]);
            std::cout << "<" << elem_1 << ", " << elem_2 << ">";
        }
        std::cout << std::endl; */
        
        #pragma omp parallel for
        for (size_t i = 0; i <= borders_1_span.size(); ++i) {
            if (borders_1_span.size() == 0) {
                subsegments_1[0] = std::tuple<size_t, size_t>(0, a_1.size());
            }
            else if (i == 0) {
                auto pos = std::get<0>(borders_1_span[i]);
                subsegments_1[i] = std::tuple<size_t, size_t>(0, pos);
            }
            else if (i < borders_1_span.size()) {
                auto pos_1 = std::get<0>(borders_1_span[i-1]);
                auto pos_2 = std::get<0>(borders_1_span[i]);
                auto is_marked_1 = std::get<1>(borders_1_span[i-1]);
                if (!is_marked_1) {
                    subsegments_1[i] = std::tuple<size_t, size_t>(pos_1, pos_2);
                }
                else {
                    subsegments_1[i] = std::tuple<size_t, size_t>(pos_1+1, pos_2);
                }
            }
            else {
                auto pos = std::get<0>(borders_1_span[i-1]);
                auto is_marked = std::get<1>(borders_1_span[i-1]);
                if (!is_marked) {
                    subsegments_1[i] = std::tuple<size_t, size_t>(pos, a_1.size());
                }
                else {
                    subsegments_1[i] = std::tuple<size_t, size_t>(pos+1, a_1.size());
                }
            }
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i <= borders_2_span.size(); ++i) {
            if (borders_2_span.size() == 0) {
                subsegments_2[0] = std::tuple<size_t, size_t>(0, a_2.size());
            }
            else if (i == 0) {
                auto pos = std::get<0>(borders_2_span[i]);
                subsegments_2[i] = std::tuple<size_t, size_t>(0, pos);
            }
            else if (i < borders_2_span.size()) {
                auto pos_1 = std::get<0>(borders_2_span[i-1]);
                auto pos_2 = std::get<0>(borders_2_span[i]);
                auto is_marked_1 = std::get<1>(borders_2_span[i-1]);
                if (!is_marked_1) {
                    subsegments_2[i] = std::tuple<size_t, size_t>(pos_1, pos_2);
                }
                else {
                    subsegments_2[i] = std::tuple<size_t, size_t>(pos_1+1, pos_2);
                }
            }
            else {
                auto pos = std::get<0>(borders_2_span[i-1]);
                auto is_marked = std::get<1>(borders_2_span[i-1]);
                if (!is_marked) {
                    subsegments_2[i] = std::tuple<size_t, size_t>(pos, a_2.size());
                }
                else {
                    subsegments_2[i] = std::tuple<size_t, size_t>(pos+1, a_2.size());
                }
            }
        }
        
        /*using segment_type = std::tuple<size_t, size_t> 
        using segment_list_type = std::vector<segment_type>;
        
        size_t segments_1_size = a_1.size()/stepsize + (a_1.size() % stepsize > 0);
        size_t segments_2_size = a_2.size()/stepsize + (a_2.size() % stepsize > 0);
        auto segments_1 = util::make_container<segment_list_type>(segments_1_size);
        auto segments_2 = util::make_container<segment_list_type>(segments_2_size);
        
        #pragma omp parallel for
        for (size_t i = 0; i < segments_1.size()+segments_2.size(); ++i) {
            if (i < segments_1.size()) {
                segments_1[i] = std::vector<segment_type>();
                auto segment_list = segments_1[i];
                size_t left = i*stepsize;
                size_t right = std::min((i+1)*stepsize-1, a_1.size());
                segment_list.push_back(segment_type(left, right));
            }
            else {
                size_t j = segments_1.size()-i;
                segments_2[j] = std::vector<segment_type>();
                auto segment_list = segments_2[j];
                size_t left = j*stepsize;
                size_t right = std::min((j+1)*stepsize-1, a_2.size());
                segment_list.push_back(segment_type(left, right));
            }
        }
        
        //TODO: Segmente in Subsegmente unterteilen
        #pragma omp parallel for
        for (size_t i = 0; i < pos_1.size()+pos_2.size(); ++i) {
            if (i < pos_1.size()) {
                size_t segment = pos_1[i]/stepsize;
            }
            else {
                j = pos_1.size()-i;
                
            }
        }
        
        //TODO: Subsegmente in richtigen Speicherbereich kopieren*/
    }
    
    /*template <typename sa_index>
    void determine_subsegments(util::span<sa_index> a_1, util::span<sa_index> a_2,
            util::span<std::tuple<size_t, bool>> pos_1, util::span<std::tuple<size_t, bool>> pos_2, 
            size_t stepsize, util::span<std::tuple<size_t, size_t>>& subsegments_1, 
            util::span<std::tuple<size_t, size_t>>& subsegments_2) {
        auto elem_1 = std::tuple<size_t, size_t>(0,1);
        auto elem_2 = std::tuple<size_t, size_t>(1,1);
        auto elem_3 = std::tuple<size_t, size_t>(1,3);
        auto elem_4 = std::tuple<size_t, size_t>(4,5);
        subsegments_1[0] = elem_1;
        subsegments_1[1] = elem_2;
        subsegments_1[2] = elem_3;
        subsegments_1[3] = elem_4;
        
        auto elem_5 = std::tuple<size_t, size_t>(0,3);
        auto elem_6 = std::tuple<size_t, size_t>(4,7);
        auto elem_7 = std::tuple<size_t, size_t>(8,9);
        auto elem_8 = std::tuple<size_t, size_t>(9,10);
        subsegments_2[0] = elem_5;
        subsegments_2[1] = elem_6;
        subsegments_2[2] = elem_7;
        subsegments_2[3] = elem_8;
    }*/

    template <typename sa_index, typename C, typename Compare>
    size_t binarysearch(const C array, const size_t left, 
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
                const size_t right, sa_index key, const bool swapped, const Compare comp, size_t p) {
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
        util::container<bool> comp_results = util::container<bool>(p);
        #pragma omp parallel
        {
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
        
        return binarysearch_parallel(array, new_left, new_right, key, swapped, comp, p);
    }
} // namespace sacabench::util