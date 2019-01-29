/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/kd_array.hpp>
#include <util/prefix_sum.hpp>
#include <math.h>

#include <omp.h>

#include <tudocomp_stat/StatPhase.hpp>

#pragma once
namespace sacabench::util {
    
    //TODO: Use parallelized prefix_sum
    template <typename Array>
    void prefix_sum(Array& array) {
        auto add = [&](size_t a, size_t b) {
            return a + b;
        };
        util::par_prefix_sum_eff(array, array, true, add, (size_t)0);
            /*
            for (size_t i = 1; i < array.size(); ++i) {
                array[i] += array[i-1];
            }*/
    }
    
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
     * bolobolo
     */
    template <typename T, typename sa_index>
    void induce_type_s_parallel(const T& t, util::span<sa_index> sa, const size_t max_character) {

        DCHECK_MSG(sa.size() == t.size(),
                   "sa must be initialised and must have the same length as t.");

        // TODO: generate Buckets
        
        for (size_t c = max_character+1; c > 0; --c) {
            size_t curr_char = c-1;
            
            induce_repititions_parallel(t, sa, curr_char);
            
        }
    }
    
    template <typename T, typename sa_index>
    void induce_type_s_repititions_parallel(const T& t, util::span<sa_index> sa, const size_t curr_char) {
        size_t s = 0;
        size_t e = 1;
        util::span<sa_index> interval = sa.slice(s, e); // TODO: get Interval of initialized SA entrys with first character curr_char
            
        while (interval.size() != 0) {
            // count number of positions which have curr_char before the first character in interval
            size_t count = 0;
            #pragma omp parallel for reduction(+:count)
            for (size_t i = 0; i < interval.size(); ++i)  {
                if (sa[i] > 0 && t[sa[i]-1] = curr_char) {
                    count += 1;
                }
            }
            
            //TODO: Reihenfolge ist hier nicht korrekt, Verwirrung wegen A, B und B* Suffixen
            
            // copy entries which have curr_char before the first character to new_interval
            util::span<sa_index> new_interval = sa.slice(s-count, s);
            auto has_same_prev_char = [&](sa_index a) {
                return curr_char == t[a-1];
            };
            filter(interval, new_interval, has_same_prev_char);
            
            // correct the new entries in sa
            interval = new_interval;
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                new_interval[i]--;
            }
        }
    }
    
    template <typename T, typename sa_index>
    void induce_type_l_parallel(const T& t, util::span<sa_index> sa, const size_t max_character) {

        //TODO
        /*DCHECK_MSG(sa.size() == t.size(),
                   "sa must be initialised and must have the same length as t.");*/

        tdc::StatPhase induce("Generate Buckets");
        
        auto buckets_cont = util::container<size_t>(max_character+1);
        util::span<size_t> buckets = buckets_cont;
        generate_buckets_parallel(t, buckets);
        auto l_buckets_cont = util::container<size_t>(max_character+1);
        util::span<size_t> l_buckets = l_buckets_cont;
        std::copy(buckets.begin(), buckets.end(), l_buckets.begin());
        
        //std::cout << "text: " << t << std::endl;
        //std::cout << "a" << std::endl;
        
        for (size_t c = 0; c <= max_character; ++c) {
            induce.split("Induce Repetitions");
            //std::cout << "sa_begin: " << sa << std::endl;
            
            size_t curr_char = c;
            
            //std::cout << "b" << std::endl;
            
            induce_type_l_repititions_parallel(t, sa, curr_char, buckets, l_buckets);
            //std::cout << "sa_after_repitition: " << sa << std::endl;
            
            if (c == max_character) { break; } //TODO: unschön
            
            //std::cout << "bb" << std::endl;
            
            // get Interval of bucket with first character curr_char
            size_t s = buckets[curr_char];
            size_t e = 0;
            if (curr_char < max_character) {
                e = buckets[curr_char+1];
            }
            else {
                e = sa.size();
            }
            //std::cout << "curr_char: " << curr_char << ", s: " << s << ", e: " << e << std::endl;
            util::span<sa_index> interval = sa.slice(s, e);
            
            //std::cout << "c" << std::endl;
            
            induce.split("Initialize bucket_sums");
            
            // bucket_sums[d][i] = 1 if position i in interval writes to bucket d
            // TODO (possible optimization): Since we can write only in greater buckets, we can reduce the array size
            auto bucket_sums = util::array2d<size_t>({max_character+1, interval.size()});
            // initialize bucket_sums with 0
            /*#pragma omp parallel for
            for (size_t k = 0; k < (max_character+1)*interval.size(); ++k) {
                size_t first = k/interval.size();
                size_t second = k % interval.size();
                bucket_sums[first][second] = 0;
            }*/
            
            induce.split("fill bucket_sums");
            //std::cout << "cc" << std::endl;
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                if (interval[i] != (sa_index)-1 && interval[i] != (sa_index)0) { 
                    sa_index pre_index = interval[i]-(sa_index)1;
                    if (is_l_type(t, l_buckets, interval[i], s+i)
                            && t[pre_index] != t[interval[i]]) { // To ensure repititions are only calculated once
                        bucket_sums[t[pre_index]][i] = 1;
                    }
                }
            }
            
            //std::cout << "d" << std::endl;
        
            induce.split("perform prefix sums on bucket_sums");
            
            // perform prefix sums on each bucket_sum array
            #pragma omp parallel for
            for (size_t d = 0; d <= max_character; ++d) {
                //TODO: workaround to access 1d-array bucket_sums[d]: copy elements to new container
                auto bucket_sums_1d_cont = util::container<size_t>(interval.size());
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums_1d[i] = bucket_sums[d][i];
                }                
                
                prefix_sum(bucket_sums_1d);
                
                // copy elements back to bucket_sums[d]
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums[d][i] = bucket_sums_1d[i];
                } 
            }
            
            //std::cout << "e" << std::endl;
            
            induce.split("induce not repetitions");
            
            // initialize new entries
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                //std::cout << "ea" << std::endl;
                if (interval[i] != (sa_index)-1 && interval[i] != (sa_index)0) { 
                    sa_index pre_index = interval[i]-(sa_index)1;
            //std::cout << "eb" << std::endl;
                    auto pre_char = t[pre_index];
            //std::cout << "ec" << std::endl;
                    if (is_l_type(t, l_buckets, interval[i], s+i)
                            && t[pre_index] != t[interval[i]]) { // To ensure repititions are only calculated once
            //std::cout << "ed" << std::endl;
                        sa[l_buckets[pre_char]+bucket_sums[pre_char][i]-1] = pre_index;
            //std::cout << "ee" << std::endl;
                    }
                }
            }
            
            //std::cout << "f" << std::endl;
            
            induce.split("Increase bucket pointers");
            
            // increase bucket pointers
            #pragma omp parallel for
            for (size_t d = 0; d < l_buckets.size(); ++d) {
                l_buckets[d] += bucket_sums[d][interval.size()-1];
            }
            
            //std::cout << "sa_end: " << sa << std::endl;
        }
        //std::cout << "Ende: " << std::endl;
    }
    
    template <typename T, typename sa_index, typename B>
    void induce_type_l_repititions_parallel(const T& t, util::span<sa_index> sa, const size_t curr_char, 
                const B& buckets, B& l_buckets) {
        // get Interval of initialized type-L SA entrys with first character curr_char
        size_t s = buckets[curr_char];
        size_t e = l_buckets[curr_char];
        util::span<sa_index> interval = sa.slice(s, e);
        
        //std::cout << "s: " << s << ", e: " << e << std::endl;
            
        while (interval.size() != 0) {
            // count number of positions which have curr_char before the first character in interval
            size_t count = 0;
            #pragma omp parallel for reduction(+:count)
            for (size_t i = 0; i < interval.size(); ++i)  {
                sa_index curr_elem = interval[i];
                if (curr_elem != (sa_index)-1 && curr_elem != (sa_index)0  
                            && t[curr_elem-(sa_index)1] == curr_char) {
                    count += 1;
                }
            }
            
            //std::cout << "count: " << count << std::endl;
            
            // copy entries which have curr_char before the first character to new_interval
            util::span<sa_index> new_interval = sa.slice(e, e+count);
            auto has_same_prev_char = [&](sa_index a) {
                if (a > (sa_index)0) {
                    return curr_char == t[a-(sa_index)1];
                }
                else { return false; }
            };
            filter(interval, new_interval, has_same_prev_char);
            
            //std::cout << "x" << std::endl;
            
            // correct the new entries in sa
            s = e;
            e +=count;
            interval = sa.slice(s, e);
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                new_interval[i]--;
            }
            
            //std::cout << "y" << std::endl;
            
            // increase bucket pointer
            l_buckets[curr_char] += count;
            
            //std::cout << "z" << std::endl;
        }
    }
    
    template <typename T, typename B, typename sa_index>
    bool is_l_type(T& t, B& l_buckets, sa_index curr_pos_in_t, size_t curr_pos_in_sa) {
        sa_index pre_pos_in_t = curr_pos_in_t-(sa_index)1;
        bool curr_pos_is_l_type = (curr_pos_in_sa < l_buckets[t[curr_pos_in_t]]); // bucket pointer points at l-type-positions 
        return t[pre_pos_in_t] > t[curr_pos_in_t] || (t[pre_pos_in_t] == t[curr_pos_in_t] && curr_pos_is_l_type);
    }
    
    template <typename Array, typename Filter>
    void filter(Array interval, Array out_interval, Filter filter_func) {
        /* indicator for interval. filtered_elements[i] = 1 means that interval[i]
           is filtered*/
        auto filtered_elements_cont = util::container<size_t>(interval.size());
        util::span<size_t> filtered_elements = filtered_elements_cont;
        
        //std::cout << "xx" << std::endl;
        // fill filtered_elements
        #pragma omp parallel for
        for (size_t i = 0; i < interval.size(); ++i) {
            if (filter_func(interval[i])) {
                filtered_elements[i] = 1;
            }
        }
        
        //std::cout << "yy" << std::endl;
        // perform prefix_sum on filtered_elements
        prefix_sum(filtered_elements);
        
        //std::cout << "zz" << std::endl;
        // write filtered elements to out_interval
        #pragma omp parallel for
        for (size_t i = 0; i < interval.size(); ++i) {
            if (filter_func(interval[i])) {
                auto pos = filtered_elements[i]-1;
                out_interval[pos] = interval[i];
            }
        }
        //std::cout << "zzz" << std::endl;
    }
    
    template <typename T, typename B>
    void generate_buckets_parallel(T& text, B& buckets) {
        #pragma omp parallel for
        for (size_t i = 0; i < buckets.size(); ++i) {
            buckets[i] = 0;
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < text.size(); ++i) {
            auto curr_char = text[i];
            if (curr_char < buckets.size()-1) {
                #pragma omp atomic 
                buckets[++curr_char]++;
            }
        }
        
        prefix_sum(buckets);
    }
} // namespace sacabench::util
