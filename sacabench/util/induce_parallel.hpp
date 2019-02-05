/*******************************************************************************
 * Copyright (C) 2019 Nico Bertram <nico.bertram@tu-dortmund.de>
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
    template <typename Array>
    void prefix_sum(Array& array) {
        auto add = [&](size_t a, size_t b) {
            return a + b;
        };
        util::par_prefix_sum_eff(array, array, true, add, (size_t)0);
    }
    
    /**\brief Parallel version of S-Type-Inducing
     * \tparam T input string
     * \tparam sa_index content of SA
     * \param t input text
     * \param sa SA after L-Type-Inducing
     * \param max_character Value of the largest character in effective alphabet
     *
     * Parallel version of S-Type-Inducing by Labeit. Takes the input string t,
     * the SA after L-Type-Inducing and the value of the largest character in the
     * effective alphabet as parameter. After calling this function sa contains the 
     * correct Suffixarray of t.
     */
    template <typename T, typename sa_index>
    void induce_type_s_parallel(const T& t, util::span<sa_index> sa, const size_t max_character) {

        //TODO
        /*DCHECK_MSG(sa.size() == t.size(),
                   "sa must be initialised and must have the same length as t.");*/

        tdc::StatPhase induce("Generate Buckets");
        
        auto buckets_cont = util::container<size_t>(max_character+1);
        util::span<size_t> buckets = buckets_cont;
        generate_buckets_parallel(t, buckets);
        auto s_buckets_cont = util::container<size_t>(max_character+1);
        util::span<size_t> s_buckets = s_buckets_cont;
        #pragma omp parallel for
        for (size_t i = 0; i < s_buckets.size(); ++i) {
            if (i < s_buckets.size()-1) { s_buckets[i] = buckets[i+1]-1; }
            else { s_buckets[i] = sa.size()-1; }
        }
        
        for (size_t c = max_character+1; c > 0; --c) {
            size_t curr_char = c-1;
            
            // get Interval of bucket with first character curr_char
            size_t s = buckets[curr_char];
            size_t e = 0;
            if (curr_char < max_character) {
                e = buckets[curr_char+1];
            }
            else {
                e = sa.size();
            }
            util::span<sa_index> interval = sa.slice(s, e);
            
            size_t n = t.size();
            size_t block_size = log2(n)*(max_character+1); // blocks of size sigma*logn
            size_t block_count = interval.size()/block_size + (interval.size() % block_size != 0);
            
            induce.split("Sequential");
            if (interval.size() < 4*block_size) {
                induce_type_s_interval_seq(t, interval, s, sa, s_buckets);
                continue;
            }
            
            induce.split("Induce Repetitions");
            
            induce_type_s_repititions_parallel(t, sa, curr_char, buckets, s_buckets);
            
            if (curr_char == 0) { break; } //TODO: unschön
            
            induce.split("Initialize bucket_sums");
            
            // bucket_sums[d][i] = 1 if position i in interval writes to bucket d
            //size_t n = std::max((size_t)2, interval.size());
            auto bucket_sums = util::array2d<size_t>({max_character+1, block_count+1});
            
            // initialize bucket_sums with 0
            /*#pragma omp parallel for
            for (size_t k = 0; k < (max_character+1)*interval.size(); ++k) {
                size_t first = k/interval.size();
                size_t second = k % interval.size();
                bucket_sums[first][second] = 0;
            }*/
            
            induce.split("fill bucket_sums");
            #pragma omp parallel for
            for (size_t i = 0; i < block_count; ++i) {
                // get current block
                util::span<sa_index> block = interval.slice(0); // Dummy
                auto offset = i*block_size;
                if (interval.size() >= offset+block_size) {
                    block = interval.slice(interval.size()-(offset+block_size), 
                                    interval.size()-(offset));
                }
                else {
                    block = interval.slice(0, interval.size()-offset);
                }
                
                //TODO: false sharing bei bucket_sums?
                auto bucket_sums_1d_cont = util::container<size_t>(max_character+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                
                // fill bucket_sums
                for (size_t j = block.size(); j > 0; --j) {
                    auto curr_j = j-1;
                    auto elem = block[curr_j];
                    if (elem != (sa_index)-1 && elem != (sa_index)0) { 
                        sa_index pre_index = elem-(sa_index)1;
                        auto pre_char = t[pre_index];
                        size_t curr_pos_in_sa = e-offset+curr_j;
                        if (pre_char != curr_char // To ensure repititions are only calculated once
                                && is_s_type(t, s_buckets, elem, curr_pos_in_sa, pre_char)) {
                            //bucket_sums[pre_char][i]++;
                            bucket_sums_1d[pre_char]++;
                        }
                    }
                }
                
                #pragma omp parallel for
                for (size_t j = 0; j <= max_character; ++j) {
                    bucket_sums[j][i] = bucket_sums_1d[j];
                }
            }
        
            induce.split("perform prefix sums on bucket_sums");
            
            // perform prefix sums on each bucket_sum array
            #pragma omp parallel for
            for (size_t d = 0; d <= max_character; ++d) {
                //TODO: workaround to access 1d-array bucket_sums[d]: copy elements to new container
                auto bucket_sums_1d_cont = util::container<size_t>(block_count+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums_1d[i] = bucket_sums[d][i];
                }                
                
                auto add = [&](size_t a, size_t b) {
                    return a + b;
                };
                util::par_prefix_sum_eff(bucket_sums_1d, bucket_sums_1d, false, add, (size_t)0);
                
                // copy elements back to bucket_sums[d]
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums[d][i] = bucket_sums_1d[i];
                } 
            }
            
            induce.split("induce not repetitions");
            
            // initialize new entries
            #pragma omp parallel for
            for (size_t i = 0; i < block_count; ++i) {
                // get current block
                util::span<sa_index> block = interval.slice(0); // Dummy
                auto offset = i*block_size;
                if (interval.size() >= offset+block_size) {
                    block = interval.slice(interval.size()-(offset+block_size), 
                                    interval.size()-(offset));
                }
                else {
                    block = interval.slice(0, interval.size()-offset);
                }
                
                //TODO: false sharing bei bucket_sums?
                auto bucket_sums_1d_cont = util::container<size_t>(max_character+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                #pragma omp parallel for
                for (size_t j = 0; j <= max_character; ++j) {
                    bucket_sums_1d[j] = bucket_sums[j][i];
                }
                
                for (size_t j = block.size(); j > 0; --j) {
                    auto curr_j = j-1;
                    auto elem = block[curr_j];
                    if (elem != (sa_index)-1 && elem != (sa_index)0) { 
                        sa_index pre_index = elem-(sa_index)1;
                        auto pre_char = t[pre_index];
                        size_t curr_pos_in_sa = e-offset+curr_j;
                        if (pre_char != curr_char // To ensure repititions are only calculated once
                                && is_s_type(t, s_buckets, elem, curr_pos_in_sa, pre_char)) { 
                            auto s_bucket_last = s_buckets[pre_char];
                            //auto bucket_write_count_prev = bucket_sums[pre_char][i]++;
                            auto bucket_write_count_prev = bucket_sums_1d[pre_char]++;
                            sa[s_bucket_last-bucket_write_count_prev] = pre_index;
                        }
                    }
                }
            }
            
            induce.split("Update bucket pointers");
            
            // update bucket pointers
            #pragma omp parallel for
            for (size_t d = 0; d < s_buckets.size(); ++d) {
                s_buckets[d] -= bucket_sums[d][block_count];
            }
        }
    }
    
    /* This function induces all S-Type suffixes which have multiple occurences 
       of curr_char at the beginning of the suffix. */
    template <typename T, typename sa_index, typename B>
    void induce_type_s_repititions_parallel(const T& t, util::span<sa_index> sa, const size_t curr_char, 
                const B& buckets, B& s_buckets) {
        // get Interval of initialized type-S SA entrys with first character curr_char
        size_t s = s_buckets[curr_char]+1;
        size_t e = curr_char != buckets.size()-1 ? buckets[curr_char+1] : sa.size();
        util::span<sa_index> interval = sa.slice(s, e);
            
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
            
            // copy entries which have curr_char before the first character to new_interval
            util::span<sa_index> new_interval = sa.slice(s-count, s);
            auto has_same_prev_char = [&](sa_index a) {
                if (a > (sa_index)0) {
                    return curr_char == t[a-(sa_index)1];
                }
                else { return false; }
            };
            filter(interval, new_interval, has_same_prev_char);
            
            // correct the new entries in sa
            e = s;
            s -= count;
            interval = sa.slice(s, e);
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                new_interval[i]--;
            }
            
            // update bucket pointer
            s_buckets[curr_char] -= count;
        }
    }
    
    /* This function induces all S-Type suffixes sequentially in an interval of sa.*/
    template <typename T, typename sa_index, typename B>
    void induce_type_s_interval_seq(const T& t, const util::span<sa_index> interval, const size_t s, 
            util::span<sa_index> sa, B& s_buckets) {
        for (size_t i = interval.size(); i > 0; --i) {
            auto curr_i = i-1;
            auto elem = interval[curr_i];
            if (elem != (sa_index)-1 && elem != (sa_index)0) {
                sa_index pre_index = elem-(sa_index)1;
                auto pre_char = t[pre_index];
                if (is_s_type(t, s_buckets, elem, s+curr_i, pre_char)) {
                    sa[s_buckets[pre_char]--] = pre_index;
                }
            }
        }
    }
    
    /* This function checks if the position in t before curr_pos_in_t is of type S. */
    template <typename T, typename B, typename sa_index>
    bool is_s_type(T& t, B& s_buckets, sa_index curr_pos_in_t, size_t curr_pos_in_sa, size_t prev_char) {
        auto curr_char = t[curr_pos_in_t];
        bool curr_pos_is_s_type = (curr_pos_in_sa > s_buckets[curr_char]); // bucket pointer points at s-type-positions 
        return prev_char < curr_char || (prev_char == curr_char && curr_pos_is_s_type);
    }
    
    /**\brief Parallel version of L-Type-Inducing
     * \tparam T input string
     * \tparam sa_index content of SA
     * \param t input text
     * \param sa SA after placing the LMS-Suffixes in their buckets
     * \param max_character Value of the largest character in effective alphabet
     *
     * Parallel version of L-Type-Inducing by Labeit. Takes the input string t,
     * the SA after after placing the LMS-Suffixes in their buckets and the value of 
     * the largest character in the effective alphabet as parameter. After calling 
     * this function sa contains the correct order of the L-Type Suffixes of t.
     */
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
        
        for (size_t c = 0; c <= max_character; ++c) {
            size_t curr_char = c;
            
            // get Interval of bucket with first character curr_char
            size_t s = buckets[curr_char];
            size_t e = 0;
            if (curr_char < max_character) {
                e = buckets[curr_char+1];
            }
            else {
                e = sa.size();
            }
            util::span<sa_index> interval = sa.slice(s, e);
            
            size_t n = t.size();
            size_t block_size = log2(n)*(max_character+1); // blocks of size sigma*logn
            size_t block_count = interval.size()/block_size + (interval.size() % block_size != 0);
            
            induce.split("Sequential");
            if (interval.size() < 4*block_size) {
                induce_type_l_interval_seq(t, interval, s, sa, l_buckets);
                continue;
            }
            
            induce.split("Induce Repetitions");
            
            induce_type_l_repititions_parallel(t, sa, curr_char, buckets, l_buckets);
            
            if (curr_char == max_character) { break; } //TODO: unschön
            
            induce.split("Initialize bucket_sums");
            
            // bucket_sums[d][i] = 1 if position i in interval writes to bucket d
            //size_t n = std::max((size_t)2, interval.size());
            auto bucket_sums = util::array2d<size_t>({max_character+1, block_count+1});
            
            // initialize bucket_sums with 0
            /*#pragma omp parallel for
            for (size_t k = 0; k < (max_character+1)*interval.size(); ++k) {
                size_t first = k/interval.size();
                size_t second = k % interval.size();
                bucket_sums[first][second] = 0;
            }*/
            
            induce.split("fill bucket_sums");
            #pragma omp parallel for
            for (size_t i = 0; i < block_count; ++i) {
                // get current block
                util::span<sa_index> block = interval.slice(0); // Dummy
                auto offset = i*block_size;
                if (offset+block_size <= interval.size()) {
                    block = interval.slice(offset, offset+block_size);
                }
                else {
                    block = interval.slice(offset);
                }
                
                //TODO: false sharing bei bucket_sums?
                auto bucket_sums_1d_cont = util::container<size_t>(max_character+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                
                // fill bucket_sums
                for (size_t j = 0; j < block.size(); ++j) {
                    auto elem = block[j];
                    if (elem != (sa_index)-1 && elem != (sa_index)0) { 
                        sa_index pre_index = elem-(sa_index)1;
                        auto pre_char = t[pre_index];
                        size_t curr_pos_in_sa = s+offset+j;
                        if (pre_char != curr_char // To ensure repititions are only calculated once
                                && is_l_type(t, l_buckets, elem, curr_pos_in_sa, pre_char)) {
                            //bucket_sums[pre_char][i]++;
                            bucket_sums_1d[pre_char]++;
                        }
                    }
                }
                
                #pragma omp parallel for
                for (size_t j = 0; j <= max_character; ++j) {
                    bucket_sums[j][i] = bucket_sums_1d[j];
                }
            }
        
            induce.split("perform prefix sums on bucket_sums");
            
            // perform prefix sums on each bucket_sum array
            //#pragma omp parallel for
            for (size_t d = 0; d <= max_character; ++d) {
                //TODO: workaround to access 1d-array bucket_sums[d]: copy elements to new container
                auto bucket_sums_1d_cont = util::container<size_t>(block_count+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums_1d[i] = bucket_sums[d][i];
                }
                
                auto add = [&](size_t a, size_t b) {
                    return a + b;
                };
                util::par_prefix_sum_eff(bucket_sums_1d, bucket_sums_1d, false, add, (size_t)0);
                
                // copy elements back to bucket_sums[d]
                #pragma omp parallel for
                for (size_t i = 0; i < bucket_sums_1d.size(); ++i) {
                    bucket_sums[d][i] = bucket_sums_1d[i];
                } 
            }
            
            induce.split("induce not repetitions");
            
            // initialize new entries
            #pragma omp parallel for
            for (size_t i = 0; i < block_count; ++i) {
                // get current block
                util::span<sa_index> block = interval.slice(0); // Dummy
                auto offset = i*block_size;
                if (offset+block_size <= interval.size()) {
                    block = interval.slice(offset, offset+block_size);
                }
                else {
                    block = interval.slice(offset);
                }
                
                //TODO: false sharing bei bucket_sums?
                auto bucket_sums_1d_cont = util::container<size_t>(max_character+1);
                util::span<size_t> bucket_sums_1d = bucket_sums_1d_cont;
                #pragma omp parallel for
                for (size_t j = 0; j <= max_character; ++j) {
                    bucket_sums_1d[j] = bucket_sums[j][i];
                }
                
                for (size_t j = 0; j < block.size(); ++j) {
                    auto elem = block[j];
                    if (elem != (sa_index)-1 && elem != (sa_index)0) { 
                        sa_index pre_index = elem-(sa_index)1;
                        auto pre_char = t[pre_index];
                        size_t curr_pos_in_sa = s+offset+j;
                        if (pre_char != curr_char // To ensure repititions are only calculated once
                                && is_l_type(t, l_buckets, elem, curr_pos_in_sa, pre_char)) { 
                            auto l_bucket_first = l_buckets[pre_char];
                            //auto bucket_write_count_prev = bucket_sums[pre_char][i]++;
                            auto bucket_write_count_prev = bucket_sums_1d[pre_char]++;
                            sa[l_bucket_first+bucket_write_count_prev] = pre_index;
                        }
                    }
                }
            }
            
            induce.split("Increase bucket pointers");
            
            // increase bucket pointers
            #pragma omp parallel for
            for (size_t d = 0; d < l_buckets.size(); ++d) {
                l_buckets[d] += bucket_sums[d][block_count];
            }
        }
    }
    
    /* This function induces all L-Type suffixes which have multiple occurences 
       of curr_char at the beginning of the suffix. */
    template <typename T, typename sa_index, typename B>
    void induce_type_l_repititions_parallel(const T& t, util::span<sa_index> sa, const size_t curr_char, 
                const B& buckets, B& l_buckets) {
        // get Interval of initialized type-L SA entrys with first character curr_char
        size_t s = buckets[curr_char];
        size_t e = std::min(sa.size(), l_buckets[curr_char]);
        util::span<sa_index> interval = sa.slice(s, e);
            
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
            
            // copy entries which have curr_char before the first character to new_interval
            util::span<sa_index> new_interval = sa.slice(e, e+count);
            auto has_same_prev_char = [&](sa_index a) {
                if (a > (sa_index)0) {
                    return curr_char == t[a-(sa_index)1];
                }
                else { return false; }
            };
            filter(interval, new_interval, has_same_prev_char);
            
            // correct the new entries in sa
            s = e;
            e +=count;
            interval = sa.slice(s, e);
            #pragma omp parallel for
            for (size_t i = 0; i < interval.size(); ++i) {
                new_interval[i]--;
            }
            
            // increase bucket pointer
            l_buckets[curr_char] += count;
        }
    }
    
    /* This function induces all L-Type suffixes sequentially in an interval of sa.*/
    template <typename T, typename sa_index, typename B>
    void induce_type_l_interval_seq(const T& t, const util::span<sa_index> interval, const size_t s, 
            util::span<sa_index> sa, B& l_buckets) {
        for (size_t i = 0; i < interval.size(); ++i) {
            auto elem = interval[i];
            if (elem != (sa_index)-1 && elem != (sa_index)0) { 
                sa_index pre_index = elem-(sa_index)1;
                auto pre_char = t[pre_index];
                if (is_l_type(t, l_buckets, elem, s+i, pre_char)) {
                    sa[l_buckets[pre_char]++] = pre_index;
                }
            }
        }
    }
    
    /* This function checks if the position in t before curr_pos_in_t is of type S. */
    template <typename T, typename B, typename sa_index>
    bool is_l_type(T& t, B& l_buckets, sa_index curr_pos_in_t, size_t curr_pos_in_sa, size_t prev_char) {
        auto curr_char = t[curr_pos_in_t];
        bool curr_pos_is_l_type = (curr_pos_in_sa < l_buckets[curr_char]); // bucket pointer points at l-type-positions 
        return prev_char > curr_char || (prev_char == curr_char && curr_pos_is_l_type);
    }
    
    /**\brief Filters all elements in an interval by a filter function
     * \tparam Array type for consecutive memory space
     * \tparam Filter filter function
     * \param interval input interval
     * \param out_interval output interval which contains all elements of
                interval for which filter_func returns true after calling
                this function
     * \param filter_func filter function which must return a bool value
     *
     * Takes as input an interval and a filter function and writes all elements
     * for which the filter function returns true into an output interval.
     * This operation is parallelized.
     */
    template <typename Array, typename Filter>
    void filter(Array interval, Array out_interval, Filter filter_func) {
        /* indicator for interval. filtered_elements[i] = 1 means that interval[i]
           is filtered*/
        auto filtered_elements_cont = util::container<size_t>(interval.size());
        util::span<size_t> filtered_elements = filtered_elements_cont;
        
        // fill filtered_elements
        #pragma omp parallel for
        for (size_t i = 0; i < interval.size(); ++i) {
            if (filter_func(interval[i])) {
                filtered_elements[i] = 1;
            }
        }
        
        // perform prefix_sum on filtered_elements
        prefix_sum(filtered_elements);
        
        // write filtered elements to out_interval
        #pragma omp parallel for
        for (size_t i = 0; i < interval.size(); ++i) {
            if (filter_func(interval[i])) {
                auto pos = filtered_elements[i]-1;
                out_interval[pos] = interval[i];
            }
        }
    }
    
    /* This Function calculates the bucket borders of each character 
       of text parallely. */
    template <typename T, typename B>
    void generate_buckets_parallel(T& text, B& buckets) {
        const size_t buckets_length = buckets.size();
        auto buckets_ptr = buckets.data();
        
        #pragma omp parallel for reduction(+:buckets_ptr[:buckets_length])
        for (size_t i = 0; i < text.size(); ++i) {
            auto curr_char = text[i];
            if (curr_char < buckets.size()-1) {
                ++curr_char;
                buckets_ptr[curr_char]++;
            }
        }
        
        prefix_sum(buckets);
    }
} // namespace sacabench::util
