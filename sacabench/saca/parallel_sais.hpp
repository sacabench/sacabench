/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/type_extraction.hpp>
#include<vector>
#include<thread>
#include<iostream>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::parallel_sais {
using namespace sacabench::util;
class parallel_sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "PARALLEL_SAIS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Nong, Zhang and Chan";    
    
    static const bool L_Type = 0;
    static const bool S_Type = 1;

    static bool is_LMS(span<bool> t, ssize position) {
        return (position > 0) && (t[position] == S_Type && t[position - 1] == L_Type);
    }

    template <typename T>
    static void compute_types(span<bool> t, T s, span<size_t> thread_border, span<bool> thread_info, ssize part_length, ssize rest_length, size_t thread_count) {
        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < thread_border.size() - 1; i++) { thread_border[i] = part_length; }
        thread_border[thread_count - 1] = rest_length;

        t[s.size() - 1] = S_Type;

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                threads.push_back(std::thread(compute_types_first_pass<T>, t, s, i * part_length, part_length, i, thread_border, thread_info));
            }
            else {
                threads.push_back(std::thread(compute_types_first_pass<T>, t, s, i * part_length, rest_length, i, thread_border, thread_info));
            }
        }
       
        for (auto& t : threads) {
            t.join();
        }

        // if many threads were not able to classify, use the last thread that has borderinfo for all the others
        for (ssize i = threads.size() - 1; i >= 0; i--) {
            if (thread_border[i] == 0) {
                thread_info[i] = thread_info[i + 1];
            }
        }

        threads.clear();

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                threads.push_back(std::thread(compute_types_second_pass<T>, t, i * part_length, part_length, i, thread_border, thread_info));
            }
            else {
                threads.push_back(std::thread(compute_types_second_pass<T>, t, i * part_length, rest_length, i, thread_border, thread_info));
            }
        }

        for (auto& t : threads) {
            t.join();
        }
    }
    
    template <typename T>
    static void compute_types_first_pass(span<bool> t, T s, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
        // first pass - classify all elements that are possible to be classified within the thread
        for (ssize i = len - 1; i >= 0; i--) {           
            if ((size_t)i + offset + 1 < s.size()) {
                if (s[i + offset + 1] < s[i + offset]) {
                    t[i + offset] = L_Type;
                }
                else if (s[i + offset + 1] > s[i + offset]) {
                    t[i + offset] = S_Type;
                }
                else {
                // do not use types from another thread as we do not know if they are already calculated
                    if (((size_t)i + 1 < len && thread_border[thread_id] != (size_t)i + 1) || (thread_id == thread_border.size() - 1)) {
                        t[i + offset] = t[i + offset + 1];
                    }
                    else {
                        thread_border[thread_id] = i;
                    }
                }
            }
        }

        if (thread_border[thread_id] != 0) {
            thread_info[thread_id] = t[offset];
        }
    }

    template <typename T>
    static void compute_types_second_pass(span<bool> t, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
        // second pass - use info of threads what the type of their border character was
        for (size_t i = thread_border[thread_id]; i < len; i++) {
            t[i + offset] = thread_info[thread_id + 1];
        }
    }    
    
    template <typename T, typename sa_index>
    static void generate_buckets(T s, span<sa_index> buckets, size_t K,
                                 bool end) {
        size_t sum = 0;

        for (size_t i = 0; i < K; i++) {
            buckets[i] = 0;
        }

        // bucket size for each char
        for (size_t i = 0; i < s.size(); i++) {
            buckets[s.at(i)]++;
        }

        // sum up to bucket ends
        for (size_t i = 0; i < K; i++) {
            sum += buckets[i];
            buckets[i] = (end ? sum : sum - buckets[i]);
        }
    }


    template <typename T, typename sa_index>
    static void prepare_parallel(T s, ssize part_length, span<std::pair<char,sa_index>> r,
                                 span<sa_index> SA, span<bool> t, bool suffix_type, size_t thread_count, ssize rest_length){
            std::vector<std::thread> threads;

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                threads.push_back(std::thread(prepare<T,sa_index>, s, part_length, part_length, r,SA, t, suffix_type, i));
            }
            else {
                threads.push_back(std::thread(prepare<T,sa_index>, s,part_length, rest_length, r,SA, t, suffix_type, i));
            }
        }

        for (auto& t : threads) {
            t.join();
        }
    };

    template <typename T, typename sa_index>
    static void prepare(T s, ssize part_length, ssize actual_part_length, span<std::pair<char,sa_index>> r, span<sa_index> SA, span<bool> t, bool suffix_type, size_t k){


        for(ssize_t i = k*part_length;i<((k*part_length)+actual_part_length);i++){
            r[i].first = '\0';
            r[i].second = static_cast<sa_index>(-1);
        }
        size_t j = 0;
        sa_index pos;
        char chr;
        for(ssize_t i = 0;i<actual_part_length;i++){
            j = (k*part_length)+i;
            if(SA[j]!= static_cast<sa_index>(-1)){
                pos = SA[j]-static_cast<sa_index>(1);
                if(pos>=static_cast<sa_index>(0) && pos!=static_cast<sa_index>(-1) && t[pos] == suffix_type){
                    chr = s[pos];
                    r[i] = std::make_pair(chr, pos);
                }
            }
        }

    }

    template <typename T, typename sa_index>
    static void induce_L_Types(T s, span<sa_index> buckets, span<bool> t, size_t K,
                               bool end, span<sa_index> SA) {
        generate_buckets<T, sa_index>(s, buckets, K, end);
        for (size_t i = 0; i < s.size(); i++) {
            ssize pre_index =
                SA[i] - (sa_index)1; // pre index of the ith suffix array position

            if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 &&
                t[pre_index] == L_Type) { // pre index is type L
                SA[buckets[s.at(pre_index)]++] =
                    pre_index; // "sort" index in the bucket
            }
        }
    }

    template <typename T, typename sa_index>
    static void induce_S_Types(T s, span<sa_index> buckets, span<bool> t, size_t K,
                               bool end, span<sa_index> SA) {
        generate_buckets<T, sa_index>(s, buckets, K, end);
        for (ssize i = s.size() - 1; i >= 0; i--) {
            ssize pre_index = SA[i] - (sa_index)1;

            if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && t[pre_index] == S_Type) {
                SA[--buckets[s.at(pre_index)]] = pre_index;
            }
        }
    }

    // Updating and writing into the SuffixArray, w needs to be properly connected to the rest of the code now
    template <typename sa_index>
    static void update_SA(ssize part_length, span<std::pair<char, sa_index>> w, span<sa_index> SA) {
        for (ssize_t i = 0; i < part_length; i++) {

            if (w[i].first != '\0' && w[i].second != static_cast<sa_index>(-1))
            {
                SA[w[i].second] = w[i].first;
            }

        }
    }

    // Initialization of the Write Buffer, maybe can be put together with the Preparing-Phase later
    template <typename sa_index>
    static void init_Write_Buffer(ssize part_length, container<std::pair<char, sa_index>> &w) {
        
        for (ssize_t i = 0; i < part_length; i++) {
            w[i].first = '\0';
            w[i].second = static_cast<sa_index>(-1);
        }
    }

    template <typename T, typename sa_index>
    static void run_saca(T s, span<sa_index> SA, size_t K) {

        container<sa_index> buckets = make_container<sa_index>(K);
        container<bool> t = make_container<bool>(SA.size());
        container<size_t> thread_border = make_container<size_t>(std::thread::hardware_concurrency());
        container<bool> thread_info = make_container<bool>(std::thread::hardware_concurrency());
        
        // Prepare blocks for parallel computing
        size_t thread_count = std::thread::hardware_concurrency();
        thread_count = std::min(thread_count, s.size() - 1);
        ssize part_length = s.size() / thread_count;
        ssize rest_length = (s.size() - (thread_count - 1) * part_length);

        // Read/Write Buffer for the pipeline
        auto r = make_container<std::pair<char,sa_index>>(s.size());
        container<std::pair<char, sa_index>> w = make_container<std::pair<char,sa_index>>(s.size());

        init_Write_Buffer(part_length, w);

        compute_types(t, s, thread_border, thread_info, part_length, rest_length, thread_count);
        
        generate_buckets<T, sa_index>(s, buckets, K, true);
        // Initialize each entry in SA with -1
        for (size_t i = 0; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        // iterate from left to right (starting by 1 cause 0 can never be LMS)
        // and put LMS to end of the bucket and move bucket's tail backwards
        for (size_t i = 1; i < s.size(); i++) {
            if (is_LMS(t, i)) {
                SA[--buckets[s.at(i)]] = i;
            }
        }
        
        // sort LMS substrings
        induce_L_Types<T, sa_index>(s, buckets, t, K, false, SA);
        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);
        
        // because we have at most n/2 LMS, we can store the sorted indices in
        // the first half of the SA
        ssize n1 = 0;
        for (size_t i = 0; i < s.size(); i++) {
            if (is_LMS(t, SA[i]) == 1 || s.size() == 1) {
                SA[n1++] = SA[i];
            }
        }

        // All LMS are now stored in the (at most) first half of the SA, so the
        // rest half of the suffix array can be used
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }

        // The given names correspond to the buckets the LMS are sorted into.
        // To find the names, the strings have to be compared by its type and
        // lexicographical value per char
        ssize name = 0;
        ssize previous_LMS = -1;
        for (ssize i = 0; i < n1; i++) { // max n/2 iterations
            bool diff = false;
            // compare types and chars
            ssize current_LMS = SA[i];
            for (size_t j = 0; j < s.size(); j++) {
                if (previous_LMS == -1 ||
                    s.at(current_LMS + j) != s.at(previous_LMS + j) ||
                    t[current_LMS + j] != t[previous_LMS + j]) {
                    diff = true;
                    break;
                } else if (j > 0 &&
                           (is_LMS(t, current_LMS + j) ||
                            is_LMS(t,
                                   previous_LMS +
                                       j))) { // check if next LMS is reached
                    break;                    // no diff was found
                }
            }

            if (diff) { // if diff was found, adjust the name and continue with
                        // current LMS as previous
                name++;
                previous_LMS = current_LMS;
            }

            current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2
                                                 : (current_LMS - 1) / 2;
            SA[n1 + current_LMS] = name - 1;
        }

        // not needed
        for (ssize i = s.size() - 1, j = s.size() - 1; i >= n1; i--) {
            if (SA[i] >= (sa_index)0 && SA[i] != ((sa_index)-1)) {
                SA[j--] = SA[i];
            }
        }


        span<sa_index> s1 = SA.slice(s.size() - n1, s.size());

        if (name < n1) {
            run_saca<span<sa_index const>, sa_index>(s1, SA, name);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        //prepare_parallel<T, sa_index>(s,part_length,r,SA,t, L_Type, thread_count, rest_length);

        // induce the final SA
        generate_buckets<T, sa_index>(s, buckets, K, true);
        size_t j;
        for (size_t i = 1, j = 0; i < s.size(); i++) {
            if (is_LMS(t, i)) {
                s1[j++] = i;
            }
        }
        for (ssize i = 0; i < n1; i++) {
            SA[i] = s1[SA[i]];
        }
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = (sa_index)-1;
            SA[--buckets[s.at(j)]] = j;
        }

        induce_L_Types<T, sa_index>(s, buckets, t, K, false, SA);
        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        tdc::StatPhase parallel_sais("Main Phase");
        if (text.size() > 1) {
            run_saca<string_span, sa_index>(text, out_sa, alphabet.size_with_sentinel());
        }
    }
};
} // namespace sacabench::parallel_sais
