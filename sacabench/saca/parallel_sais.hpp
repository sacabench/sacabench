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
#include<string.h>
#include<sstream>

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
    static void prepare_parallel(T s, ssize part_length, container<std::pair<sa_index, sa_index>> *r_,
                                 span<sa_index> SA, span<bool> t, bool suffix_type, size_t thread_count, size_t blocknum){

        auto &r = *r_;
        // overwrite readbuffer with NULLs
        for (ssize i = 0; i < (ssize)r.size(); i++) {
            r[i].first = (sa_index)(0);
            r[i].second = static_cast<sa_index>(-1);
        }

            std::vector<std::thread> threads;

        for (size_t i = 0; i < thread_count-1; i++) {
                threads.push_back(std::thread(prepare<T,sa_index>, s, part_length, r_,SA, t, suffix_type, blocknum, i));
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    template <typename T, typename sa_index>
    static void prepare(T s, ssize part_length, container<std::pair<sa_index, sa_index>> *r_, span<sa_index> SA, span<bool> t, bool suffix_type, size_t k, size_t i){

        auto &r = *r_;

        size_t j = 0;
        sa_index pos;
        sa_index chr;

        j = (k*part_length)+i;
        if(j < (size_t)SA.size() && SA[j]!= static_cast<sa_index>(-1)){
            pos = SA[j]-static_cast<sa_index>(1);
            if(pos >=static_cast<sa_index>(0) && pos!=static_cast<sa_index>(-1) && pos < t.size() && t[pos] == suffix_type){
                chr = s[pos];
                r[i] = std::make_pair(chr, pos);
                std::cout << "Write Tuple <" << (ssize)chr << ", " << (ssize)pos << "> to r, j = " << j << ", k = " << k << ", i = " << i << ", pl = " << part_length << std::endl;
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

    // Induce L_Types for Block B_ks
    // k = blocknum
    template <typename T, typename sa_index>
    static void induce_L_Types_Pipelined(T s, span<sa_index> SA, span<sa_index> buckets, span<bool> t, size_t blocknum,
        container<std::pair<sa_index, sa_index>> &r, container<std::pair<sa_index, sa_index>> &w, ssize part_length) {

        // translate: translates the position in the block to global pos
        // w_count: pointer for the next free entry in write-buffer w
        sa_index translate = (sa_index)((blocknum)*(part_length));
        size_t w_count = 0;
        sa_index chr;

        for (ssize i = 0; i < (ssize)part_length && i+(ssize)translate < (ssize)SA.size(); i++)
        {
            ssize pos = ((ssize)SA[i + translate] - 1);
            std::cout << "i: " << (sa_index)i << ", pos: " << pos << ", i+trans: " << (i+translate) << std::endl;

            if ((ssize)SA[(sa_index)i + translate] >= (ssize)(0) && pos >= (ssize)0 && pos < (ssize)SA.size() && t[pos] == L_Type)
            {

                if (r[i].first == static_cast<sa_index>(0))
                    chr = (sa_index)s[(sa_index)pos];
                else
                    chr = (sa_index)r[i].first;

                std::cout << "chr: " << (sa_index)chr << std::endl;

                sa_index idx = buckets[chr];
                buckets[chr]++;

                std::cout << "idx: " << (sa_index)idx << std::endl;

                // if idx is in Block k or Block k+1
                if (translate <= idx && idx <= translate + (sa_index)(2 * part_length) && idx < SA.size()) {
                    SA[idx] = (sa_index)pos;
                    std::cout << "Directly written " << (size_t)pos << " to pos " << (size_t)idx << std::endl;
                }
                else if (idx < SA.size()) {
                    w[w_count++] = std::make_pair((sa_index)idx, (sa_index)pos);
                    std::cout << "Inserted Touple <" << (ssize)idx << ", " << (ssize)pos << "> at w_count " << (ssize)(w_count - 1) << " in w" << std::endl;
                    std::cout << "Insertion Check <" << (ssize)(w[w_count-1].first) << ", " << (ssize)(w[w_count - 1].second) << "> at w_count " << (ssize)(w_count - 1) << " in w" << std::endl;
                }

            }
        }

        // Print w
        for (sa_index i = 0; i < w.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "w: [ ";

            std::cout << "<" << (ssize)w[i].first << "," << (ssize)w[i].second << "> ";

            if (i == (sa_index)w.size() - (sa_index)1)
                std::cout << "]" << std::endl;
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
    static void update_SA(ssize part_length, container<std::pair<sa_index, sa_index>> *w_, span<sa_index> SA) {
        
        auto &w = *w_;

        for (ssize_t i = 0; i < part_length; i++) {

            if (w[i].first != static_cast<sa_index>(0) && w[i].second != static_cast<sa_index>(-1) && (size_t)w[i].first < SA.size())
            {
                SA[w[i].first] = w[i].second; 
            }


            w[i].first = static_cast<sa_index>(0);
            w[i].second = (sa_index)(-1);

        }
    }

    template <typename sa_index>
    static void update_parallel(size_t thread_count, ssize part_length, container<std::pair<sa_index, sa_index>> *w, span<sa_index> SA) {

        std::vector<std::thread> threads;

        // std::cout << "Start updating" << std::endl;

        for (size_t i = 0; i < thread_count; i++) {
            threads.push_back(std::thread(update_SA<sa_index>, part_length, w, SA));
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    // Initialization of the Write Buffer, maybe can be put together with the Preparing-Phase later
    template <typename sa_index>
    static void init_Write_Buffer(ssize part_length, container<std::pair<sa_index, sa_index>> &w) {
        
        for (ssize_t i = 0; i < part_length; i++) {
            w[i].first = (sa_index)(0);
            w[i].second = static_cast<sa_index>(-1);
        }
    }

    template <typename T, typename sa_index>
    static void run_saca(T s, span<sa_index> SA, size_t K) {

        for (size_t i = 0; i < s.size(); i++)
        {
            if (i == 0)
                std::cout << "    Text : ";

            std::cout << (ssize)(s[i]) << " ";
        }

        std::cout << std::endl;

        std::cout << "s.size : " << s.size() << std::endl;
        std::cout << "SA.size: " << SA.size() << std::endl;


        container<sa_index> buckets = make_container<sa_index>(K);
        container<bool> t = make_container<bool>(SA.size());
        container<size_t> thread_border = make_container<size_t>(std::thread::hardware_concurrency());
        container<bool> thread_info = make_container<bool>(std::thread::hardware_concurrency());
        
        // Prepare blocks for parallel computing
        size_t thread_count = std::thread::hardware_concurrency();
        thread_count = std::min(thread_count, s.size() - 1);
        thread_count = 1;
        ssize part_length = s.size() / thread_count;
        ssize rest_length = (s.size() - (thread_count - 1) * part_length);

        // Read/Write Buffer for the pipeline
        container<std::pair<sa_index, sa_index>> r = make_container<std::pair<sa_index,sa_index>>(s.size());
        container<std::pair<sa_index, sa_index>> w = make_container<std::pair<sa_index,sa_index>>(s.size());

        init_Write_Buffer(part_length, w);

        compute_types(t, s, thread_border, thread_info, part_length, rest_length, thread_count);

        // First Induction ###################################################

        // Inserting LMS 
        {
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
        }

        generate_buckets<T, sa_index>(s, buckets, K, false);
       
        // Main Loop for each block, need to add shifted parallelization for blocks later
        for (size_t blocknum = 0; blocknum <= thread_count; blocknum++)
        {
            std::cout << "loop for block " << blocknum << std::endl;

            // Parallel Preparation Phase
            // TODO: Wait until preparing is fixed, give blocknum as parameter
            prepare_parallel<T, sa_index>(s, part_length, &r, SA, t, L_Type, thread_count, blocknum);

            induce_L_Types_Pipelined<T, sa_index>(s, SA, buckets, t, blocknum, r, w, part_length);

            // Parallel Updating Phase

            update_parallel<sa_index>(thread_count, part_length, &w, SA);
        }

        // Print out SA
        for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA at the end of L pipeline :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }

        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);

        // Print out SA
        for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA after inducing S-Types :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }
        
        // Recursion Call #############################################################
        
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

        for (ssize i = s.size() - 1, j = s.size() - 1; i >= n1; i--) {
            if (SA[i] >= (sa_index)0 && SA[i] != ((sa_index)-1)) {
                SA[j--] = SA[i];
            }
        }


        span<sa_index> s1 = SA.slice(s.size() - n1, s.size());
        span<sa_index> sa_ = SA.slice(0, n1);

        if (name < n1) {
            run_saca<span<sa_index const>, sa_index>(s1, sa_, name);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        std::cout << "start final inducing..." << std::endl;
        for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA before inducing LMS :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }
        // FINAL INDUCING ##########################################################

        // induce the final SA
        generate_buckets<T, sa_index>(s, buckets, K, true);
        std::cout << "buckets generated" << std::endl;
        size_t j;
        for (size_t i = 1, j = 0; i < s.size(); i++) {
            std::cout << "check is_LMS" << std::endl;
            if (is_LMS(t, i)) {
                std::cout << "finished checking, j = " << j << ", i = "  << i << std::endl;
                s1[j++] = i;
            }
            std::cout << "finished iteration" << std::endl;

        }
        std::cout << "start final inducing1..." << std::endl;
        for (ssize i = 0; i < n1; i++) {
            SA[i] = s1[SA[i]];
        }
        std::cout << "start final inducing2..." << std::endl;
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        std::cout << "start final inducing3..." << std::endl;
        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = (sa_index)-1;
            SA[--buckets[s.at(j)]] = j;
        }

        std::cout << "finished inducing LMS..." << std::endl;

        // Print out SA
        /*for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA before final Inducing :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }*/

         generate_buckets<T, sa_index>(s, buckets, K, false);

        //Main Loop for each block, need to add shifted parallelization for blocks later
        
        
                // Shifted parallelization is applied here -> B0 preparing, B0 Inducing B1 Preparing, B0 Updating B1 Inducing B2 Preparing ...
        // For n blocks, n + 2 iterations are neeeded.
        
        ssize preparing_block = -1;
        ssize inducing_block = -2;
        ssize updating_block = -3;
        
        for (size_t blocknum = 0; blocknum <= thread_count + 2; blocknum++) {
        
            // Preparing, inducing and updating. Note, that in the first iteration, only the first block is being prepared.
            if (blocknum < thread_count + 1) {
                preparing_block++;
                inducing_block++;
                updating_block++;
            }
            
            // Preparing for all block finished
            if (blocknum == thread_count + 1) {
                preparing_block = -1;
                inducing_block++;
                updating_block++;
            }
            
            // Preparing and inducing finished
            if (blocknum == thread_count + 2) {
                preparing_block = -1;
                inducing_block = -1;
                updating_block++;
            }
            
            
            
            if (preparing_block >= (ssize)0) {
                std::cout<<"Prep for block" << preparing_block << " in iteration " << blocknum << std::endl;
                prepare_parallel<T, sa_index>(s, part_length, &r, SA, t, L_Type, thread_count, preparing_block);
            }
            
            if (inducing_block >= (ssize)0) {
                std::cout<<"L-In for block" << inducing_block << " in iteration " << blocknum << std::endl;
                induce_L_Types_Pipelined<T, sa_index>(s, SA, buckets, t, inducing_block, r, w, part_length);
            }
            
            if (updating_block >= (ssize)0) {
                std::cout<<"Upda for block" << updating_block << " in iteration " << blocknum << std::endl;
                update_parallel<sa_index>(thread_count, part_length, &w, SA);
            }
        }
        
        /*
        for (size_t blocknum = 0; blocknum <= thread_count; blocknum++)
        {
            std::cout << "loop for block " << blocknum << std::endl;

            // Parallel Preparation Phase
            // TODO: Wait until preparing is fixed, give blocknum as parameter
            prepare_parallel<T, sa_index>(s, part_length, &r, SA, t, L_Type, thread_count, blocknum);

            induce_L_Types_Pipelined<T, sa_index>(s, SA, buckets, t, blocknum, r, w, part_length);

            // Parallel Updating Phase

            update_parallel<sa_index>(thread_count, part_length, &w, SA);
        }
        */

        // induce_L_Types<T, sa_index>(s, buckets, t, K, false, SA);
        std::cout << "finished inducing L Types" << std::endl;
        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);
        std::cout << "finished inducing S Types" << std::endl;

        for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA after final Inducing FIN :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        tdc::StatPhase parallel_sais("Main Phase");
        std::cout << std::endl << std::endl;
        if (text.size() > 1) {
            run_saca<string_span, sa_index>(text, out_sa, alphabet.size_with_sentinel());
        }
    }
};
} // namespace sacabench::parallel_sais
