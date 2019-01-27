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

    static bool is_LMS(std::vector<bool>& t, ssize position) {
        return (position > 0) && (t[position] == S_Type && t[position - 1] == L_Type);
    }

    template <typename T>
    static void compute_types_sequential(std::vector<bool>& t, T s) {
        t[s.size() - 1] = S_Type;

        for (ssize i = s.size() - 2; i >= 0; i--) {
            if (s.at(i + 1) < s.at(i)) {
                t[i] = L_Type;
            }
            else if (s.at(i + 1) > s.at(i)) {
                t[i] = S_Type;
            }
            else {
                t[i] = t[i + 1];
            }
        }
    }

    template <typename T>
    static void compute_types(std::vector<bool>& t, T s, span<size_t> thread_border, span<bool> thread_info, ssize part_length, ssize rest_length, size_t thread_count) {

        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < thread_border.size() - 1; i++) 
        { 
            thread_border[i] = part_length; 
        }

        thread_border[thread_count - 1] = rest_length;

        t[s.size() - 1] = S_Type;

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                threads.push_back(std::thread(compute_types_first_pass<T>, std::ref(t), s, i * part_length, part_length, i, thread_border, thread_info));
            }
            else {
                threads.push_back(std::thread(compute_types_first_pass<T>, std::ref(t), s, i * part_length, rest_length, i, thread_border, thread_info));
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
                threads.push_back(std::thread(compute_types_second_pass<T>, std::ref(t), i * part_length, part_length, i, thread_border, thread_info));
            }
            else {
                threads.push_back(std::thread(compute_types_second_pass<T>, std::ref(t), i * part_length, rest_length, i, thread_border, thread_info));
            }
        }
        

        for (auto& t : threads) {
            t.join();
        }
    }
    
    template <typename T>
    static void compute_types_first_pass(std::vector<bool>& t, T s, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
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
    static void compute_types_second_pass(std::vector<bool>& t, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
        // second pass - use info of threads what the type of their border character was
        for (size_t i = thread_border[thread_id]; i < len; i++) {
            t[i + offset] = thread_info[thread_id + 1];
        }
    }    
    
    template <typename T, typename sa_index>
    static void generate_buckets(T s, span<sa_index> buckets, size_t K, bool end) {
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
    static void prepare_parallel(T s, ssize part_length, span<std::pair<sa_index, sa_index>> r,
                                 span<sa_index> SA, std::vector<bool>& t, bool suffix_type, size_t thread_count, size_t blocknum){

        // overwrite readbuffer with NULLs
        for (ssize i = 0; i < (ssize)r.size(); i++) {
            r[i].first = (sa_index)(0);
            r[i].second = static_cast<sa_index>(-1);
        }

            std::vector<std::thread> threads;

        for (size_t i = 0; i < r.size() && i < thread_count - 1; i++) {
                threads.push_back(std::thread(prepare<T,sa_index>, s, part_length, r, SA, std::ref(t), suffix_type, blocknum, i));
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    template <typename T, typename sa_index>
    static void prepare(T s, ssize part_length, span<std::pair<sa_index, sa_index>> r, span<sa_index> SA, std::vector<bool>& t, bool suffix_type, size_t k, size_t i){
        // std::cout << "Started preparing " << i << std::endl;

        size_t j = 0;
        sa_index pos;
        sa_index chr;

        j = (k*part_length)+i; 
        // std::cout << "Prepare for pos j = " << j << std::endl;
        if(j < (size_t)SA.size() && SA[j]!= static_cast<sa_index>(-1)){
            pos = SA[j]-static_cast<sa_index>(1);
            if(pos >=static_cast<sa_index>(0) && pos!=static_cast<sa_index>(-1) && pos < t.size() && t[pos] == suffix_type){
                chr = s[pos];
                r[i] = std::make_pair(chr, pos);
                // std::cout << "Write Tuple <" << (ssize)chr << ", " << (ssize)pos << "> to r, j = " << j << ", k = " << k << ", i = " << i << ", pl = " << part_length << std::endl;
            }
        }

        // std::cout << "Finished preparing " << i << std::endl;

    }

    //template <typename T, typename sa_index>
    //static void induce_L_Types(T s, span<sa_index> buckets, span<bool> t, size_t K,
    //                           bool end, span<sa_index> SA) {
    //    generate_buckets<T, sa_index>(s, buckets, K, end);
    //    for (size_t i = 0; i < s.size(); i++) {
    //        ssize pre_index =
    //            SA[i] - (sa_index)1; // pre index of the ith suffix array position

    //        if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 &&
    //            t[pre_index] == L_Type) { // pre index is type L
    //            SA[buckets[s.at(pre_index)]++] =
    //                pre_index; // "sort" index in the bucket
    //        }
    //    }
    //}

        //template <typename T, typename sa_index>
    //static void induce_S_Types(T s, span<sa_index> buckets, span<bool> t, size_t K,
    //                           bool end, span<sa_index> SA) {
    //    generate_buckets<T, sa_index>(s, buckets, K, end);
    //    for (ssize i = s.size() - 1; i >= 0; i--) {
    //        ssize pre_index = SA[i] - (sa_index)1;

    //        if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && t[pre_index] == S_Type) {
    //            SA[--buckets[s.at(pre_index)]] = pre_index;
    //        }
    //    }
    //}


    // Induce L_Types for Block B_ks
    // k = blocknum
    template <typename T, typename sa_index>
    static void induce_L_Types_Pipelined(T s, span<sa_index> SA, span<sa_index> buckets, std::vector<bool>& t, size_t blocknum,
        span<std::pair<sa_index, sa_index>> r, span<std::pair<sa_index, sa_index>> w, ssize part_length, size_t *w_count) {

        // translate: translates the position in the block to global pos
        // w_count: pointer for the next free entry in write-buffer w
        sa_index translate = (sa_index)(blocknum*part_length);
        sa_index chr;

        // std::cout << "At the beginning of L Inducing w_count is " << *w_count << std::endl;

        for (ssize i = 0; i < (ssize)part_length && i+(ssize)translate < (ssize)SA.size(); i++)
        {
            ssize pos = ((ssize)SA[i + translate] - 1);
            //std::cout << "i: " << (sa_index)i << ", pos: " << pos << ", i+trans: " << (i+translate) << std::endl;

            if ((ssize)SA[(sa_index)i + translate] >= (ssize)(0) && pos >= (ssize)0 && pos < (ssize)SA.size() && t[pos] == L_Type)
            {

                if (r[i].first == static_cast<sa_index>(0))
                    chr = (sa_index)s[(sa_index)pos];
                else
                    chr = (sa_index)r[i].first;

                //std::cout << "chr: " << (sa_index)chr << std::endl;

                sa_index idx = buckets[chr];
                buckets[chr]++;

                //std::cout << "idx: " << (sa_index)idx << std::endl;

                // if idx is in Block k or Block k+1
                if (translate <= idx && idx <= translate + (sa_index)(2 * part_length) && idx < SA.size()) {
                    SA[idx] = (sa_index)pos;
                    //std::cout << "Directly written " << (size_t)pos << " to pos " << (size_t)idx << std::endl;
                }
                else if (idx < SA.size()) {
                    w[(*w_count)++] = std::make_pair((sa_index)idx, (sa_index)pos);
                    //std::cout << "Inserted Touple <" << (ssize)idx << ", " << (ssize)pos << "> at w_count " << (ssize)(w_count - 1) << " in w" << std::endl;
                    //std::cout << "Insertion Check <" << (ssize)(w[w_count-1].first) << ", " << (ssize)(w[w_count - 1].second) << "> at w_count " << (ssize)(w_count - 1) << " in w" << std::endl;
                }

            }
        }

        // std::cout << "At the end of L Inducing w_count is " << *w_count << std::endl;

    }

    // Induce S Types for Block B_ks
    // k = blocknum
    template <typename T, typename sa_index>
    static void induce_S_Types_Pipelined(T s, span<sa_index> SA, span<sa_index> buckets, std::vector<bool>& t, size_t blocknum,
        span<std::pair<sa_index, sa_index>> r, span<std::pair<sa_index, sa_index>> w, ssize part_length, size_t *w_count) {

        // translate: translates the position in the block to global pos
        // w_count: pointer for the next free entry in write-buffer w
        sa_index translate = (sa_index)(blocknum*part_length);
        sa_index chr;

        const auto end = std::min(part_length, (ssize)(SA.size() - translate)) - 1;

        for (ssize i = end; i >= 0; --i)
        {
            ssize pos = (ssize)SA[i + translate] - 1;
            // std::cout << "i: " << (sa_index)i << ", pos: " << pos << ", i+trans: " << (i+translate) << std::endl;

            if ((ssize)SA[(sa_index)i + translate] >= (ssize)0 && pos >= (ssize)0 && pos < (ssize)SA.size() && t[pos] == S_Type)
            {

                if (r[i].first == (sa_index)0)
                    chr = (sa_index)s[(sa_index)pos];
                else
                    chr = r[i].first;

                // std::cout << "chr: " << (sa_index)chr << std::endl;

                sa_index idx = --buckets[chr];

                // std::cout << "idx: " << (sa_index)idx << std::endl;

                // if idx is in Block k or Block k+1
                if (translate <= idx && idx <= translate + (sa_index)(2 * part_length) && idx < SA.size()) {
                    SA[idx] = (sa_index)pos;
                }
                else if (idx < SA.size()) {
                    w[(*w_count)++] = std::make_pair(idx, (sa_index)pos);
                }

            }
        }
    }


    // Updating and writing into the SuffixArray, w needs to be properly connected to the rest of the code now
    template <typename sa_index>
    static void update_SA(ssize part_length, span<std::pair<sa_index, sa_index>> w, span<sa_index> SA, size_t thread_id, size_t *w_count) {
        
        // std::cout << "Started updating " << thread_id << std::endl;
        ssize offset = thread_id * part_length;

        for (ssize i = offset; i < part_length + offset; i++) {

            // std::cout << " Update for pos i = " << i << " " << std::endl;
            if ((size_t)i > *w_count)
                break;

            if (i < (ssize)w.size() && w[i].first != static_cast<sa_index>(0) && w[i].second != static_cast<sa_index>(-1)) {

                if((size_t)w[i].first < SA.size())
                {
                    SA[w[i].first] = w[i].second;
                }
                else
                {
                    std::cout << "### ERROR IN W-BUFFER ###" << std::endl;
                }


                w[i].first = (sa_index)(0);
                w[i].second = (sa_index)(-1);
            }
        }

        // std::cout << "Finished updating " << thread_id << std::endl;
    }

    template <typename sa_index>
    static void update_parallel(size_t thread_count, ssize part_length, span<std::pair<sa_index, sa_index>> w, span<sa_index> SA, size_t *w_count) {

        // std::cout << "At the beginning of Updating w_count is " << (*w_count) << std::endl;
        std::vector<std::thread> threads;

            for (size_t i = 0; i < thread_count && (size_t)(i*part_length) < *w_count; i++) {
                threads.push_back(std::thread(update_SA<sa_index>, part_length, w, SA, i, w_count));
            }

            for (auto& t : threads) {
                t.join();
            }

        *w_count = 0;
    }

    // Initialization of the Write Buffer, maybe can be put together with the Preparing-Phase later
    /*template <typename sa_index>
    static void init_Write_Buffer(span<std::pair<sa_index, sa_index>> w) {  
        
        for (ssize i = 0; i < (ssize)w.size(); i++) {
            w[i].first = (sa_index)(0);
            w[i].second = static_cast<sa_index>(-1);
        }
    }*/

    template <typename T, typename sa_index>
    static void pipelined_Inducing(T s, span<sa_index> SA, std::vector<bool>& t, span<sa_index> buckets, size_t K, size_t thread_count,
        span<std::pair<sa_index, sa_index>> r1, span<std::pair<sa_index, sa_index>> r2, span<std::pair<sa_index, sa_index>> w1, span<std::pair<sa_index, sa_index>> w2, ssize part_length, bool type) {

        generate_buckets<T, sa_index>(s, buckets, K, type);

        ssize preparing_block = -1;
        ssize inducing_block = -2;
        ssize updating_block = -3;

        ssize blocknum = 0;

        size_t write_amount_1 = 0;
        size_t write_amount_2 = 0;

        // To differentiate the L-Type-Inducing from the S-Type-Inducing we just invert the blocknumber we are currently handling when methods are being called
        // (Since the only difference between L- and S-Type inducing is basically in which order the text is being read and which types to look for while reading)

        while (blocknum <= (ssize)thread_count + 2) {

            // Preparing, inducing and updating. Note, that in the first iteration, only the first block is being prepared.
            if (blocknum < (ssize)thread_count + 1) {
                preparing_block++;
                inducing_block++;
                updating_block++;
            }

            // Preparing for all block finished
            if (blocknum == (ssize)thread_count + 1) {
                preparing_block = -1;
                inducing_block++;
                updating_block++;
            }

            // Preparing and inducing finished
            if (blocknum == (ssize)thread_count + 2) {
                preparing_block = -1;
                inducing_block = -1;
                updating_block++;
            }


            if (updating_block >= (ssize)0) {
                ssize cur_update_block = ((type == L_Type) ? updating_block : (thread_count - updating_block));
                // ssize cur_blocknum = ((type == L_Type) ? blocknum : (thread_count - blocknum + 2));

                auto& w = cur_update_block % 2 == 0 ? w1 : w2;
                size_t& write_amount = cur_update_block % 2 == 0 ? write_amount_1 : write_amount_2;

                // std::cout << "Upda for block" << cur_update_block << " in iteration " << cur_blocknum << std::endl;
                update_parallel<sa_index>(thread_count, part_length, w, SA, &write_amount);
            }

            if (inducing_block >= (ssize)0) {

                if (type == L_Type) {
                    auto& r = inducing_block % 2 == 0 ? r1 : r2;
                    auto& w = inducing_block % 2 == 0 ? w1 : w2;
                    size_t& write_amount = inducing_block % 2 == 0 ? write_amount_1 : write_amount_2;

                    // std::cout << "L-In for block" << inducing_block << " in iteration " << blocknum << std::endl;
                    induce_L_Types_Pipelined<T, sa_index>(s, SA, buckets, t, inducing_block, r, w, part_length, &write_amount);
                }
                else
                {
                    auto& r = (thread_count - inducing_block) % 2 == 0 ? r1 : r2;  
                    auto& w = (thread_count - inducing_block) % 2 == 0 ? w1 : w2;
                    size_t& write_amount = (thread_count - inducing_block) % 2 == 0 ? write_amount_1 : write_amount_2;

                    // std::cout << "S-In for block" << (thread_count - inducing_block) << " in iteration " << (thread_count - blocknum)+2 << std::endl;
                    induce_S_Types_Pipelined<T, sa_index>(s, SA, buckets, t, (thread_count - inducing_block), r, w, part_length, &write_amount);
                }

                // std::cout << "write_1: " << write_amount_1 << ", write_2: " << write_amount_2 << std::endl;
            }

            if (preparing_block >= (ssize)0) {
                ssize cur_prepare_block = ((type == L_Type) ? preparing_block : (thread_count - preparing_block));
                // ssize cur_blocknum = ((type == L_Type) ? blocknum : (thread_count - blocknum + 2));

                auto& r = cur_prepare_block % 2 == 0 ? r1 : r2;

                // std::cout << "Prep for block" << cur_prepare_block << " in iteration " << cur_blocknum << std::endl;
                prepare_parallel<T, sa_index>(s, part_length, r, SA, t, type, thread_count, cur_prepare_block);
            }

            blocknum++;

        }

    }


    template <typename T, typename sa_index>
    static void run_saca(T s, span<sa_index> SA, size_t K, container<std::pair<sa_index, sa_index>> &buff) {

        /*for (size_t i = 0; i < s.size(); i++)
        {
            if (i == 0)
                std::cout << "    Text : ";

            std::cout << (ssize)(s[i]) << " ";
        }

        std::cout << std::endl;

        std::cout << "s.size : " << s.size() << std::endl;*/


        container<sa_index> buckets = make_container<sa_index>(K);
        std::vector<bool> t(s.size());
        container<size_t> thread_border = make_container<size_t>(std::thread::hardware_concurrency());
        container<bool> thread_info = make_container<bool>(std::thread::hardware_concurrency());
        
        // Prepare blocks for parallel computing
        size_t thread_count = std::thread::hardware_concurrency();
        thread_count = std::min(thread_count, s.size() - 1);
        // thread_count = 1;
        ssize part_length = s.size() / thread_count;
        ssize rest_length = (s.size() - (thread_count - 1) * part_length);

        // for very small inputs, so that we can always assure that rest_length <= part_length
        while (rest_length > part_length && thread_count > 1)
        {
            thread_count--;
            part_length = s.size() / thread_count;
            rest_length = (s.size() - (thread_count - 1) * part_length);
        }

        // Read/Write Buffer for the pipeline, one single buffer cut into 4 seperate ones, each with length "part_length + 1"
        container<std::pair<sa_index, sa_index>> buffers;
        span<std::pair<sa_index, sa_index>> r1;
        span<std::pair<sa_index, sa_index>> w1;
        span<std::pair<sa_index, sa_index>> r2;
        span<std::pair<sa_index, sa_index>> w2;

        if (buff.size() >= (size_t)(4 * part_length + 4)) 
        {
            r1 = buff.slice(0, 1 * part_length + 1);
            w1 = buff.slice(1 * part_length + 1, 2 * part_length + 2);
            r2 = buff.slice(2 * part_length + 2, 3 * part_length + 3);
            w2 = buff.slice(3 * part_length + 3, 4 * part_length + 4);
        }
        else
        {
            buffers = make_container<std::pair<sa_index, sa_index>>(4 * part_length + 4);
            r1 = buffers.slice(0, 1 * part_length + 1);
            w1 = buffers.slice(1 * part_length + 1, 2 * part_length + 2);
            r2 = buffers.slice(2 * part_length + 2, 3 * part_length + 3);
            w2 = buffers.slice(3 * part_length + 3, 4 * part_length + 4);
        }

        // compute_types(t, s, thread_border, thread_info, part_length, rest_length, thread_count);

        compute_types_sequential(t, s);

        // std::cout << "thread_count: " << thread_count << ", part_length: " << part_length << ", rest_length: " << rest_length << std::endl;

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

        pipelined_Inducing(s, SA, t, buckets.slice(), K, thread_count, r1, r2, w1, w2, part_length, L_Type);
        pipelined_Inducing(s, SA, t, buckets.slice(), K, thread_count, r1, r2, w1, w2, part_length, S_Type);

        // Recursion Call #############################################################
        
        // because we have at most n/2 LMS, we can store the sorted indices in
        // the first half of the SA
        ssize n1 = 0;
        for (size_t i = 0; i < s.size(); i++) {
            const ssize sa_idx = SA[i] == sa_index(-1) ? ssize(-1) : ssize(SA[i]);
            if (is_LMS(t, sa_idx) == 1 || s.size() == 1) {
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
            run_saca<span<sa_index const>, sa_index>(s1, sa_, name, buffers.size() > buff.size() ? buffers : buff);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        // std::cout << "start final inducing..." << std::endl;

        // FINAL INDUCING ##########################################################

        // induce the final SA
        generate_buckets<T, sa_index>(s, buckets, K, true);
        // std::cout << "buckets generated" << std::endl;
        size_t j;
        for (size_t i = 1, j = 0; i < s.size(); i++) {
            if (is_LMS(t, i)) {
                s1[j++] = i;
            }

        }
        // std::cout << "start final inducing1..." << std::endl;
        for (ssize i = 0; i < n1; i++) {
            SA[i] = s1[SA[i]];
        }
        // std::cout << "start final inducing2..." << std::endl;
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        // std::cout << "start final inducing3..." << std::endl;
        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = (sa_index)-1;
            SA[--buckets[s.at(j)]] = j;
        }

        // std::cout << "finished inducing LMS..." << std::endl;

        
        pipelined_Inducing(s, SA, t, buckets.slice(), K, thread_count, r1, r2, w1, w2, part_length, L_Type);
        pipelined_Inducing(s, SA, t, buckets.slice(), K, thread_count, r1, r2, w1, w2, part_length, S_Type);

        // generate_buckets<T, sa_index>(s, buckets, K, true);

        // // Main Loop for each block, need to add shifted parallelization for blocks later
        //for (ssize blocknum = thread_count; blocknum >= 0; blocknum--)
        //{
        //    // Parallel Preparation Phase
        //    prepare_parallel<T, sa_index>(s, part_length, &r1, SA, t, S_Type, thread_count, blocknum);
        //    induce_S_Types_Pipelined<T, sa_index>(s, SA, buckets, t, blocknum, r1, w1, part_length);

        //    // Parallel Updating Phase
        //    update_parallel<sa_index>(thread_count, part_length, &w1, SA);
        //}

        /*for (sa_index i = 0; i < s.size(); i++)
        {
            if (i == (sa_index)0)
                std::cout << "SA after final Inducing FIN :   [ ";

            std::cout << (ssize)SA[i] << " ";

            if (i == (sa_index)SA.size() - (sa_index)1)
                std::cout << "]" << std::endl;
        }*/
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        // tdc::StatPhase parallel_sais("Main Phase");
        // std::cout << std::endl << std::endl;
        container<std::pair<sa_index, sa_index>> buffers = make_container<std::pair<sa_index, sa_index>>(0);
        if (text.size() > 1) {
            run_saca<string_span, sa_index>(text, out_sa, alphabet.size_with_sentinel(), buffers);
        }
    }
};
} // namespace sacabench::parallel_sais
