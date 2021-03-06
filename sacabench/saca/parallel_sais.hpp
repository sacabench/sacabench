/*******************************************************************************
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
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

#include <util/saca.hpp>
#include "saca/sais.hpp"

#include <chrono>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::parallel_sais {

using namespace sacabench::util;
using std::chrono::steady_clock;

class parallel_sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "PARALLEL_SAIS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Nong, Zhang and Chan";    
    
    static const int L_Type = 0;
    static const int S_Type = 1;

    static const int parallel_min_size = 5000000; // SAIS is usually still faster on inputs of the size of ~5MB

    static uint8_t getBit(std::vector<uint8_t>& t, int index) {
        return (t[index/8] >> (7-(index & 0x7)) & 0x1);
    }   

    static void setBit(std::vector<uint8_t>& t, int index, int value) {
        t[index/8] = t[index/8] | (value & 0x1) << (7-(index & 0x7));
    }

    static bool is_LMS(std::vector<uint8_t>& t, ssize position) {
        return (position > 0) && (getBit(t, position) == S_Type && getBit(t, position - 1) == L_Type);
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
    static void compute_types(std::vector<uint8_t>& t, T s, span<size_t> thread_border, span<bool> thread_info, size_t thread_count) {

        ssize part_length = s.size() / thread_count;
        part_length -= part_length % (sizeof(uint8_t) * 8);
        ssize rest_length = (s.size() - (thread_count - 1) * part_length);
        
        // std::cout << "pl" << part_length << std::endl;
        // std::cout << "rl" << rest_length << std::endl;
        DCHECK_EQ(part_length % 8, 0);

        for (size_t i = 0; i < thread_border.size() - 1; i++) 
        { 
            thread_border[i] = part_length; 
        }

        thread_border[thread_count - 1] = rest_length;

        setBit(t, s.size() - 1, S_Type);

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                #pragma omp task shared(t)
                compute_types_first_pass<T>(t, s, i * part_length, part_length, i, thread_border, thread_info);
            }
            else {
                #pragma omp task shared(t)
                compute_types_first_pass<T>(t, s, i * part_length, rest_length, i, thread_border, thread_info);
            }
        }

        #pragma omp taskwait

        // if many threads were not able to classify, use the last thread that has borderinfo for all the others
        for (ssize i = thread_count - 2; i >= 0; i--) {
            if (thread_border[i] == 0) {
                thread_info[i] = thread_info[i + 1];
            }
        }

        for (size_t i = 0; i < thread_count; i++) {
            if (i < thread_count - 1) {
                #pragma omp task shared(t)
                compute_types_second_pass<T>(t, i * part_length, part_length, i, thread_border, thread_info);
            }
            else {
                #pragma omp task shared(t)
                compute_types_second_pass<T>(t, i * part_length, rest_length, i, thread_border, thread_info);
            }
        }
        
        #pragma omp taskwait
    }
    
    template <typename T>
    static void compute_types_first_pass(std::vector<uint8_t>& t, T s, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
        // std::cout << "running " << omp_get_thread_num() << std::endl;

        // first pass - classify all elements that are possible to be classified within the thread
        for (ssize i = len - 1; i >= 0; i--) {           
            if ((size_t)i + offset + 1 < s.size()) {
                if (s[i + offset + 1] < s[i + offset]) {
                    // std::cout << omp_get_thread_num() << " set " << ( i + offset ) << " to L" << std::endl;
                    setBit(t, i + offset, L_Type);
                }
                else if (s[i + offset + 1] > s[i + offset]) {
                    // std::cout << omp_get_thread_num() << " set " << ( i + offset ) << " to S" << std::endl;
                    setBit(t, i + offset, S_Type);
                }
                else {
                    // do not use types from another thread as we do not know if they are already calculated
                    if (((size_t)i + 1 < len && thread_border[thread_id] != (size_t)i + 1) || (thread_id == thread_border.size() - 1)) {
                        // std::cout << omp_get_thread_num() << " set " << ( i + offset ) << " to a copy of " << (getBit(t, i + offset + 1) == L_Type ? "L" : "S") << std::endl;
                        setBit(t, i + offset, getBit(t, i + offset + 1));
                    }
                    else {
                        thread_border[thread_id] = i;
                    }
                }
            }
        }

        if (thread_border[thread_id] != 0) {
            thread_info[thread_id] = getBit(t, offset);
        }
    }

    template <typename T>
    static void compute_types_second_pass(std::vector<uint8_t>& t, size_t offset, size_t len, size_t thread_id, span<size_t> thread_border, span<bool> thread_info) {
        // second pass - use info of threads what the type of their border character was
        for (size_t i = thread_border[thread_id]; i < len; i++) {
            setBit(t, i + offset, thread_info[thread_id + 1]);
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
                                 span<sa_index> SA, std::vector<uint8_t>& t, bool suffix_type, size_t blocknum){

        // overwrite readbuffer with NULLs
        for (ssize i = 0; i < (ssize)r.size(); i++) {
            r[i].first = (sa_index)(0);
            r[i].second = (sa_index)(-1);
        }

        #pragma omp taskgroup
        {

            #pragma omp parallel for schedule(static, 102400) shared(t)
            for (size_t i = 0; i < r.size(); i++)
                {
                    prepare<T, sa_index>(s, part_length, r, SA, t, suffix_type, blocknum, i);
                }
        }
    }

    template <typename T, typename sa_index>
    static void prepare(T s, ssize part_length, span<std::pair<sa_index, sa_index>> r, span<sa_index> SA, std::vector<uint8_t>& t, bool suffix_type, size_t k, size_t i){

        size_t j = 0;
        sa_index pos;
        sa_index chr;

        j = (k*part_length)+i; 
        if(j < (size_t)SA.size() && SA[j]!= static_cast<sa_index>(-1)){
            pos = SA[j]-static_cast<sa_index>(1);
            if(pos >=static_cast<sa_index>(0) && pos!=static_cast<sa_index>(-1) && pos < SA.size() && getBit(t, pos) == suffix_type){
                chr = s[pos];
                r[i] = std::make_pair(chr, pos);
            }
        }
    }


    // Induce L_Types for Block B_ks
    // k = blocknum
    template <typename T, typename sa_index>
    static void induce_L_Types_Pipelined(T s, span<sa_index> SA, span<sa_index> buckets, std::vector<uint8_t>& t, size_t blocknum,
        span<std::pair<sa_index, sa_index>> r, span<std::pair<sa_index, sa_index>> w, ssize part_length, size_t *w_count) {

        // translate: translates the position in the block to global pos
        // w_count: pointer for the next free entry in write-buffer w
        sa_index translate = (sa_index)(blocknum*part_length);
        sa_index chr;

        for (ssize i = 0; i < (ssize)part_length && (size_t)i+translate < SA.size(); i++)
        {
            ssize pos = ((ssize)SA[(sa_index)i + translate] - 1);

            if ((pos+1) >= (ssize)(0) && pos >= (ssize)0 && pos < (ssize)SA.size() && getBit(t, pos) == L_Type)
            {
                if (r[i].first == (sa_index)(0))
                {
                    chr = (sa_index)s[(sa_index)pos];
                }
                else
                {
                    chr = (sa_index)r[i].first;
                    r[i].first = (sa_index)0;
                    r[i].second = (sa_index)(-1);
                }

                sa_index idx = buckets[chr];
                buckets[chr]++;

                // if idx is in Block k or Block k+1
                if (translate <= idx && idx <= translate + (sa_index)(2 * part_length)) {
                    SA[idx] = (sa_index)pos;
                }
                else {
                    w[(*w_count)++] = std::make_pair((sa_index)idx, (sa_index)pos);
                }

            }
        }

    }

    // Induce S Types for Block B_ks
    // k = blocknum
    template <typename T, typename sa_index>
    static void induce_S_Types_Pipelined(T s, span<sa_index> SA, span<sa_index> buckets, std::vector<uint8_t>& t, size_t blocknum,
        span<std::pair<sa_index, sa_index>> r, span<std::pair<sa_index, sa_index>> w, ssize part_length, size_t *w_count) {

        // translate: translates the position in the block to global pos
        // w_count: pointer for the next free entry in write-buffer w
        sa_index translate = (sa_index)(blocknum*part_length);
        sa_index chr;

        const auto end = std::min(part_length, (ssize)(SA.size() - translate)) - 1;

        for (ssize i = end; i >= 0; --i)
        {
            ssize pos = (ssize)SA[i + translate] - 1;

            if (pos+1 >= (ssize)(0) && pos >= (ssize)0 && pos < (ssize)SA.size() && getBit(t, pos) == S_Type)
            {
                if (r[i].first == (sa_index)0)
                    chr = (sa_index)s[(sa_index)pos];
                else {
                    chr = r[i].first;
                    r[i].first = (sa_index)0;
                    r[i].second = (sa_index)(-1);
                }

                sa_index idx = --buckets[chr];


                // if idx is in Block k-1 or Block k
                if (ssize(translate) - 2 * part_length <= ssize(idx) && idx <= i + translate) {
                    SA[idx] = (sa_index)pos;
                }
                else {
                    w[(*w_count)++] = std::make_pair(idx, (sa_index)pos);
                }
            }
        }
    }


        // Updating and writing into the SuffixArray, w needs to be properly connected to the rest of the code now
    template <typename sa_index>
    static void update_SA(span<std::pair<sa_index, sa_index>> w, span<sa_index> SA, size_t i) {

        if (w[i].second != static_cast<sa_index>(-1) && w[i].first != static_cast<sa_index>(0)) {

            SA[w[i].first] = w[i].second;
            w[i].first = (sa_index)(0);
            w[i].second = (sa_index)(-1);
        }

    }
   

    template <typename sa_index>
    static void update_parallel(span<std::pair<sa_index, sa_index>> w, span<sa_index> SA, size_t *w_count) {

        size_t max = *w_count + 1;

        #pragma omp taskgroup
        {
            #pragma omp parallel for schedule(static, 102400)
            for (size_t i = 0; i < w.size(); i++) {
                if (i > max)
                    i = w.size();
                else
                    update_SA<sa_index>(w, SA, i);
            }
        }

        *w_count = 0;
    }

    template <typename T, typename sa_index>
    static void pipelined_Inducing(T s, span<sa_index> SA, std::vector<uint8_t>& t, span<sa_index> buckets, size_t K, size_t thread_count,
        span<std::pair<sa_index, sa_index>> r1, span<std::pair<sa_index, sa_index>> r2, span<std::pair<sa_index, sa_index>> w1, span<std::pair<sa_index, sa_index>> w2, ssize part_length, bool type) {

        generate_buckets<T, sa_index>(s, buckets, K, type);

        ssize preparing_block = -1;
        ssize inducing_block = -2;
        ssize updating_block = -3;

        ssize blocknum = 0;

        size_t write_amount_1 = 0;
        size_t write_amount_2 = 0;

        // double timePreparing = 0;
        // double timeInducing = 0;
        // double timeUpdating = 0;


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

            #pragma omp taskgroup
            {
                if (preparing_block >= (ssize)0) {
                    ssize cur_prepare_block = ((type == L_Type) ? preparing_block : (thread_count - preparing_block));

                    #pragma omp task default(shared)
                    {
                        // auto start = steady_clock::now();

                        auto& r = cur_prepare_block % 2 == 0 ? r1 : r2;
                        prepare_parallel<T, sa_index>(s, part_length, r, SA, t, type, cur_prepare_block);

                        // auto end = steady_clock::now();
                        // double timeAddition = ((end - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
                        // timePreparing += timeAddition;
                        // std::cout << "prep: " << timeAddition * 1000 << std::endl;
                    }
                }

                if (inducing_block >= (ssize)0) {
                    if (type == L_Type)
                    {
                        #pragma omp task default(shared)
                        {
                            // auto start = steady_clock::now();

                            auto& r = inducing_block % 2 == 0 ? r1 : r2;
                            auto& w = inducing_block % 2 == 0 ? w1 : w2;
                            size_t* write_amount = inducing_block % 2 == 0 ? &write_amount_1 : &write_amount_2;
                            induce_L_Types_Pipelined<T, sa_index>(s, SA, buckets, t, inducing_block, r, w, part_length, write_amount);
                        
                            // auto end = steady_clock::now();
                            // double timeAddition = ((end - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
                            // timeInducing += timeAddition;
                            // std::cout << "indu: " << timeAddition * 1000 << std::endl;
                        }
                    }
                    else
                    {
                        #pragma omp task default(shared)
                        {
                            // auto start = steady_clock::now();

                            auto& r = (thread_count - inducing_block) % 2 == 0 ? r1 : r2;
                            auto& w = (thread_count - inducing_block) % 2 == 0 ? w1 : w2;
                            size_t* write_amount = (thread_count - inducing_block) % 2 == 0 ? &write_amount_1 : &write_amount_2;
                            induce_S_Types_Pipelined<T, sa_index>(s, SA, buckets, t, (thread_count - inducing_block), r, w, part_length, write_amount);
                        
                            // auto end = steady_clock::now();
                            // double timeAddition = ((end - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
                            // timeInducing += timeAddition;
                            // std::cout << "indu: " << timeAddition * 1000 << std::endl;
                        }
                    }
                }

                if (updating_block >= (ssize)0) {
                    ssize cur_update_block = ((type == L_Type) ? updating_block : (thread_count - updating_block));

                    #pragma omp task default(shared)
                    {
                        // auto start = steady_clock::now();

                        auto& w = cur_update_block % 2 == 0 ? w1 : w2;
                        size_t* write_amount = cur_update_block % 2 == 0 ? &write_amount_1 : &write_amount_2;
                        update_parallel<sa_index>(w, SA, write_amount);

                        // auto end = steady_clock::now();
                        // double timeAddition = ((end - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
                        // timeUpdating += timeAddition;
                        // std::cout << "upda: " << timeAddition * 1000 << std::endl;
                    }
                }
            }

            blocknum++;
        }

       //  std::cout << "Time Preparing: " << timePreparing*1000 << ", Time Inducing: " << timeInducing * 1000 << ", Time Updating: " << timeUpdating * 1000 << std::endl;
    }

    //template <typename T, typename sa_index>
    //static void induce_L_Types_sequential(T s, span<sa_index> buckets, std::vector<uint8_t>& t, size_t K, bool end, span<sa_index> SA) {
    //    generate_buckets<T, sa_index>(s, buckets, K, end);
    //    for (size_t i = 0; i < s.size(); i++) {
    //        ssize pre_index = SA[i] - (sa_index)1; // pre index of the ith suffix array position

    //        if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && getBit(t, pre_index) == L_Type) { // pre index is type L
    //            SA[buckets[s.at(pre_index)]++] = pre_index; // "sort" index in the bucket

    //        }
    //    }
    //}

    //template <typename T, typename sa_index>
    //static void induce_S_Types_sequential(T s, span<sa_index> buckets, std::vector<uint8_t>& t, size_t K, bool end, span<sa_index> SA) {
    //    generate_buckets<T, sa_index>(s, buckets, K, end);
    //    for (ssize i = s.size() - 1; i >= 0; i--) {
    //        ssize pre_index = SA[i] - (sa_index)1;

    //        if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && getBit(t, pre_index) == S_Type) {
    //            SA[--buckets[s.at(pre_index)]] = pre_index;
    //        }
    //    }
    //}

    template <typename T, typename sa_index>
    static void run_saca(T s, span<sa_index> SA, size_t K, container<std::pair<sa_index, sa_index>> &buff, size_t depth) {

        size_t beta = 10000000; // Buffer size of 10MB

        if (parallel_min_size > s.size())
        {
            sacabench::sais::sais::run_saca<T, sa_index>(s, SA, K);
            return;
        }

        container<sa_index> buckets = make_container<sa_index>(K);
        std::vector<uint8_t> t(s.size() / 8 + 1);
        // size_t thread_count = std::thread::hardware_concurrency();
        // thread_count = std::min(thread_count, s.size() - 1);

        ssize part_length = beta;
        size_t block_amount = std::max((size_t)1,(size_t)(s.size() / part_length));
        // ssize rest_length = (s.size() % part_length);

        container<size_t> thread_border = make_container<size_t>(block_amount);
        container<bool> thread_info = make_container<bool>(block_amount);
               

        // for very small inputs, so that we can always assure that rest_length <= part_length
        /*while (rest_length > part_length && thread_count > 1)
        {
            thread_count--;
            part_length = s.size() / thread_count;
            rest_length = (s.size() - (thread_count - 1) * part_length);
        }*/
       
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

        compute_types(t, s, thread_border, thread_info, block_amount);

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

        pipelined_Inducing(s, SA, t, buckets.slice(), K, block_amount, r1, r2, w1, w2, part_length, L_Type);
        pipelined_Inducing(s, SA, t, buckets.slice(), K, block_amount, r1, r2, w1, w2, part_length, S_Type);

        // Prepare and Check for Recursion Call #############################################################
        
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
                    getBit(t, current_LMS + j) != getBit(t, previous_LMS + j)) {
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
            run_saca<span<sa_index const>, sa_index>(s1, sa_, name, buffers.size() > buff.size() ? buffers : buff, depth+1);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        // FINAL INDUCING ##########################################################

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

        pipelined_Inducing(s, SA, t, buckets.slice(), K, block_amount, r1, r2, w1, w2, part_length, L_Type);
        pipelined_Inducing(s, SA, t, buckets.slice(), K, block_amount, r1, r2, w1, w2, part_length, S_Type);
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        // tdc::StatPhase parallel_sais("Main Phase");
        // std::cout << std::endl << std::endl;
        container<std::pair<sa_index, sa_index>> buffers = make_container<std::pair<sa_index, sa_index>>(0);
        if (text.size() > 1) {
            #pragma omp parallel
            #pragma omp master
            run_saca<string_span, sa_index>(text, out_sa, alphabet.size_with_sentinel(), buffers, 0);
        }
    }
};
} // namespace sacabench::parallel_sais
