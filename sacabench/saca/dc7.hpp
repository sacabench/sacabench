/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <array>
#include <math.h>
#include <ostream>
#include <queue>
#include <string>
#include <tuple>
#include <util/container.hpp>
#include <util/induce_sa_dc.hpp>
#include <util/merge_sa_dc.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#define OUTPUT false 

namespace sacabench::dc7 {

class dc7 {
public:
    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t alphabet_size,
                             util::span<sa_index> out_sa) {

        if (text.size() == 0){}
        else if(text.size() == 1) out_sa[0] = 0;
        else{

            auto modified_text =
                sacabench::util::make_container<sacabench::util::character>(
                    text.size() + 7);

            for (size_t i = 0; i < text.size() + 7; i++) {
                if (i < text.size()) {
                    modified_text[i] = text[i];
                } else {
                    modified_text[i] = '\0';
                }
            }

            construct_sa_dc7<sa_index, false, sacabench::util::character>(
                modified_text, alphabet_size, out_sa);
        }
    }

private:
    template <typename C, typename T, typename S>
    static void determine_tuples(const T& INPUT_STRING, S& tuples_124) {

        size_t n = INPUT_STRING.size() - 6;
        //DCHECK_MSG(tuples_124.size() ==3 * (n) / 7 + 1 - (((n % 7) == 0)) - (((n % 7) == 1))),
        //           "tuples_124 must have the length (3*INPUT_STRING.size()/7)");

        
        // Container to store all tuples with the same length as tuples_124
        // Tuples contains six chararcters and the start position
        // i mod 3 = 1 || 2 || 4
        auto tuples_124_to_be_sorted = sacabench::util::make_container<
            std::tuple<C, C, C, C, C, C, C, size_t>>(tuples_124.size());

        size_t counter = 0;

        for (size_t i = 1; i < n; i++) {
            if (((i % 7) == 1) || ((i % 7) == 2) || ((i % 7) == 4)) {
                tuples_124_to_be_sorted[counter++] =
                    std::tuple<C, C, C, C, C, C, C, size_t>(
                        INPUT_STRING[i], INPUT_STRING[i + 1],
                        INPUT_STRING[i + 2], INPUT_STRING[i + 3],
                        INPUT_STRING[i + 4], INPUT_STRING[i + 5],
                        INPUT_STRING[i + 6], i);
            }
        }

        // TODO: sort Tupels with radix_sort
        // radix_sort(sa0_to_be_sorted, sa0);
        std::sort(tuples_124_to_be_sorted.begin(),
                  tuples_124_to_be_sorted.end());

        for (size_t i = 0; i < tuples_124_to_be_sorted.size(); i++) {
            tuples_124[i] = std::get<7>(tuples_124_to_be_sorted[i]);
        }
    }

    template <typename T, typename S>
    static void determine_leq(const T& INPUT_STRING, const S& tuples_124,
                              S& t_124, const size_t start_of_pos_2, const size_t start_of_pos_4, bool& recursion) {

        DCHECK_MSG(tuples_124.size() == t_124.size(),
                   "tuples_124 must have the same length as t_124");

        size_t n = INPUT_STRING.size() - 6;

        size_t leq_name = 1;

        if(OUTPUT){
            std::cout << "position2: " << start_of_pos_2 << " und position4: " << start_of_pos_4 << " hier nochmal ((((n-1) % 7) == 1)): " <<  ((((n-1) % 7) == 1)) << std::endl;
        }
        
        size_t pos_to_store_leq;
        for (size_t i = 0; i < tuples_124.size(); i++) {
            // set the lexicographical names at correct positions:
            //[----names at positions i mod 7 = 1----||
            // ----names at positions i mod 7 = 2----||
            // ----names at positions i mod 7 = 4----]
            if (tuples_124[i] % 7 == 1) {
                pos_to_store_leq = tuples_124[i] / 7;
            } else if (tuples_124[i] % 7 == 2) {
                pos_to_store_leq = start_of_pos_2 + (tuples_124[i] / 7);
            } else {
                pos_to_store_leq = start_of_pos_4 + (tuples_124[i] / 7);
            }
            t_124[pos_to_store_leq] = leq_name;

            if (i + 1 < tuples_124.size()) {
                if (sacabench::util::span(&INPUT_STRING[tuples_124[i]], 7) !=
                    sacabench::util::span(&INPUT_STRING[tuples_124[i + 1]],
                                          7)) {
                    leq_name++;
                } else { // if lexicographical names are not uniqe set recursion
                         // = true
                    recursion = true;
                }
            }
        }
        // TODO: Abfragen ob INPUT_STRING und tuples_124 out of bounce
    }

    template <typename S, typename I>
    static void determine_isa(const S& t_124, I& isa_124) {
        // Anfang
        // Debug-Informationen--------------------------------------
        if(OUTPUT){
        std::cout << "determine_isa" << std::endl;
        }
        // Ende
        // Debug-Informationen--------------------------------------
        DCHECK_MSG(isa_124.size() == t_124.size(),
                   "isa_124 must have the same length as t_124");

        for (size_t i = 0; i < t_124.size(); i++) {
            isa_124[t_124[i] - 1] = i + 1;
        }
    }

    template <typename S, typename I>
    static void determine_sa(const S& t_124, I& isa_124) {

        DCHECK_MSG(isa_124.size() == t_124.size(),
                   "isa_124 must have the same length as t_124");
        // Anfang
        // Debug-Informationen--------------------------------------
        if(OUTPUT){
        std::cout << "determine_sa" << std::endl;
        }
        // Ende
        // Debug-Informationen--------------------------------------
        for (size_t i = 0; i < t_124.size(); ++i) {
            isa_124[t_124[i]] = i + 1;
        }
    }

    template <typename sa_index, bool rec, typename C, typename S>
    static void construct_sa_dc7(S text, size_t alphabet_size,
                                 util::span<sa_index> out_sa) {

        const size_t n = text.size() - 6;

        
        // Anfang
        // Debug-Informationen-----------------------------------------
        if(OUTPUT){
        if (rec) {
            std::cout << std::endl
                      << "Wir befinden uns in der Rekursion" << std::endl;
        } else {
            std::cout << std::endl
                      << "Wir befinden uns in der Hauptmethode" << std::endl;
        }
        }
        // Ende
        // Debug-Informationen-------------------------------------

        // empty container which will contain indices of triplet
        // at positions i mod 3 != 0
        //auto tuples_124 = sacabench::util::make_container<size_t>(
        //    3 * (n) / 7 + ((((n - 1) % 7) != 0)));
        
        auto tuples_124 = sacabench::util::make_container<size_t>(
            3 * (n) / 7 + 1 - (((n % 7) == 0)) - (((n % 7) == 1)));
        
        //const size_t start_of_pos_2 = ((n / 7) + ((((n - 1) % 7) != 0)));
        const size_t start_of_pos_2 = tuples_124.size()/3 + ((tuples_124.size()%3)!=0);
        const size_t start_of_pos_4 = 2 * start_of_pos_2 - (((n % 7) == 2));
        
        if(OUTPUT){
        std::cout << "tuples_124.size(): " << tuples_124.size() << std::endl;
        }
        // determine positions and calculate the sorted order
        determine_tuples<C>(text, tuples_124);

        // empty SA which should be filled correctly with lexicographical
        // names of triplets
        auto t_124 = sacabench::util::make_container<size_t>(tuples_124.size());

        // Anfang
        // Debug-Informationen-------------------------------------
        if(OUTPUT){
        std::cout << "tuples:    ";
        for (size_t i = 0; i < tuples_124.size(); i++) {
            std::cout << tuples_124[i] << " ";
        }
        std::cout << std::endl;
        }
        // Ende
        // Debug-Informationen-------------------------------------

        // bool which will be set true in determine_leq if the names are not
        // unique
        bool recursion = false;

        // fill t_124 with lexicographical names
        determine_leq(text, tuples_124, t_124, start_of_pos_2, start_of_pos_4, recursion);

        // Anfang
        // Debug-Informationen-------------------------------------
        if(OUTPUT){
        std::cout << "t_124:    ";
        for (size_t i = 0; i < t_124.size(); i++) {
            std::cout << t_124[i] << " ";
        }
        std::cout << std::endl;
        }
        // Ende
        // Debug-Informationen--------------------------------------

        util::span<sa_index> sa_124 = util::span(&out_sa[0], t_124.size());

        // Anfang
        // Debug-Informationen-------------------------------------
        if(OUTPUT){
        std::cout << "vor der Rekursion" << std::endl;
        }
        // Ende
        // Debug-Informationen------------------------------------

        // run the algorithm recursivly if the names are not unique
        if (recursion) {

            // add two sentinals to the end of the text
            auto modified_text =
                sacabench::util::make_container<size_t>(t_124.size() + 7);

            for (size_t i = 0; i < t_124.size() + 7; i++) {
                if (i < t_124.size()) {
                    modified_text[i] = t_124[i];
                } else {
                    modified_text[i] = '\0';
                }
            }

            // run algorithm recursive
            construct_sa_dc7<sa_index, true, size_t>(modified_text,
            alphabet_size, sa_124);
        }

        // Anfang
        // Debug-Informationen--------------------------------------
        if(OUTPUT){
        std::cout << "nach der Rekursion" << std::endl;
        }
        // Ende
        // Debug-Informationen--------------------------------------

        // empty isa_124 which should be filled correctly with method
        // determine_isa
        auto isa_124 = sacabench::util::make_container<size_t>(0);

        // empty merge_isa_124 to be filled with inverse suffix array in format
        // for merge_sa_dc
        auto merge_isa_124 =
            sacabench::util::make_container<size_t>(tuples_124.size());

        // if in recursion use temporary sa. Otherwise t_124
        if (recursion) {

            isa_124 = sacabench::util::make_container<size_t>(sa_124.size());
            determine_sa(sa_124, isa_124);

            // Anfang
            // Debug-Informationen-----------------------------------
            if(OUTPUT){
            std::cout << "sa_124:    ";
            for (size_t i = 0; i < sa_124.size(); i++) {
                std::cout << sa_124[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "isa_124:    ";
            for (size_t i = 0; i < isa_124.size(); i++) {
                std::cout << isa_124[i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "start_of_pos_2: " << start_of_pos_2 << ", start_of_pos_4: " << start_of_pos_4 << std::endl;
            }
            // Ende
            // Debug-Informationen------------------------------------------------------------------------

            // correct the order of sa_124 with result of recursion
            for (size_t i = 0; i < tuples_124.size(); i++) {
                if (i < start_of_pos_2) {
                    tuples_124[isa_124[i] - 1] = 7 * i + 1;
                } else if(i <  start_of_pos_4){
                    tuples_124[isa_124[i] - 1] = 7 * (i-start_of_pos_2) + 2;
                }else{
                    tuples_124[isa_124[i] - 1] = 7 * (i-start_of_pos_4) + 4;
                }
            }

            // Anfang
            // Debug-Informationen------------------------------------------------------------------------
            if(OUTPUT){
            std::cout << "korrigierte triplets:    ";
            for (size_t i = 0; i < tuples_124.size(); i++) {
                std::cout << tuples_124[i] << " ";
            }
            std::cout << std::endl;
            }
            // Ende
            // Debug-Informationen------------------------------------------------------------------------

        } else {
            isa_124 = sacabench::util::make_container<size_t>(t_124.size());

            // copy in loop to suppress warnings. Will be changed in
            // optimization
            for (size_t i = 0; i < isa_124.size(); ++i) {
                isa_124[i] = t_124[i];
            }
            determine_isa(isa_124, sa_124);
        }
        if(OUTPUT){
        std::cout << "sa_124: ";
        for (size_t i = 0; i < sa_124.size(); ++i) {
            std::cout << sa_124[i] << " ";
        }
        std::cout << std::endl;
        }

        // positions i mod 7 = 0 of text
        auto t_0 =
            sacabench::util::make_container<size_t>(n / 7 + ((n % 7) != 0));
        // positions i mod 7 = 3 of text
        auto t_3 = sacabench::util::make_container<size_t>(
            (n - 3) / 7 + (((n - 3) % 7) != 0));
        // positions i mod 7 = 5 of text
        auto t_5 = sacabench::util::make_container<size_t>(
            (n - 5) / 7 + (((n - 5) % 7) != 0));
        // positions i mod 7 = 6 of text
        auto t_6 = sacabench::util::make_container<size_t>(
            (n - 6) / 7 + (((n - 6) % 7) != 0));

        // fill container with characters at specific positions i mod 7 = ...
        size_t counter = 0;
        for (size_t i = 0; i < n; i += 7) {
            t_0[counter++] = i;
        }
        counter = 0;
        for (size_t i = 3; i < n; i += 7) {
            t_3[counter++] = i;
        }
        counter = 0;
        for (size_t i = 5; i < n; i += 7) {
            t_5[counter++] = i;
        }
        counter = 0;
        for (size_t i = 6; i < n; i += 7) {
            t_6[counter++] = i;
        }

        // empty sa_0 which should be filled correctly with method induce_sa_dc
        auto sa_0 = sacabench::util::make_container<size_t>(t_0.size());
        // empty sa_3 which should be filled correctly with method induce_sa_dc
        auto sa_3 = sacabench::util::make_container<size_t>(t_3.size());
        // empty sa_5 which should be filled correctly with method induce_sa_dc
        auto sa_5 = sacabench::util::make_container<size_t>(t_5.size());
        // empty sa_6 which should be filled correctly with method induce_sa_dc
        auto sa_6 = sacabench::util::make_container<size_t>(t_6.size());

        // TODO: Delete all isa_0 and determina_sa for isa. Not Needed
        // anymore!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // empty sa_0 which should be filled correctly with method induce_sa_dc
        auto isa_0 = sacabench::util::make_container<size_t>(t_0.size());
        // empty sa_3 which should be filled correctly with method induce_sa_dc
        auto isa_3 = sacabench::util::make_container<size_t>(t_3.size());
        // empty sa_5 which should be filled correctly with method induce_sa_dc
        auto isa_5 = sacabench::util::make_container<size_t>(t_5.size());
        // empty sa_6 which should be filled correctly with method induce_sa_dc
        auto isa_6 = sacabench::util::make_container<size_t>(t_6.size());

        // fill sa_3 by inducing with characters at i mod 7 = 3 and ranks of
        // tupels beginning in positions i mod 7 = 2
        // start_pos: position, of ranks i mod 7 = 2 of t_124
        //TODO: maybe use inverse suffix array?
        //size_t start_pos = t_124.size() / 3 + ((t_124.size() % 3) != 0) + 1;
        induce_sa_dc<C>(text, t_3, isa_124, sa_3, start_of_pos_2+1);
        // fill sa_5 by inducing with characters at i mod 7 = 5 and ranks of
        // tupels beginning in positions i mod 7 = 4
        //start_pos += t_3.size() / 3 + ((t_3.size() % 3) == 2);
        induce_sa_dc<C>(text, t_5, isa_124, sa_5, start_of_pos_4+1);
        
        // fill sa_6 by inducing with characters at i mod 7 = 6 and ranks of
        // tupels beginning in positions i mod 7 = 5
        size_t start_pos = 1;
        determine_sa(sa_5, isa_5);
        induce_sa_dc<C>(text, t_6, isa_5, sa_6, start_pos);
        // fill sa_0 by inducing with characters at i mod 7 = 0 and ranks of
        // tupels beginning in positions i mod 7 = 6
        start_pos = 0;
        determine_sa(sa_6, isa_6);
        induce_sa_dc<C>(text, t_0, isa_6, sa_0, start_pos);

        determine_sa(sa_0, isa_0);
        determine_sa(sa_3, isa_3);
        determine_sa(sa_5, isa_5);
        determine_sa(sa_6, isa_6);

        // Anfang
        // Debug-Informationen------------------------------------------------------------------------
        if(OUTPUT){
        std::cout << "text:   ";
        for (size_t i = 0; i < text.size() - 6; i++) {
            std::cout << text[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "sa_0:   ";}
        for (size_t i = 0; i < sa_0.size(); i++) {
            sa_0[i] = sa_0[i] * 7 + 0;
            if(OUTPUT){
            std::cout << sa_0[i] << " ";
            }
        }if(OUTPUT){
        std::cout << std::endl;

        std::cout << "sa_3:   ";}
        for (size_t i = 0; i < sa_3.size(); i++) {
            sa_3[i] = sa_3[i] * 7 + 3;
            if(OUTPUT){
            std::cout << sa_3[i] << " ";}
        }if(OUTPUT){
        std::cout << std::endl;

        std::cout << "sa_5:   ";}
        for (size_t i = 0; i < sa_5.size(); i++) {
            sa_5[i] = sa_5[i] * 7 + 5;
            if(OUTPUT){
            std::cout << sa_5[i] << " ";
            }
        }if(OUTPUT){
        std::cout << std::endl;

        std::cout << "sa_6:   ";}
        for (size_t i = 0; i < sa_6.size(); i++) {
            sa_6[i] = sa_6[i] * 7 + 6;
            if(OUTPUT){
            std::cout << sa_6[i] << " ";
            }
        }
        if(OUTPUT){
        std::cout << std::endl;

        std::cout << "sa_124:   ";
        
        for (size_t i = 0; i < tuples_124.size(); i++) {
            std::cout << tuples_124[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "isa_124:   ";
            for (size_t i = 0; i < isa_124.size(); i++) {
                std::cout << isa_124[i] << " ";
            }

        std::cout << std::endl;}

        // merge_sa_dc<C>(text, sa_0, sa_124, sa_3,sa_5,sa_6, isa_0,
        // isa_124,isa_3, isa_5, isa_6, out_sa);
        // Ende
        // Debug-Informationen------------------------------------------------------------------------

        // merging the SA's of 7-tuples in i mod 7 = 1, 2, 4  and ranks of i mod
        // 3 = 0, 3, 5, 6

        // temporary suffix array, because we had to add a dummy triplet
        // This dummy triplet has to be deleted before merging
        auto tmp_out_sa = sacabench::util::container<size_t>(out_sa.size() + 1);

        
        if constexpr (rec) {
            merge_sa_dc<size_t>(text, sa_0, tuples_124, sa_3, sa_5, sa_6, isa_124,
                                start_of_pos_2, start_of_pos_4, tmp_out_sa);
        } else {
            merge_sa_dc<sacabench::util::character>(
                text, sa_0, tuples_124, sa_3, sa_5, sa_6, isa_124,
                start_of_pos_2, start_of_pos_4, tmp_out_sa);
        }

        for (size_t i = 1; i < tmp_out_sa.size(); ++i) {
            out_sa[i - 1] = tmp_out_sa[i];
        }

        // Anfang
        // Debug-Informationen------------------------------------------------------------------------
        if(OUTPUT){
        std::cout << "sa:   ";
        for (size_t i = 0; i < out_sa.size(); i++) {
            std::cout << out_sa[i] << " ";
        }
        std::cout << std::endl << std::endl;
        }
        // Ende
        // Debug-Informationen------------------------------------------------------------------------
        
    }
    template <typename C, typename T, typename Text, typename I, typename S>
    static void induce_sa_dc(const Text& input_text, const T& t, const I& isa,
                             S& sa, size_t start_pos) {

        DCHECK_MSG(sa.size() == t.size(), "sa must have the same length as t");

        // Container to store all tuples with the same length as sa
        // Tuples contains six characters, a rank and the start position
        auto sa_to_be_sorted = sacabench::util::make_container<
            std::tuple<C, C, C, C, C, C, size_t, size_t>>(sa.size());

            //TODO: abfangen, ob aus Grenzen i mod 7 = 2 raus..
        for (size_t i = 0; i < t.size(); ++i) {
            if ((start_pos + i) < isa.size()) {
                sa_to_be_sorted[i] =
                    (std::tuple<C, C, C, C, C, C, size_t, size_t>(
                        input_text[t[i]], input_text[t[i] + 1],
                        input_text[t[i] + 2], input_text[t[i] + 3],
                        input_text[t[i] + 4], input_text[t[i] + 5],
                        isa[start_pos + i], i));
            } else {
                sa_to_be_sorted[i] =
                    (std::tuple<C, C, C, C, C, C, size_t, size_t>(
                        input_text[t[i]], input_text[t[i] + 1],
                        input_text[t[i] + 2], input_text[t[i] + 3],
                        input_text[t[i] + 4], input_text[t[i] + 5], 0, i));
            }
        }

        // TODO: sort Tupels with radix_sort
        // radix_sort(sa0_to_be_sorted, sa0);

        std::sort(sa_to_be_sorted.begin(), sa_to_be_sorted.end());
        for (size_t i = 0; i < sa_to_be_sorted.size(); i++) {
            sa[i] = std::get<7>(sa_to_be_sorted[i]);
            if(OUTPUT){
            std::cout << sa[i] << std::endl;
            }
        }if(OUTPUT){
        std::cout << std::endl;}
    }

    template <typename C, typename T, typename S, typename S_124, typename I,
              typename SA>
    static void merge_sa_dc(const T& text, const S& sa_0, const S_124& sa_124,
                            const S& sa_3, const S& sa_5, const S& sa_6,
                            const I& isa_124, const size_t start_of_pos_2,
                            const size_t start_of_pos_4, SA& sa) {

        // TODO: DCHECK_MSG(...)

        /*static auto merge_table = sacabench::util::make_container<
            std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>>(
            7);
        merge_table[0] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                0, 6, 5, 6, 3, 3, 5));
        merge_table[1] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                6, 0, 0, 6, 0, 4, 4));
        merge_table[2] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                5, 0, 0, 1, 0, 1, 5));
        merge_table[3] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                6, 6, 1, 0, 2, 1, 2));
        merge_table[4] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                3, 0, 0, 2, 0, 3, 2));
        merge_table[5] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                3, 4, 1, 1, 3, 0, 4));
        merge_table[6] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                5, 4, 5, 2, 2, 4, 0));*/

        // shift values to merge. For example if SA_5 and SA_6 are compared,
        // the tupels with length std::get<5>(merge_table[6])=3 will be
        // compared. If the tuples are the same, the ranks of the Difference
        // Cover {1,2,4} at indexes 5+3=8 and 6+3=9 will be compared, which
        // cannot be the same.
        static auto merge_table = sacabench::util::make_container<
            std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>>(
            7);
        merge_table[0] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                0, 1, 2, 1, 4, 4, 2));
        merge_table[1] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                1, 0, 0, 1, 0, 3, 3));
        merge_table[2] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                2, 0, 0, 6, 0, 6, 2));
        merge_table[3] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                1, 1, 6, 0, 5, 6, 5));
        merge_table[4] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                4, 0, 0, 5, 0, 4, 5));
        merge_table[5] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                4, 3, 6, 6, 4, 0, 3));
        merge_table[6] =
            (std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t>(
                2, 3, 2, 5, 5, 3, 0));

            if(OUTPUT){
        for (size_t i = 0; i < 7; ++i) {
            std::cout << std::get<0>(merge_table[i]) << " ";
            std::cout << std::get<1>(merge_table[i]) << " ";
            std::cout << std::get<2>(merge_table[i]) << " ";
            std::cout << std::get<3>(merge_table[i]) << " ";
            std::cout << std::get<4>(merge_table[i]) << " ";
            std::cout << std::get<5>(merge_table[i]) << " ";
            std::cout << std::get<6>(merge_table[i]) << " ";
            std::cout << std::endl;
        }
            }

        // Compare
        auto all_sa = sacabench::util::make_container<std::vector<size_t>>(5);

        for (size_t i = 0; i < sa_0.size(); ++i) {
            all_sa[0].push_back(sa_0[i]);
        }
        for (size_t i = 0; i < sa_124.size(); ++i) {
            all_sa[1].push_back(sa_124[i]);
        }
        for (size_t i = 0; i < sa_3.size(); ++i) {
            all_sa[2].push_back(sa_3[i]);
        }
        for (size_t i = 0; i < sa_5.size(); ++i) {
            all_sa[3].push_back(sa_5[i]);
        }
        for (size_t i = 0; i < sa_6.size(); ++i) {
            all_sa[4].push_back(sa_6[i]);
        }

        // 0 = 0, 1 = 124, 2 = 3, 3 = 4, 4 = 5
        std::array<size_t, 5> counters = {0, 0, 0, 0, 0};
        std::array<size_t, 5> max_counters;
        for (size_t i = 0; i < max_counters.size(); ++i) {
            max_counters[i] = all_sa[i].size();
        }

        std::queue<size_t> queue;
        auto to_be_compared =
            sacabench::util::make_container<std::vector<C>>(2);
        for (size_t sa_ind = 0; sa_ind < sa.size(); ++sa_ind) {
            for (size_t queue_ind = 0; queue_ind < max_counters.size();
                 ++queue_ind) {
                if (max_counters[queue_ind] > counters[queue_ind]) {
                    queue.push(queue_ind);
                }
            }
            
            size_t comp_1;
            size_t comp_2;
            size_t smallest = queue.front();
            while (queue.size() > 1) {

                comp_1 = queue.front();
                queue.pop();
                comp_2 = queue.front();
                queue.pop();
                size_t length;
                if(OUTPUT){
                std::cout << "alle SAs: " << std::endl;
                std::cout << "sa_0" << ": ";
                for(size_t j = 0 ; j < all_sa[0].size(); ++j){
                        std::cout << all_sa[0][j] << ", ";
                }    
                std::cout << std::endl;
                
                std::cout << "sa_124" << ": ";
                for(size_t j = 0 ; j < all_sa[1].size(); ++j){
                        std::cout << all_sa[1][j] << ", ";
                }    
                std::cout << std::endl;
                
                std::cout << "sa_3" << ": ";
                for(size_t j = 0 ; j < all_sa[2].size(); ++j){
                        std::cout << all_sa[2][j] << ", ";
                }    
                std::cout << std::endl;
                
                std::cout << "sa_5" << ": ";
                for(size_t j = 0 ; j < all_sa[3].size(); ++j){
                        std::cout << all_sa[3][j] << ", ";
                }    
                std::cout << std::endl;
                std::cout << "sa_6" << ": ";
                for(size_t j = 0 ; j < all_sa[4].size(); ++j){
                        std::cout << all_sa[4][j] << ", ";
                }    
                std::cout << std::endl;
                
                
                
                std::cout << "Vergleich zwischen " << comp_1 << " ("
                          << counters[comp_1] << ") "
                          << " und: " << comp_2 << " (" << counters[comp_2]
                          << ") " << std::endl;
                std::cout << "Vergleich zwischen "
                          << all_sa[comp_1][counters[comp_1]]
                          << " und: " << all_sa[comp_2][counters[comp_2]]
                          << std::endl;
                }

                switch (comp_1) {
                case 0:
                    length = std::get<0>(
                        merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    break;
                case 1: {
                    if (all_sa[1][counters[1]] % 7 == 1)
                        length = std::get<1>(
                            merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    else if (all_sa[1][counters[1]] % 7 == 2)
                        length = std::get<2>(
                            merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    else
                        length = std::get<4>(
                            merge_table[all_sa[comp_2][counters[comp_2]] % 7]);

                    break;
                }
                case 2:
                    length = std::get<3>(
                        merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    break;
                case 3:
                    length = std::get<5>(
                        merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    break;
                case 4:
                    length = std::get<6>(
                        merge_table[all_sa[comp_2][counters[comp_2]] % 7]);
                    break;
                default: { break; }
                }
                if(OUTPUT){
                std::cout << "length: " << length << std::endl;
                }

                // somehow, the container is not empty. That's why the size will
                // be shrinked to 0
                to_be_compared[0].resize(0);
                to_be_compared[1].resize(0);

                for (size_t i = 0; i < length; ++i) {
                    to_be_compared[0].push_back(
                        text[all_sa[comp_1][counters[comp_1]] + i]);
                    to_be_compared[1].push_back(
                        text[all_sa[comp_2][counters[comp_2]] + i]);
                }
                if(OUTPUT){
                std::cout << "Liste1                   Liste2" << std::endl;
                for (size_t i = 0; i < to_be_compared[0].size(); ++i) {

                    std::cout << to_be_compared[0][i]
                              << "                           "
                              << to_be_compared[1][i] << std::endl;
                }}

                if (to_be_compared[0] < to_be_compared[1]) {
                    //std::cout << "smaller" << std::endl;
                    smallest = comp_1;

                } else if (to_be_compared[0] > to_be_compared[1]) {
                    //std::cout << "greater" << std::endl;
                    smallest = comp_2;
                } else {

                    const size_t index_1 =
                        all_sa[comp_1][counters[comp_1]] + length;
                    const size_t index_2 =
                        all_sa[comp_2][counters[comp_2]] + length;

                    size_t pos_1;
                    size_t pos_2;

                    switch (index_1 % 7) {
                    case 1: {
                        pos_1 = 0;
                        if (index_2 % 7 == 2)
                            pos_2 = start_of_pos_2;
                        else
                            pos_2 = start_of_pos_4;
                        break;
                    }
                    case 2: {
                        pos_1 = start_of_pos_2;
                        if (index_2 % 7 == 1)
                            pos_2 = 0;
                        else
                            pos_2 = start_of_pos_4;
                        break;
                    }
                    case 4: {
                        pos_1 = start_of_pos_4;
                        if (index_2 % 7 == 1)
                            pos_2 = 0;
                        else
                            pos_2 = start_of_pos_2;
                        break;
                    }
                    default:
                        //std::cout << "Merge error!" << std::endl;
                        break;
                    }
                    //std::cout << "index1: " << index_1 << " vergleich mit: "
                    //          << "index2: " << index_2 << std::endl;
                    if (isa_124[pos_1 + index_1 / 7] <
                        isa_124[pos_2 + index_2 / 7])
                        smallest = comp_1;
                    else
                        smallest = comp_2;
                }
                queue.push(smallest);
            }
            queue.pop();
            //std::cout << "smallest: " << smallest << " , "
            //          << all_sa[smallest][counters[smallest]] << std::endl;
            sa[sa_ind] = all_sa[smallest][counters[smallest]++];
        }
        if(OUTPUT){
            std::cout << "sa: ";
        for (size_t i = 0; i < sa.size(); ++i) {
            std::cout << sa[i] << " ";
        }
        }
    }
}; // class dc7

} // namespace sacabench::dc7
