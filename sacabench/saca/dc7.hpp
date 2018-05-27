/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <util/string.hpp>
#include <util/span.hpp>
#include <util/container.hpp>
#include <util/induce_sa_dc.hpp>
#include <util/merge_sa_dc.hpp>
#include <tuple>
#include <string>
#include <math.h>

namespace sacabench::dc7 {
    
    class dc7 {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                if(text.size()==0){
                    
                }else if(text.size()==1){
                    out_sa[0] = 0;
                }else if(text.size()==2){
                    out_sa[0] = 0;
                }else if(text.size()==3){
                    out_sa[0] = 0;
                }else if(text.size()==4){
                    out_sa[0] = 0;
                }else if(text.size()==5){
                    out_sa[0] = 0;
                }else{
                    auto modified_text = sacabench::util::make_container<sacabench::util::character>(text.size()+6);

                    for(size_t i = 0; i < text.size()+6 ; i++){
                        if(i < text.size()){
                            modified_text[i] = text[i];
                        }else{
                            modified_text[i] = ' ';
                        }
                    }
                    construct_sa_dc7<size_t, false, sacabench::util::character>(modified_text, alphabet_size, out_sa);
                }
            }
            
            
            
            
    private:
        template<typename C, typename T, typename S>
        static void determine_tuples(const T& INPUT_STRING, S& tuples_124) {
            
            
            size_t n = INPUT_STRING.size()-6;
            DCHECK_MSG(tuples_124.size() == 3*n/7 + (((INPUT_STRING.size() % 7) !=0)), "tuples_124 must have the length (3*INPUT_STRING.size()/7)");
            
            const unsigned char SMALLEST_CHAR = ' ';
            //Container to store all tuples with the same length as tuples_124
            //Tuples contains three chararcters (triplet) and the start position i mod 3 != 0 
            auto tuples_124_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, C, C, C, C, size_t>>(tuples_124.size());
    
            size_t counter = 0;
            
            //--------------------------------hier ohne extra Sentinals--------------------------------------//
   
            /*for(size_t i = 1; i < INPUT_STRING.size(); i++) {
                if(i % 3 != 0){
                    if((i+2) >= INPUT_STRING.size()){
                        if((i+1) >= INPUT_STRING.size()){
                            tuples_124_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                        }else{
                            tuples_124_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                        }
                    }else{
                        tuples_124_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                    }
                }
            }*/  
            
            //-----------------------------------------------------------------------------------------------//
            
            
            for(size_t i = 1; i < n; i++) {
                std::cout << i << std::endl;
                if(((i % 7) == 1) || ((i % 7) == 2) || ((i % 7) == 4)){
                    std::cout << i << ": " << INPUT_STRING[i] << " " << INPUT_STRING[i+1] << " " << INPUT_STRING[i+2]
                    << " " << INPUT_STRING[i+3] << " " << INPUT_STRING[i+4]
                    << " " << INPUT_STRING[i+5] << " " << INPUT_STRING[i+6]<< std::endl;
                    
                    std::cout << counter << " von " << tuples_124_to_be_sorted.size() << std::endl;
                    tuples_124_to_be_sorted[counter++] = std::tuple<C, C, C, C, C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1],
                                                                                                   INPUT_STRING[i+2], INPUT_STRING[i+3],
                                                                                                   INPUT_STRING[i+4], INPUT_STRING[i+5], INPUT_STRING[i+6], i);
                }
            }
            
            //TODO: sort Tupels with radix_sort
            //radix_sort(sa0_to_be_sorted, sa0);        
            std::sort(tuples_124_to_be_sorted.begin(),tuples_124_to_be_sorted.end());
            
            
            for(size_t i = 0; i<tuples_124_to_be_sorted.size();i++){   
                tuples_124[i] = std::get<7>(tuples_124_to_be_sorted[i]);
            }
                    
        }
        
        template<typename T, typename S>
        static void determine_leq(const T& INPUT_STRING, const S& tuples_124, S& t_124, bool& recursion){
            
            DCHECK_MSG(tuples_124.size() == t_124.size(), "tuples_124 must have the same length as t_124");
            
            size_t n = INPUT_STRING.size() - 6;
            
            size_t leq_name = 0;
            size_t start_of_pos_2 = n/7 + ((((n-1) % 7) !=0));
            size_t start_of_pos_4 = 2 * start_of_pos_2 - (((n % 6) == 3));
            size_t pos_to_store_leq;
            for(size_t i = 0; i < tuples_124.size(); i++) {
                //set the lexicographical names at correct positions:
                //[----names at positions i mod 7 = 1----||----names at positions i mod 7 = 2----||----names at positions i mod 7 = 4----]
                if(tuples_124[i] % 7 == 1){
                    pos_to_store_leq = tuples_124[i]/7;
                }else if(tuples_124[i] % 7 == 2){
                    pos_to_store_leq = start_of_pos_2 + tuples_124[i]/7;
                }else{
                    pos_to_store_leq = start_of_pos_4 + tuples_124[i]/7;
                }
                t_124[pos_to_store_leq] = leq_name;
                if(i+1 < tuples_124.size()){
                    if(sacabench::util::span(&INPUT_STRING[tuples_124[i]], 3) != sacabench::util::span(&INPUT_STRING[tuples_124[i+1]], 3)){
                        leq_name++;
                    }else{ //if lexicographical names are not uniqe set recursion = true
                        recursion = true;
                    }
                }
                
                //--------------------------------hier ohne extra Sentinals--------------------------------------//
                
                /*if(i+1 < tuples_124.size()){
                    
                    if(tuples_124[i]+3 < INPUT_STRING.size()){
                        
                        if(tuples_124[i+1]+3 < INPUT_STRING.size(){
                            if(sacabench::util::span(&INPUT_STRING[tuples_124[i]], 3) != sacabench::util::span(&INPUT_STRING[tuples_124[i+1]], 3)){
                                leq_name++;
                            }else{ //if lexicographical names are not uniqe set recursion = true
                                recursion = true;
                            }
                        }else{
                            leq_name++;
                        }
                        
                    }else
                        leq_name++;
                } */ 
                
                //-----------------------------------------------------------------------------------------------//
                
            }  
            //TODO: Abfragen ob INPUT_STRING und tuples_124 out of bounce
        }
        
        template<typename S, typename I>
        static void determine_isa(const S& t_124, I& isa_124){
            
            DCHECK_MSG(isa_124.size() == t_124.size(), "isa_124 must have the same length as t_124");
            
            for(size_t i = 0; i < t_124.size(); i++) {
                isa_124[t_124[i]] = i+1;
            }
            
        }
        
        template<typename sa_index, bool rec, typename C, typename S>
        static void construct_sa_dc7(S text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
            
            
                size_t n = text.size()-6;
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(rec){
                    std::cout << std::endl << "Wir befinden uns in der Rekursion" << std::endl;
                }else{
                    std::cout << std::endl << "Wir befinden uns in der Hauptmethode" << std::endl;
                }
                
                std::cout << "text der Länge: " << n << ":  ";
                for(size_t i = 0; i < n; i++){
                    std::cout << text[i];
                }
                std::cout << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //empty container which will contain indices of triplet 
                //at positions i mod 3 != 0
                auto tuples_124 = sacabench::util::make_container<size_t>(3*(n)/7 + (((text.size() % 7) !=0)));
                std::cout << tuples_124.size() << std::endl;
                //determine positions and calculate the sorted order
                determine_tuples<C>(text, tuples_124);   
                
                //empty SA which should be filled correctly with lexicographical 
                //names of triplets
                auto t_124 = sacabench::util::make_container<size_t>(3*(n)/7 + (((text.size() % 7) !=0)));
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "tuples:    ";
                for(size_t i = 0; i < tuples_124.size() ; i++){
                        std::cout << tuples_124[i] << " ";
                }
                std::cout << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                
                //bool which will be set true in determine_leq if the names are not unique
                bool recursion = false;
                                
                //fill t_124 with lexicographical names
                determine_leq(text, tuples_124, t_124, recursion);
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "t_124:    ";
                for(size_t i = 0; i < t_124.size() ; i++){
                        std::cout << t_124[i] << " ";
                }
                std::cout << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                util::span<sa_index> sa_124 =  util::span(&out_sa[0], t_124.size());
            
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "vor der Rekursion" << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //run the algorithm recursivly if the names are not unique
                if(recursion){
                    
                    //add two sentinals to the end of the text
                    auto modified_text = sacabench::util::make_container<size_t>(t_124.size()+2);

                    for(size_t i = 0; i < t_124.size()+6 ; i++){
                        if(i < t_124.size()){
                            modified_text[i] = t_124[i];
                        }else{
                            modified_text[i] = ' ';
                        }
                    }
                    
                    //run algorithm recursive
                    construct_sa_dc7<size_t, true, size_t>(modified_text, alphabet_size, sa_124);              
                }
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "nach der Rekursion" << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //empty isa_124 which should be filled correctly with method determine_isa
                auto isa_124 = sacabench::util::make_container<size_t>(0);
                
                //empty merge_isa_124 to be filled with inverse suffix array in format for merge_sa_dc
                auto merge_isa_124 = sacabench::util::make_container<size_t>(tuples_124.size());
                
                //if in recursion use temporary sa. Otherwise t_124
                if(recursion){
                    
                    isa_124 = sacabench::util::make_container<size_t>(sa_124.size());
                    determine_isa(sa_124, isa_124);
                    
                    //Anfang Debug-Informationen------------------------------------------------------------------------
                    std::cout << "sa_124:    ";
                    for(size_t i = 0; i < sa_124.size() ; i++){
                            std::cout << sa_124[i] << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "isa_124:    ";
                    for(size_t i = 0; i < isa_124.size() ; i++){
                            std::cout << isa_124[i] << " ";
                    }
                    std::cout << std::endl;
                    //Ende Debug-Informationen------------------------------------------------------------------------
                    
                    
                    //index of the first value which represents the positions i mod 3 = 2
                    size_t end_of_mod_eq_1 = tuples_124.size()/2; // + ((tuples_124.size()/2) % 2 == 0);
                    if(tuples_124.size() % 2 != 0){
                        end_of_mod_eq_1++;
                    }
                    
                    //correct the order of sa_124 with result of recursion
                    for(size_t i = 0; i < tuples_124.size(); i++){
                        if(i < end_of_mod_eq_1){
                            tuples_124[isa_124[i]-1] = 3 * i + 1; 
                        }else{
                            tuples_124[isa_124[i]-1] = 3 * (i-end_of_mod_eq_1) + 2;
                        }
                    }
                    
                    //Anfang Debug-Informationen------------------------------------------------------------------------
                    std::cout << "korrigierte triplets:    ";
                    for(size_t i = 0; i < tuples_124.size() ; i++){
                            std::cout << tuples_124[i] << " ";
                    }
                    std::cout << std::endl;
                    //Ende Debug-Informationen------------------------------------------------------------------------
                    
                    //convert isa_124 to the correct format for merge_sa_dc.
                    auto merge_isa_124_to_be_sorted = sacabench::util::make_container<std::tuple<size_t, size_t>>(tuples_124.size());
                    for(size_t i = 0; i < merge_isa_124.size(); i++){
                        merge_isa_124_to_be_sorted[i] = (std::tuple<size_t, size_t>(tuples_124[i], i));
                    }
                    std::sort(merge_isa_124_to_be_sorted.begin(),merge_isa_124_to_be_sorted.end());
                    for(size_t i = 0; i<merge_isa_124_to_be_sorted.size();i++){
                        merge_isa_124[i] = std::get<1>(merge_isa_124_to_be_sorted[i]);
                    }
                }else{
                    isa_124 = t_124;
                    determine_isa(isa_124, sa_124);
                }
                
                //characters of positions i mod 7 = 0 of text
                auto t_0 = sacabench::util::make_container<C>(n/7 + ((n % 7) != 0));
                //characters of positions i mod 7 = 3 of text
                auto t_3 = sacabench::util::make_container<C>((n-3)/7 + (((n-3) % 7) != 0));
                //characters of positions i mod 7 = 5 of text
                auto t_5 = sacabench::util::make_container<C>((n-5)/7 + (((n-5) % 7) != 0));
                //characters of positions i mod 7 = 6 of text
                auto t_6 = sacabench::util::make_container<C>((n-6)/7 + (((n-6) % 7) != 0));

                //fill container with characters at specific positions i mod 7 = ...
                size_t counter = 0;
                for(size_t i = 0; i < n; i+=7){
                    t_0[counter++] =  text[i];
                }
                counter = 0;
                for(size_t i = 3; i < n; i+=7){
                    t_3[counter++] =  text[i];
                }
                counter = 0;
                for(size_t i = 5; i < n; i+=7){
                    t_5[counter++] =  text[i];
                }
                counter = 0;
                for(size_t i = 6; i < n; i+=7){
                    t_6[counter++] =  text[i];
                }
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "t_0:    ";
                for(size_t i = 0; i < t_0.size() ; i++){
                        std::cout << t_0[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "t_3:    ";
                for(size_t i = 0; i < t_3.size() ; i++){
                        std::cout << t_3[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "t_5:    ";
                for(size_t i = 0; i < t_5.size() ; i++){
                        std::cout << t_5[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "t_6:    ";
                for(size_t i = 0; i < t_6.size() ; i++){
                        std::cout << t_6[i] << " ";
                }
                std::cout << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                
                //empty sa_0 which should be filled correctly with method induce_sa_dc
                auto sa_0 = sacabench::util::make_container<size_t>(t_0.size());
                //empty sa_3 which should be filled correctly with method induce_sa_dc
                auto sa_3 = sacabench::util::make_container<size_t>(t_3.size());
                //empty sa_5 which should be filled correctly with method induce_sa_dc
                auto sa_5 = sacabench::util::make_container<size_t>(t_5.size());
                //empty sa_6 which should be filled correctly with method induce_sa_dc
                auto sa_6 = sacabench::util::make_container<size_t>(t_6.size());
                
                
                //fill sa_3 by inducing with characters at i mod 7 = 3 and ranks of tupels 
                //beginning in positions i mod 7 = 2
                sacabench::util::induce_sa_dc<C>(t_3, isa_124, sa_3);
                //fill sa_5 by inducing with characters at i mod 7 = 5 and ranks of tupels 
                //beginning in positions i mod 7 = 4
                sacabench::util::induce_sa_dc<C>(t_5, isa_124, sa_5);
                //fill sa_6 by inducing with characters at i mod 7 = 6 and ranks of tupels 
                //beginning in positions i mod 7 = 5
                sacabench::util::induce_sa_dc<C>(t_6, t_5, sa_6);
                //fill sa_0 by inducing with characters at i mod 7 = 0 and ranks of tupels 
                //beginning in positions i mod 7 = 6
                sacabench::util::induce_sa_dc<C>(t_0, t_6, sa_0);
                
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "text:   ";
                for(size_t i = 0; i < text.size()-2; i++){
                    std::cout << text[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "sa_0:   ";
                for(size_t i = 0; i < sa_0.size(); i++){
                    std::cout << sa_0[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "sa_3:   ";
                for(size_t i = 0; i < sa_3.size(); i++){
                    std::cout << sa_3[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "sa_5:   ";
                for(size_t i = 0; i < sa_5.size(); i++){
                    std::cout << sa_5[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "sa_6:   ";
                for(size_t i = 0; i < sa_6.size(); i++){
                    std::cout << sa_6[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "sa_124:   ";
                for(size_t i = 0; i < tuples_124.size(); i++){
                    std::cout << tuples_124[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "isa_124:   ";
                if(rec){
                    for(size_t i = 0; i < isa_124.size() ; i++){
                        std::cout << isa_124[i] << " ";
                    }
                }else{
                    for(size_t i = 0; i < merge_isa_124.size(); i++){
                        std::cout << merge_isa_124[i] << " ";
                    }
                }
                std::cout << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //merging the SA's of 7-tuples in i mod 7 = 1, 2, 4  and ranks of i mod 3 = 0, 3, 5, 6
                /*if constexpr(rec){ 
                    sacabench::util::merge_sa_dc<const size_t>(sacabench::util::span(&text[0], text.size()-2), sa_0, tuples_124,
                        isa_124, out_sa, comp_recursion, get_substring_recursion);
                }else{
                    sacabench::util::merge_sa_dc<const sacabench::util::character>(sacabench::util::span(&text[0], text.size()-2), sa_0, tuples_124,
                        merge_isa_124, out_sa, comp, get_substring);
                }*/
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                std::cout << "sa:   ";
                for(size_t i = 0; i < out_sa.size(); i++){
                    std::cout << out_sa[i] << " ";
                }
                std::cout << std::endl << std::endl;
                //Ende Debug-Informationen------------------------------------------------------------------------
            }         
            
            // implementation of get_substring method with type of character not in recursion
            static const sacabench::util::span<const sacabench::util::character> get_substring(const sacabench::util::span<sacabench::util::character>& t, const sacabench::util::character* ptr,
                    size_t n, size_t index) {
                return sacabench::util::span(ptr, n);
            }

            // implementation of comp method
            static bool comp(const sacabench::util::string_span& a, const sacabench::util::string_span& b) {
                return a < b;
            }
            
            // implementation of get_substring method with type size_t in recursion
            static const sacabench::util::span<const size_t> get_substring_recursion(const sacabench::util::span<size_t>& t, const size_t* ptr,
                    size_t n, size_t index) {
                return sacabench::util::span(ptr, n);
            }

            // implementation of comp method
            static bool comp_recursion(const sacabench::util::span<const size_t>& a, const sacabench::util::span<const size_t>& b) {
                return a < b;
            }
    }; // class dc7
        
}  // namespace sacabench::saca
