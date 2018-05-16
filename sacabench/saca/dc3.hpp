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
#include <vector>
#include <util/container.hpp>
#include <util/induce_sa_dc.hpp>
#include <util/merge_sa_dc.hpp>
#include <tuple>
#include <string>
#include <math.h>

namespace sacabench::dc3 {
    
    /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
    * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input text
    * \param it_12 calculated ISA for triplets beginning in i mod 3 != 0
    * \param sa_0 memory block for resulting SA for positions beginning in i mod 3 = 0
    *
    * This method identifies the order of the characters in input_string in positions i mod 3 = 0
    * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
    * This method works correct because of the difference cover idea.
    */
    
    template<typename C, typename T, typename S>
    void determine_triplets(const T& INPUT_STRING, S& t_12) {
        
        
        DCHECK_MSG(t_12.size() == 2*INPUT_STRING.size()/3, "t_12 must have the length (2*INPUT_STRING.size()/3)");
        
        
        const unsigned char SMALLEST_CHAR = ' ';
        //Container to store all tuples with the same length as t_12
        //Tuples contains three chararcters (triplet) and the start position i mod 3 != 0 
        auto t_12_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, size_t>>(2*INPUT_STRING.size()/3);
 
        size_t counter = 0;
        for(size_t i = 1; i < INPUT_STRING.size(); i++) {
            if(i % 3 != 0){
                if((i+2) >= INPUT_STRING.size()){
                    if((i+1) >= INPUT_STRING.size()){
                        t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                    }else{
                        t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                    }
                }else{
                    t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }
            }
        }  
        
        //TODO: sort Tupels with radix_sort
        //radix_sort(sa0_to_be_sorted, sa0);        
        std::sort(t_12_to_be_sorted.begin(),t_12_to_be_sorted.end());
        
        for(size_t i = 0; i<t_12_to_be_sorted.size();i++){   
            t_12[i] = std::get<3>(t_12_to_be_sorted[i]);
        }
                
    }
    
    template<typename T, typename S>
    void determine_leq(const T& INPUT_STRING, const S& t_12, S& sa_12, bool& recursion){
        
        
        DCHECK_MSG(t_12.size() == sa_12.size(), "t_12 must have the same length as sa_12");
        
        size_t leq_name = 0;
        
        for(size_t i = 0; i < t_12.size(); i++) {
            //set the lexicographical names at correct positions:
            //[----names at positions i mod 3 = 1----||----names at positions i mod 3 = 2----]
            if(t_12[i] % 3 == 1){
                sa_12[t_12[i]/3] = leq_name;
            }else{                
                if(sa_12.size() % 2 == 0){
                    sa_12[sa_12.size()/2 + t_12[i]/3] = leq_name;
                }else{
                    sa_12[sa_12.size()/2 + 1 + t_12[i]/3] = leq_name;
                }
                
            }
            if(i+1 < t_12.size()){
                if(sacabench::util::span(&INPUT_STRING[t_12[i]], 3) != sacabench::util::span(&INPUT_STRING[t_12[i+1]], 3)){
                    leq_name++;
                }else{ //if lexicographical names are not uniqe set recursion = true
                    recursion = true;
                }
            }
        }  
        //TODO: Abfragen ob INPUT_STRING und t_12 out of bounce
    }
    
    template<typename S, typename I>
    void determine_isa(const S& sa_12, I& isa_12){
        
        DCHECK_MSG(isa_12.size() == sa_12.size(), "isa_12 must have the same length as sa_12");
        
        for(size_t i = 0; i < sa_12.size(); i++) {
            isa_12[sa_12[i]] = i+1;
        }        
    }
    
    
    
    class dc3 {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                auto modified_text = sacabench::util::make_container<sacabench::util::character>(text.size());
                
                for(size_t i = 0; i < text.size() ; i++){
                    modified_text[i] = text[i];
                }
                modified_text.push_back(' ');
                modified_text.push_back(' ');
                
                construct_sa_dc3(modified_text, alphabet_size, out_sa);
            }
            
            
            
            
    private:
        template<typename C, typename T, typename S>
        static void determine_triplets(const T& INPUT_STRING, S& t_12) {
            
            
            DCHECK_MSG(t_12.size() == 2*(INPUT_STRING.size()-2)/3, "t_12 must have the length (2*INPUT_STRING.size()/3)");
            
            
            const unsigned char SMALLEST_CHAR = ' ';
            //Container to store all tuples with the same length as t_12
            //Tuples contains three chararcters (triplet) and the start position i mod 3 != 0 
            auto t_12_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, size_t>>(2*(INPUT_STRING.size()-2)/3);
    
            size_t counter = 0;
            /*for(size_t i = 1; i < INPUT_STRING.size(); i++) {
                if(i % 3 != 0){
                    if((i+2) >= INPUT_STRING.size()){
                        if((i+1) >= INPUT_STRING.size()){
                            t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                        }else{
                            t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                        }
                    }else{
                        t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                    }
                }
            }*/  
            
            std::cout << t_12_to_be_sorted.size() << std::endl;
            std::cout << t_12.size() << std::endl;
            for(size_t i = 1; i < INPUT_STRING.size()-2; i++) {
                if(i % 3 != 0){
                    std::cout << i << ": " << INPUT_STRING[i] << " " << INPUT_STRING[i+1] << " " << INPUT_STRING[i+2] << std::endl;
                    
                    t_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }
            }
            
            //TODO: sort Tupels with radix_sort
            //radix_sort(sa0_to_be_sorted, sa0);        
            std::sort(t_12_to_be_sorted.begin(),t_12_to_be_sorted.end());
            
            std::cout << t_12_to_be_sorted.size() << std::endl;
            for(size_t i = 0; i<t_12_to_be_sorted.size();i++){   
                t_12[i] = std::get<3>(t_12_to_be_sorted[i]);
            }
                    
        }
        
        template<typename T, typename S>
        static void determine_leq(const T& INPUT_STRING, const S& t_12, S& sa_12, bool& recursion){
            
            DCHECK_MSG(t_12.size() == sa_12.size(), "t_12 must have the same length as sa_12");
            
            size_t leq_name = 0;

            
            for(size_t i = 0; i < t_12.size(); i++) {
                //set the lexicographical names at correct positions:
                //[----names at positions i mod 3 = 1----||----names at positions i mod 3 = 2----]
                if(t_12[i] % 3 == 1){
                    sa_12[t_12[i]/3] = leq_name;
                }else{
                    if(sa_12.size() % 2 == 0){
                        sa_12[sa_12.size()/2 + t_12[i]/3] = leq_name;
                    }else{
                        sa_12[sa_12.size()/2 + 1 + t_12[i]/3] = leq_name;
                    }
                }

                if(i+1 < t_12.size()){
                    if(sacabench::util::span(&INPUT_STRING[t_12[i]], 3) != sacabench::util::span(&INPUT_STRING[t_12[i+1]], 3)){
                        leq_name++;
                    }else{ //if lexicographical names are not uniqe set recursion = true
                        recursion = true;
                    }
                }
                
            }  
            //TODO: Abfragen ob INPUT_STRING und t_12 out of bounce
        }
        
        template<typename S, typename I>
        static void determine_isa(const S& sa_12, I& isa_12){
            
            DCHECK_MSG(isa_12.size() == sa_12.size(), "isa_12 must have the same length as sa_12");
            
            for(size_t i = 0; i < sa_12.size(); i++) {
                isa_12[sa_12[i]] = i+1;
            }        
        }
        
        template<typename sa_index, typename S>
        static void construct_sa_dc3(S text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                
                
                //empty container which will contain indices of triplet 
                //at positions i mod 3 != 0
                auto t_12 = sacabench::util::make_container<size_t>(2*(text.size()-2)/3);
                
                //determine positions and calculate the sorted order
                determine_triplets<sacabench::util::character>(text, t_12);   
                
                //empty SA which should be filled correctly with lexicographical 
                //names of triplets
                auto sa_12 = sacabench::util::make_container<size_t>(2*(text.size()-2)/3);
                
                
                //bool which will be set true in determine_leq if the names are not unique
                bool recursion = false;
                                
                //fill sa_12 with lexicographical names
                determine_leq(text, t_12, sa_12, recursion);
                std::cout << "sa: " << std::endl;
                for(size_t i = 0; i < sa_12.size(); i++){
                    std::cout << sa_12[i] << std::endl;
                }
                
                //auto tmp_out_sa = util::span<sa_index>(&out_sa[0], 2*out_sa.size()/3);
                //auto tmp_out_sa = out_sa.slice(0, 2*out_sa.size()/3);
                
                auto tmp_tmp_out_sa = util::make_container<sa_index>(2*out_sa.size()/3);
                util::span<sa_index> tmp_out_sa = tmp_tmp_out_sa;
            
                
                std::cout << "vor der Rekursion" << std::endl;
                
                //run the algorithm recursivly if the names are not unique
                if(recursion){
                    construct_sa_dc3<size_t>(sa_12, alphabet_size, tmp_out_sa);              
                }
                
                std::cout << "nach der Rekursion" << std::endl;
                
                
                //empty isa_12 which should be filled correctly with method determine_isa
                //which calculates the ranks of triplets in i mod 3 != 0
                auto isa_12 = sacabench::util::make_container<size_t>(tmp_out_sa.size());
                //auto isa_12 = sacabench::util::span<size_t>(tmp_out_sa.size());
                
                if(recursion){
                    sacabench::dc3::determine_isa(tmp_out_sa, isa_12);
                }else{
                    sacabench::dc3::determine_isa(sa_12, isa_12);
                }
                
                std::cout << "isa: " << std::endl;
                for(size_t i = 0; i < isa_12.size(); i++){
                    std::cout << isa_12[i] << std::endl;
                }
                
                //positions i mod 3 = 0 of text
                auto t_0 = sacabench::util::make_container<sacabench::util::character>(text.size()/3);

                size_t counter = 0;
                for(size_t i = 0; i < (text.size()-2); i+=3){
                    t_0[counter++] =  text[i];
                }

                
                //empty sa_0 which should be filled correctly with method induce_sa_dc
                auto sa_0 = sacabench::util::make_container<size_t>(t_0.size());
                
                //fill sa_0 by inducing with characters at i mod 3 = 0 and ranks of triplets 
                //beginning in positions i mod 3 != 0
                sacabench::util::induce_sa_dc<sacabench::util::character>(t_0, isa_12, sa_0);
                for(size_t i = 0; i < sa_0.size(); i++){
                    std::cout << t_0[i] << "  :  " << sa_0[i] << std::endl;
                }
                
                //merging the SA's of triplets in i mod 3 != 0 and ranks of i mod 3 = 0
                sacabench::util::merge_sa_dc<sacabench::util::character, sacabench::util::container>(text, sa_0, tmp_out_sa,
                        isa_12, out_sa, comp, get_substring);

                std::cout << "Running example1" << std::endl;
            }
                        
            static sacabench::util::string_span get_substring(const sacabench::util::string& t, const sacabench::util::character* ptr,
                    int n) {
                return sacabench::util::span(ptr, n);
            }

            // implementation of comp method
            static bool comp(const sacabench::util::string_span& a, const sacabench::util::string_span& b) {
                return a < b;
            }
    }; // class dc3
        
}  // namespace sacabench::util

