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

#define OUTPUT false

namespace sacabench::dc3 {
    
    /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
    * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input text
    * \param itriplets_12 calculated ISA for triplets beginning in i mod 3 != 0
    * \param sa_0 memory block for resulting SA for positions beginning in i mod 3 = 0
    *
    * This method identifies the order of the characters in input_string in positions i mod 3 = 0
    * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
    * This method works correct because of the difference cover idea.
    */
    /*
    template<typename C, typename T, typename S>
    void determine_triplets(const T& INPUT_STRING, S& triplets_12) {
        
        
        DCHECK_MSG(triplets_12.size() == 2*INPUT_STRING.size()/3, "triplets_12 must have the length (2*INPUT_STRING.size()/3)");
        
        
        const unsigned char SMALLEST_CHAR = ' ';
        //Container to store all tuples with the same length as triplets_12
        //Tuples contains three chararcters (triplet) and the start position i mod 3 != 0 
        auto triplets_12_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, size_t>>(2*INPUT_STRING.size()/3);
 
        size_t counter = 0;
        for(size_t i = 1; i < INPUT_STRING.size(); i++) {
            if(i % 3 != 0){
                if((i+2) >= INPUT_STRING.size()){
                    if((i+1) >= INPUT_STRING.size()){
                        triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                    }else{
                        triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                    }
                }else{
                    triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }
            }
        }  
        
        //TODO: sort Tupels with radix_sort
        //radix_sort(sa0_to_be_sorted, sa0);        
        std::sort(triplets_12_to_be_sorted.begin(),triplets_12_to_be_sorted.end());
        
        for(size_t i = 0; i<triplets_12_to_be_sorted.size();i++){   
            triplets_12[i] = std::get<3>(triplets_12_to_be_sorted[i]);
        }
                
    }
    
    template<typename T, typename S>
    void determine_leq(const T& INPUT_STRING, const S& triplets_12, S& t_12, bool& recursion){
        
        
        DCHECK_MSG(triplets_12.size() == t_12.size(), "triplets_12 must have the same length as t_12");
        
        size_t leq_name = 0;
        
        for(size_t i = 0; i < triplets_12.size(); i++) {
            //set the lexicographical names at correct positions:
            //[----names at positions i mod 3 = 1----||----names at positions i mod 3 = 2----]
            if(triplets_12[i] % 3 == 1){
                t_12[triplets_12[i]/3] = leq_name;
            }else{                
                if(t_12.size() % 2 == 0){
                    t_12[t_12.size()/2 + triplets_12[i]/3] = leq_name;
                }else{
                    t_12[t_12.size()/2 + 1 + triplets_12[i]/3] = leq_name;
                }
                
            }
            if(i+1 < triplets_12.size()){
                if(sacabench::util::span(&INPUT_STRING[triplets_12[i]], 3) != sacabench::util::span(&INPUT_STRING[triplets_12[i+1]], 3)){
                    leq_name++;
                }else{ //if lexicographical names are not uniqe set recursion = true
                    recursion = true;
                }
            }
        }  
        //TODO: Abfragen ob INPUT_STRING und triplets_12 out of bounce
    }
    
    template<typename S, typename I>
    void determine_isa(const S& t_12, I& isa_12){
        
        DCHECK_MSG(isa_12.size() == t_12.size(), "isa_12 must have the same length as t_12");
        
        for(size_t i = 0; i < t_12.size(); i++) {
            isa_12[t_12[i]] = i+1;
        }        
    }*/
    
    
    
    class dc3 {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                if(text.size()==0){
                    
                }else if(text.size()==1){
                    out_sa[0] = 0;
                }else{
                    auto modified_text = sacabench::util::make_container<sacabench::util::character>(text.size());
                    
                    for(size_t i = 0; i < text.size() ; i++){
                        modified_text[i] = text[i];
                    }
                    modified_text.push_back('\0');
                    modified_text.push_back('\0');
                    
                    construct_sa_dc3<size_t, false, sacabench::util::character>(modified_text, alphabet_size, out_sa);
                }
            }
            
            
            
            
    private:
        template<typename C, typename T, typename S>
        static void determine_triplets(const T& INPUT_STRING, S& triplets_12) {
            
            
            DCHECK_MSG(triplets_12.size() == 2*(INPUT_STRING.size()-2)/3, "triplets_12 must have the length (2*INPUT_STRING.size()/3)");
            
            
            const unsigned char SMALLEST_CHAR = '\0';
            //Container to store all tuples with the same length as triplets_12
            //Tuples contains three chararcters (triplet) and the start position i mod 3 != 0 
            auto triplets_12_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, size_t>>(2*(INPUT_STRING.size()-2)/3);
    
            size_t counter = 0;
            
            //--------------------------------hier ohne extra Sentinals--------------------------------------//
   
            /*for(size_t i = 1; i < INPUT_STRING.size(); i++) {
                if(i % 3 != 0){
                    if((i+2) >= INPUT_STRING.size()){
                        if((i+1) >= INPUT_STRING.size()){
                            triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                        }else{
                            triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                        }
                    }else{
                        triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                    }
                }
            }*/  
            
            //-----------------------------------------------------------------------------------------------//
            
            
            for(size_t i = 1; i < INPUT_STRING.size()-2; i++) {
                if(i % 3 != 0){
                    //std::cout << i << ": " << INPUT_STRING[i] << " " << INPUT_STRING[i+1] << " " << INPUT_STRING[i+2] << std::endl;
                    
                    triplets_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }
            }
            
            //TODO: sort Tupels with radix_sort
            //radix_sort(sa0_to_be_sorted, sa0);        
            std::sort(triplets_12_to_be_sorted.begin(),triplets_12_to_be_sorted.end());
            
            
            for(size_t i = 0; i<triplets_12_to_be_sorted.size();i++){   
                triplets_12[i] = std::get<3>(triplets_12_to_be_sorted[i]);
            }
                    
        }
        
        template<typename T, typename S>
        static void determine_leq(const T& INPUT_STRING, const S& triplets_12, S& t_12, bool& recursion){
            
            DCHECK_MSG(triplets_12.size() == t_12.size(), "triplets_12 must have the same length as t_12");
            
            size_t leq_name = 1;

            
            for(size_t i = 0; i < triplets_12.size(); i++) {
                //set the lexicographical names at correct positions:
                //[----names at positions i mod 3 = 1----||----names at positions i mod 3 = 2----]
                if(triplets_12[i] % 3 == 1){
                    t_12[triplets_12[i]/3] = leq_name;
                }else{
                    if(t_12.size() % 2 == 0){
                        t_12[t_12.size()/2 + triplets_12[i]/3] = leq_name;
                    }else{
                        t_12[t_12.size()/2 + 1 + triplets_12[i]/3] = leq_name;
                    }
                }

                if(i+1 < triplets_12.size()){
                    if(sacabench::util::span(&INPUT_STRING[triplets_12[i]], 3) != sacabench::util::span(&INPUT_STRING[triplets_12[i+1]], 3)){
                        leq_name++;
                    }else{ //if lexicographical names are not uniqe set recursion = true
                        recursion = true;
                    }
                }
                
                //--------------------------------hier ohne extra Sentinals--------------------------------------//
                
                /*if(i+1 < triplets_12.size()){
                    
                    if(triplets_12[i]+3 < INPUT_STRING.size()){
                        
                        if(triplets_12[i+1]+3 < INPUT_STRING.size(){
                            if(sacabench::util::span(&INPUT_STRING[triplets_12[i]], 3) != sacabench::util::span(&INPUT_STRING[triplets_12[i+1]], 3)){
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
            //TODO: Abfragen ob INPUT_STRING und triplets_12 out of bounce
        }
        
        template<typename S, typename I>
        static void determine_isa(const S& t_12, I& isa_12){
            
            DCHECK_MSG(isa_12.size() == t_12.size(), "isa_12 must have the same length as t_12");
            
            for(size_t i = 0; i < t_12.size(); i++) {
                isa_12[t_12[i]-1] = i+1;
            }
            
        }
        
        template<typename S, typename I>
        static void determine_sa(const S& t_12, I& isa_12){
            
            DCHECK_MSG(isa_12.size() == t_12.size(), "isa_12 must have the same length as t_12");
            
            for(size_t i = 0; i < t_12.size(); i++) {
                isa_12[t_12[i]] = i+1;
            }
            
        }
        
        template<typename sa_index, bool rec, typename C, typename S>
        static void construct_sa_dc3(S text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                if(rec){
                    std::cout << std::endl <<"Wir befinden uns in der Rekursion" << std::endl;
                }else{
                    std::cout << std::endl << "Wir befinden uns in der Hauptmethode" << std::endl;
                }
                
                std::cout << "text:     ";
                for(size_t i = 0; i < text.size()-2; i++){
                    std::cout << text[i];
                }
                std::cout << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                
                //empty container which will contain indices of triplet 
                //at positions i mod 3 != 0
                auto triplets_12 = sacabench::util::make_container<size_t>(2*(text.size()-2)/3);
                
                //determine positions and calculate the sorted order
                determine_triplets<C>(text, triplets_12);   
                
                //empty SA which should be filled correctly with lexicographical 
                //names of triplets
                auto t_12 = sacabench::util::make_container<size_t>(2*(text.size()-2)/3);
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                std::cout << "triplets:    ";
                for(size_t i = 0; i < triplets_12.size() ; i++){
                        std::cout << triplets_12[i] << " ";
                }
                std::cout << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                
                //bool which will be set true in determine_leq if the names are not unique
                bool recursion = false;
                                
                //fill t_12 with lexicographical names
                determine_leq(text, triplets_12, t_12, recursion);
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                std::cout << "t_12:    ";
                for(size_t i = 0; i < t_12.size() ; i++){
                        std::cout << t_12[i]-1 << " ";
                }
                std::cout << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                util::span<sa_index> sa_12 =  util::span(&out_sa[0], t_12.size());
            
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                std::cout << "vor der Rekursion" << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //run the algorithm recursivly if the names are not unique
                if(recursion){
                    
                    //add two sentinals to the end of the text
                    auto modified_text = sacabench::util::make_container<size_t>(t_12.size()+2);

                    for(size_t i = 0; i < t_12.size() ; i++){
                        modified_text[i] = t_12[i]-1;
                    }
                    modified_text[modified_text.size()-2] = 0;
                    modified_text[modified_text.size()-1] = 0;
                    
                    //run algorithm recursive
                    construct_sa_dc3<size_t, true, size_t>(modified_text, alphabet_size, sa_12);              
                }
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                    std::cout << "nach der Rekursion" << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //empty isa_12 which should be filled correctly with method determine_isa
                auto isa_12 = sacabench::util::make_container<size_t>(0);
                
                //empty merge_isa_12 to be filled with inverse suffix array in format for merge_sa_dc
                auto merge_isa_12 = sacabench::util::make_container<size_t>(triplets_12.size());
                
                //if in recursion use temporary sa. Otherwise t_12
                if(recursion){
                    
                    isa_12 = sacabench::util::make_container<size_t>(sa_12.size());
                    determine_sa(sa_12, isa_12);
                    
                    if(OUTPUT){
                    std::cout << "sa_12:    ";
                    for(size_t i = 0; i < sa_12.size() ; i++){
                            std::cout << sa_12[i] << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "isa_12:    ";
                    for(size_t i = 0; i < isa_12.size() ; i++){
                            std::cout << isa_12[i] << " ";
                    }
                    std::cout << std::endl;
                    }
                    //index of the first value which represents the positions i mod 3 = 2
                    size_t end_of_mod_eq_1 = triplets_12.size()/2; // + ((triplets_12.size()/2) % 2 == 0);
                    if(triplets_12.size() % 2 != 0){
                        end_of_mod_eq_1++;
                    }
                    
                    //correct the order of sa_12 with result of recursion
                    for(size_t i = 0; i < triplets_12.size(); i++){
                        if(i < end_of_mod_eq_1){
                            triplets_12[isa_12[i]-1] = 3 * i + 1; 
                        }else{
                            triplets_12[isa_12[i]-1] = 3 * (i-end_of_mod_eq_1) + 2;
                        }
                    }
                    
                    //Anfang Debug-Informationen------------------------------------------------------------------------
                    if(OUTPUT){
                        std::cout << "korrigierte triplets:    ";
                        for(size_t i = 0; i < triplets_12.size() ; i++){
                                std::cout << triplets_12[i] << " ";
                        }
                        std::cout << std::endl;
                    }
                    //Ende Debug-Informationen------------------------------------------------------------------------
                    
                    //convert isa_12 to the correct format for merge_sa_dc.
                    auto merge_isa_12_to_be_sorted = sacabench::util::make_container<std::tuple<size_t, size_t>>(triplets_12.size());
                    for(size_t i = 0; i < merge_isa_12.size(); i++){
                        merge_isa_12_to_be_sorted[i] = (std::tuple<size_t, size_t>(triplets_12[i], i));
                    }
                    std::sort(merge_isa_12_to_be_sorted.begin(),merge_isa_12_to_be_sorted.end());
                    for(size_t i = 0; i<merge_isa_12_to_be_sorted.size();i++){
                        merge_isa_12[i] = std::get<1>(merge_isa_12_to_be_sorted[i]);
                    }
                }else{
                    isa_12 = t_12;
                    determine_isa(isa_12, sa_12);

                    //convert isa_12 to the correct format for merge_sa_dc.
                    size_t counter = 0;
                    size_t half = isa_12.size()/2 + (((isa_12.size()% 2)!=0));
                    
                    for(size_t i = 0; i<isa_12.size()/2;i++){
                        merge_isa_12[counter++] = isa_12[i]-1;
                        merge_isa_12[counter++] = isa_12[half + i]-1;
                        
                    }
                    if(isa_12.size() % 2 != 0){
                        merge_isa_12[counter] = isa_12[isa_12.size()/2]-1;
                    }
                    
                    //Anfang Debug-Informationen------------------------------------------------------------------------
                    if(OUTPUT){
                        std::cout << "isa_12:    ";
                        for(size_t i = 0; i < isa_12.size() ; i++){
                                std::cout << isa_12[i]-1 << " ";
                        }
                        std::cout << std::endl;
                    }
                //Ende Debug-Informationen------------------------------------------------------------------------
                }
                
                //characters of positions i mod 3 = 0 of text
                auto t_0 = sacabench::util::make_container<C>((text.size()-2)/3 + ((text.size()-2)%3!=0));

                size_t counter = 0;
                for(size_t i = 0; i < (text.size()-2); i+=3){
                    t_0[counter++] =  text[i];
                }
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                    std::cout << "t_0:    ";
                    for(size_t i = 0; i < t_0.size() ; i++){
                            std::cout << t_0[i] << " ";
                    }
                    std::cout << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                
                //empty sa_0 which should be filled correctly with method induce_sa_dc
                auto sa_0 = sacabench::util::make_container<size_t>(t_0.size());
                
                //fill sa_0 by inducing with characters at i mod 3 = 0 and ranks of triplets 
                //beginning in positions i mod 3 != 0
                if(recursion){
                    sacabench::util::induce_sa_dc<C>(t_0, isa_12, sa_0);
                }else{
                    sacabench::util::induce_sa_dc<C>(t_0, isa_12, sa_0);
                }
                
                
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
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
                
                std::cout << "sa_12:   ";
                for(size_t i = 0; i < triplets_12.size(); i++){
                    std::cout << triplets_12[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "isa_12:   ";
                
                    for(size_t i = 0; i < merge_isa_12.size(); i++){
                        std::cout << merge_isa_12[i] << " ";
                    }
                
                std::cout << std::endl;
                }
                //Ende Debug-Informationen------------------------------------------------------------------------
                
                //merging the SA's of triplets in i mod 3 != 0 and ranks of i mod 3 = 0
                if constexpr(rec){ 
                    sacabench::util::merge_sa_dc<const size_t>(sacabench::util::span(&text[0], text.size()-2), sa_0, triplets_12,
                        merge_isa_12, out_sa, comp_recursion, get_substring_recursion);
                }else{
                    sacabench::util::merge_sa_dc<const sacabench::util::character>(sacabench::util::span(&text[0], text.size()-2), sa_0, triplets_12,
                        merge_isa_12, out_sa, comp, get_substring);
                }
                
                //Anfang Debug-Informationen------------------------------------------------------------------------
                if(OUTPUT){
                std::cout << "sa:   ";
                for(size_t i = 0; i < out_sa.size(); i++){
                    std::cout << out_sa[i] << " ";
                }
                std::cout << std::endl << std::endl;
                }
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
    }; // class dc3
        
}  // namespace sacabench::util

