/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/sort/ternary_quicksort.hpp>


namespace sacabench::qsufsort {
    
    struct compare_first_character{
    public:
        compare_first_character(const util::string_span &_input_text):
            input_text(_input_text) {}
        template<typename sa_index>        
        int operator ()(const sa_index &a,const sa_index &b) {
            return input_text[a]-input_text[b];
        }
        const util::string_span &input_text;

    };
    
    class qsufsort {
        public:            
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::span<sa_index> out_sa) {
                
                size_t n = text.size();
                auto V = util::make_container<size_t>(n);
                auto L = util::make_container<ssize_t>(n);
                for(int i=0;i<n;++i){
                    out_sa[i]=i;
                }
                
                size_t h = 0;
                bool is_sorted= false;
                auto cmp = compare_first_character(text);                
                std::cout<<text<<std::endl;    
                
                //Sort according to first character
                util::sort::ternary_quicksort::ternary_quicksort(out_sa,cmp);
                
                for(size_t elem:out_sa){
                    std::cout<<elem<<"("<<text[elem]<<")"<<std::endl;
                }
                std::cout<<"_________"<<std::endl;
                calculate_additional_arrays(text,out_sa,V,L,h);  
                for(int i=0;i<V.size();i++)
                {
                    std::cout<<V[out_sa[i]]<<std::endl;
                }
                std::cout<<"_________"<<std::endl;
                for(auto elem:L){
                    std::cout<<elem<<", ";
                }
                std::cout<<std::endl;
                /*
                while(!is_sorted) {
                //if not sorted, double h, start again
                    //Update V und L
                    //Sort unsorted groups
                    //double h
                }
                */
            }
        private:
            template<typename sa_index>
            static void calculate_additional_arrays(util::string_span text,util::span<sa_index> out_sa, util::container<size_t> &V,util::container<ssize_t> &L, size_t h) {
                size_t n= out_sa.size();
                //TODO Remove if use sentinal
                size_t unsorted_counter=0;
                size_t sorted_counter = 0;
                bool sorted_group_started=false;
                size_t dif=0;
                V[out_sa[n-1]]=n-1;
                for(size_t i=n-2;i<n;--i) {
                     if(text[out_sa[i+1+h]]==text[out_sa[i+h]]) {
                     
                         V[out_sa[i]]=V[out_sa[i+1]];
                    }
                    else
                    {
                        V[out_sa[i]]=i;
                    }
                    
                    
                    //Calculate L                    
                    //use difference instead
                    dif=V[out_sa[i+1]]-V[out_sa[i]];
                    
                    //count for last position..
                    if(dif==0) {
                        ++unsorted_counter;
                    }
                    else
                    {
                        unsorted_counter=0;
                    }
                    
                    
                    if(dif==1) {
                        ++sorted_counter;
                        sorted_group_started=true;
                    }
                    else
                    {
                        if(sorted_group_started) {
                            L[i+2]= -sorted_counter;
                            sorted_counter=0;
                            sorted_group_started=false;
                        }
                        else {
                            L[i+1]=V[out_sa[i+1]]-V[out_sa[i]];
                        }
                    }
                }
                //easier if use sentinal...
                if(V[out_sa[0]]==V[out_sa[1]]) {
                    L[0]=++unsorted_counter;
                }
                else {
                    L[0]=-(++sorted_counter);
                }
            }
        
    }; // class qsufsort

} // namespace sacabench::qsufsort
