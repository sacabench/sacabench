/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>

#pragma once

namespace sacabench::util {
    template<typename T, typename C, typename BA> 
    //template T for input string t_0
    //template C for input characters
    //template BA for bool-array

    /**\Save the L/S-Types of the input text in a given array (boolean: 1 = L, 0 = S)
    * \param t_0 input text 
    * \param alph alphabet
    * \param ba boolean array where the result is saved
    */
    void get_types(T& t_0, C& alph, BA& ba) {

        static_assert(t_0.size() == ba.size(), "t_0 must have the same length as the type array ba");
    
        for(int i = t_0.size() -1 ; i >= 0; i--) {
            ba[i] = (i != t_0.size() -1 && 						// case 1: Symbol is sentinel
					(isLargerChar(t_0[i], t_0[i+1], alph) || 	// case 2: Symbol is larger then following Symbol
					(t_0[i] == t_0[i+1] && ba[i+1] == true)))	// case 3: Symbol is equal to following Symbol
        }  
        
    }

	template<char, char, typename C>
    /** Check if c_1 comes after c_2 in the given alphabet
    */
    bool is_larger_char(char& c_1, char& c_2, C& alph){
    	
        if(c_1 == c_2){
    	    return false;
		}
    	
    	for(int i = 0; i < alph.size(); i++){
    	    if(alph[i] == c_1){
    	        return true;
			}
			if(alph[i] == c_2){
				return false;
			}
		}
	}
}

/******************************************************************************/

