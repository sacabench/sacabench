/*******************************************************************************
 * sacabench/util/bucket_size.hpp
 * 
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include 'alphabet.hpp'
#include 'string.hpp'

inline int get_bucket_size(const string& input){
    alphabet input_alphabet = alphabet(input);
    apply_effective_alphabet(input, input_alphabet);
    int bucket_size [input_alphabet.size];
    for(const char c: input){
        bucket_size[c]+=1;
    }
    return bucket_size;
}