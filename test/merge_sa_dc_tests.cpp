/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/merge_sa_dc.hpp>
#include <util/string.hpp>
#include <util/container.hpp>

//implementation of get-substring method
span<const unsigned char> get_substring(const sacabench::util::string& t, const unsigned char* ptr,
        int n) {
    return span(ptr, n);
}

// implementation of comp method
bool comp(const span<const unsigned char>& a, const span<const unsigned char>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (b.at(i) == '$') {
            return false;
        }
        else if (a.at(i) == '$' || a.at(i) < b.at(i)) {
            return true;
        }
        else if (a.at(i) > b.at(i)) {
            return false;
        }
    }
    return false;
}

TEST(DC, merge) {    
    sacabench::util::string input_string = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a', '$'};
    
    //initialize suffix array and inverse suffix array
    auto sa_0 = sacabench::util::container<size_t> { 12,9,3,6,0 };
    auto sa_12 = sacabench::util::container<size_t> { 14,13,7,1,8,2,10,4,11,5 };
    auto isa_12 = sacabench::util::container<size_t> { 3,5,7,9,2,4,6,8,1,0 };
    
    //empty SA which should be filled correctly with method merge_sa_dc
    auto sa = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::util::merge_sa_dc<unsigned char>(input_string, sa_0, sa_12, 
            isa_12, sa, comp, get_substring);

    //expected values for merged SA with DC
    auto expected = sacabench::util::container<size_t> {14,13,12,7,1,8,2,10,4,
            9,3,11,6,0,5};
    
    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}
