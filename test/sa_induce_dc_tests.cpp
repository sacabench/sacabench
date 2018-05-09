/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/induce_sa_dc.hpp>
#include <util/string.hpp>
#include <util/container.hpp>


TEST(DC, induce) {    
    sacabench::util::string input_string = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
            
    //positions i mod 3 = 0 of input_string
    sacabench::util::string t_0 = {'c', 'b', 'c', 'b', 'a'}; 
    
    //inverse SA of triplets beginning in positions i mod 3 != 0
    auto isa_12 = sacabench::util::container<size_t> {4, 8, 3, 7, 2, 6, 10, 5, 9, 1};
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_0 = sacabench::util::container<size_t> {0,0,0,0,0};
    
    //run method to test it
    sacabench::util::induce_sa_dc<unsigned char>(t_0, isa_12, sa_0);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {4, 3, 1, 2, 0};
    
    //compare results with expected values
    ASSERT_EQ(sa_0[0], expected[0]);
    ASSERT_EQ(sa_0[1], expected[1]);
    ASSERT_EQ(sa_0[2], expected[2]);
    ASSERT_EQ(sa_0[3], expected[3]);
    ASSERT_EQ(sa_0[4], expected[4]);
}

