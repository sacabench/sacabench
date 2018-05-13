/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/induce_sa_dc.hpp>
#include <util/string.hpp>
#include <util/container.hpp>
#include <saca/dc3.hpp>
#include <iostream>


TEST(DC3, determine_triplets) {
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::saca::determine_triplets<unsigned char>(input_string, sa_12);
    
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    ASSERT_EQ(sa_12[0], expected[0]);
    ASSERT_EQ(sa_12[1], expected[1]);
    ASSERT_EQ(sa_12[2], expected[2]);
    ASSERT_EQ(sa_12[3], expected[3]);
    ASSERT_EQ(sa_12[4], expected[4]);
    ASSERT_EQ(sa_12[5], expected[5]);
    ASSERT_EQ(sa_12[6], expected[6]);
    ASSERT_EQ(sa_12[7], expected[7]);
    ASSERT_EQ(sa_12[8], expected[8]);
}

TEST(DC3, determine_triplets_with_sentinel) {    
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::saca::determine_triplets<unsigned char>(input_string, sa_12);
    
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {14, 13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    ASSERT_EQ(sa_12[0], expected[0]);
    ASSERT_EQ(sa_12[1], expected[1]);
    ASSERT_EQ(sa_12[2], expected[2]);
    ASSERT_EQ(sa_12[3], expected[3]);
    ASSERT_EQ(sa_12[4], expected[4]);
    ASSERT_EQ(sa_12[5], expected[5]);
    ASSERT_EQ(sa_12[6], expected[6]);
    ASSERT_EQ(sa_12[7], expected[7]);
    ASSERT_EQ(sa_12[8], expected[8]);
    ASSERT_EQ(sa_12[9], expected[9]);
}

