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
#include "test/saca.hpp"
#include <iostream>


TEST(DC3, determine_triplets) {
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::dc3::determine_triplets<unsigned char>(input_string, t_12);
    
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    ASSERT_EQ(t_12[0], expected[0]);
    ASSERT_EQ(t_12[1], expected[1]);
    ASSERT_EQ(t_12[2], expected[2]);
    ASSERT_EQ(t_12[3], expected[3]);
    ASSERT_EQ(t_12[4], expected[4]);
    ASSERT_EQ(t_12[5], expected[5]);
    ASSERT_EQ(t_12[6], expected[6]);
    ASSERT_EQ(t_12[7], expected[7]);
    ASSERT_EQ(t_12[8], expected[8]);
    
    
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0};
    
    bool recursion = false;
    //run method to test it
    sacabench::dc3::determine_leq(input_string, t_12, sa_12, recursion);
    
    
    //expected values for induced SA with DC
    auto expected_leq = sacabench::util::container<size_t> {1, 4, 1, 3, 0, 2, 6, 2, 5};
    
    //compare results with expected values
    ASSERT_EQ(sa_12[0], expected_leq[0]);
    ASSERT_EQ(sa_12[1], expected_leq[1]);
    ASSERT_EQ(sa_12[2], expected_leq[2]);
    ASSERT_EQ(sa_12[3], expected_leq[3]);
    ASSERT_EQ(sa_12[4], expected_leq[4]);
    ASSERT_EQ(sa_12[5], expected_leq[5]);
    ASSERT_EQ(sa_12[6], expected_leq[6]);
    ASSERT_EQ(sa_12[7], expected_leq[7]);
    ASSERT_EQ(sa_12[8], expected_leq[8]);
}

TEST(DC3, determine_triplets_with_sentinel) {    
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::dc3::determine_triplets<unsigned char>(input_string, t_12);
    
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {14, 13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    ASSERT_EQ(t_12[0], expected[0]);
    ASSERT_EQ(t_12[1], expected[1]);
    ASSERT_EQ(t_12[2], expected[2]);
    ASSERT_EQ(t_12[3], expected[3]);
    ASSERT_EQ(t_12[4], expected[4]);
    ASSERT_EQ(t_12[5], expected[5]);
    ASSERT_EQ(t_12[6], expected[6]);
    ASSERT_EQ(t_12[7], expected[7]);
    ASSERT_EQ(t_12[8], expected[8]);
    ASSERT_EQ(t_12[9], expected[9]);
}

TEST(DC3, calc_sa) {    
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::dc3::determine_triplets<unsigned char>(input_string, t_12);    
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    bool recursion = false;
    //run method to test it
    sacabench::dc3::determine_leq(input_string, t_12, sa_12, recursion);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {2, 5, 2, 4, 1, 3, 7, 3, 6, 0};
    
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
    
    
    ASSERT_EQ(recursion, true);  
}

TEST(DC3, sa_after_recursion_task_1) {    
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::dc3::determine_triplets<unsigned char>(input_string, t_12);    
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    bool recursion = false;
    //run method to test it
    sacabench::dc3::determine_leq(input_string, t_12, sa_12, recursion);
    
    if(recursion){
        auto r_t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
        
        //run method to test it
        sacabench::dc3::determine_triplets<size_t>(sa_12, r_t_12);    
    
        //empty SA which should be filled correctly with method induce_sa_dc
        auto r_sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
    
        bool recursion = false;
        //run method to test it
        sacabench::dc3::determine_leq(sa_12, r_t_12, r_sa_12, recursion);
    
        //expected values for induced SA with DC
        auto expected_sa = sacabench::util::container<size_t> {4, 0, 2, 1, 3, 5};
        
        //compare results with expected values
        ASSERT_EQ(r_sa_12[0], expected_sa[0]);
        ASSERT_EQ(r_sa_12[1], expected_sa[1]);
        ASSERT_EQ(r_sa_12[2], expected_sa[2]);
        ASSERT_EQ(r_sa_12[3], expected_sa[3]);
        ASSERT_EQ(r_sa_12[4], expected_sa[4]);
        ASSERT_EQ(r_sa_12[5], expected_sa[5]);
        
        
        //empty SA which should be filled correctly with method induce_sa_dc
        auto r_isa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
        sacabench::dc3::determine_isa(r_sa_12, r_isa_12);

        
        //expected values for induced SA with DC
        auto expected_isa = sacabench::util::container<size_t> {2, 4, 3, 5, 1, 6};
        
        //compare results with expected values
        ASSERT_EQ(r_isa_12[0], expected_isa[0]);
        ASSERT_EQ(r_isa_12[1], expected_isa[1]);
        ASSERT_EQ(r_isa_12[2], expected_isa[2]);
        ASSERT_EQ(r_isa_12[3], expected_isa[3]);
        ASSERT_EQ(r_isa_12[4], expected_isa[4]);
        ASSERT_EQ(r_isa_12[5], expected_isa[5]);
    }
    
}

TEST(DC3, dc3_complete) {    
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    //run method to test it
    sacabench::dc3::determine_triplets<unsigned char>(input_string, t_12);    
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0};
    
    bool recursion = false;
    //run method to test it
    sacabench::dc3::determine_leq(input_string, t_12, sa_12, recursion);
    
    if(recursion){
        auto r_t_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
        
        //run method to test it
        sacabench::dc3::determine_triplets<size_t>(sa_12, r_t_12);    
    
        //empty SA which should be filled correctly with method induce_sa_dc
        auto r_sa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
    
        bool recursion = false;
        //run method to test it
        sacabench::dc3::determine_leq(sa_12, r_t_12, r_sa_12, recursion);
    
        //expected values for induced SA with DC
        auto expected_sa = sacabench::util::container<size_t> {4, 0, 2, 1, 3, 5};
               
        
        //empty SA which should be filled correctly with method induce_sa_dc
        auto r_isa_12 = sacabench::util::container<size_t> {0,0,0,0,0,0};
        sacabench::dc3::determine_isa(r_sa_12, r_isa_12);

        
        //expected values for induced SA with DC
        auto expected_isa = sacabench::util::container<size_t> {2, 4, 3, 5, 1, 6};
        
        //positions i mod 3 = 0 of sa_12
        sacabench::util::string r_t_0 = {'2', '4', '7', '0'}; 
        
        //empty SA which should be filled correctly with method induce_sa_dc
        auto r_sa_0 = sacabench::util::container<size_t> {0,0,0,0};
        
        //run method to test it
        sacabench::util::induce_sa_dc<unsigned char>(r_t_0, r_isa_12, r_sa_0);    
        
        //expected values for induced SA with DC
        auto expected = sacabench::util::container<size_t> {3, 0, 1, 2};
        
        //compare results with expected values
        ASSERT_EQ(r_sa_0[0], expected[0]);
        ASSERT_EQ(r_sa_0[1], expected[1]);
        ASSERT_EQ(r_sa_0[2], expected[2]);
        ASSERT_EQ(r_sa_0[3], expected[3]);
    }
    //positions i mod 3 = 0 of input_string
    sacabench::util::string t_0 = {'c', 'b', 'c', 'b', 'a'}; 
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_0 = sacabench::util::container<size_t> {0,0,0,0,0};
    
    //run method to test it
   // sacabench::util::induce_sa_dc<unsigned char>(t_0, isa_12, sa_0);
    
    //TODO: induce und merge aufrufen
    
    /*//expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {9, 4, 2, 0, 7, 5, 3, 1, 8, 6};
    
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
    ASSERT_EQ(sa_12[9], expected[9]);*/
    
}



    
using namespace sacabench::dc3;

TEST(dc3, test) {
    test::saca_corner_cases<dc3>();
}


