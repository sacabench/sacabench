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
sacabench::util::string_span get_substring(const sacabench::util::string& t, const sacabench::util::character* ptr,
        int n) {
    // Suppress unused variable warnings:
    (void) t;        
            
    return sacabench::util::span(ptr, n);
}

TEST(DC, merge) {
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");

    //initialize suffix array and inverse suffix array
    auto sa_0 = sacabench::util::container<size_t> { 12,9,3,6,0 };
    auto sa_12 = sacabench::util::container<size_t> { 14,13,7,1,8,2,10,4,11,5 };
    auto isa_12 = sacabench::util::container<size_t> { 3,5,7,9,2,4,6,8,1,0 };

    //empty SA which should be filled correctly with method merge_sa_dc
    auto sa = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    //run method to test it
    sacabench::util::merge_sa_dc<const sacabench::util::character>(input_string, sa_0, sa_12,
            isa_12, sa, std::less<sacabench::util::string_span>(), get_substring);

    //expected values for merged SA with DC
    auto expected = sacabench::util::container<size_t> {14,13,12,7,1,8,2,10,4,
            9,3,11,6,0,5};

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}

TEST(DC, merge_parallel) {
sacabench::util::string input_string_cont = sacabench::util::make_string("caabaccaabacaa$$");
    sacabench::util::string_span input_string = input_string_cont;

    //initialize suffix array and inverse suffix array
    auto sa_0_cont = sacabench::util::container<size_t> { 12,9,3,6,0 };
    sacabench::util::span<size_t> sa_0 = sa_0_cont;
    auto sa_12_cont = sacabench::util::container<size_t> { 14,13,7,1,8,2,10,4,11,5 };
    sacabench::util::span<size_t> sa_12 = sa_12_cont;
    auto isa_12 = sacabench::util::container<size_t> { 3,5,7,9,2,4,6,8,1,0 };

    //empty SA which should be filled correctly with method merge_sa_dc
    auto sa_cont = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    sacabench::util::span<size_t> sa = sa_cont;
    
    auto comp = [&](size_t a, size_t b) {
        sacabench::util::string_span t_0;
        sacabench::util::string_span t_12;
        if (b % 3 == 1) {
            t_0 = input_string.slice(a, a+1);
            t_12 = input_string.slice(b, b+1);
        } else {
            t_0 = input_string.slice(a, a+2);
            t_12 = input_string.slice(b, b+2);
        }
        
        const bool less_than = t_0 < t_12;
        const bool eq = t_0 == t_12;
        auto lesser_suf = 
                    !((2 * (b + t_12.size())) / 3 >=
                     isa_12.size()) && // if index to compare for t_12 is
                                       // out of bounds of isa then sa_0[i]
                                       // is never lexicographically smaller
                                       // than sa_12[j]
                   ((2 * (a + t_0.size())) / 3 >=
                        isa_12
                            .size() || // if index to compare for t_0 is out
                                       // of bounds of isa then sa_0[i] is
                                       // lexicographically smaller
                    isa_12[(2 * (a + t_0.size())) / 3] <
                        isa_12[2 * ((b + t_12.size())) / 3]);
                        
        return less_than || (eq && lesser_suf);
    };

    //run method to test it
    sacabench::util::merge_sa_dc_parallel(sa_0, sa_12, sa, comp);

    //expected values for merged SA with DC
    auto expected = sacabench::util::container<size_t> {14,13,12,7,1,8,2,10,4,
            9,3,11,6,0,5};
    
    std::cout << "expected: " << expected << std::endl;   
    std::cout << "result: " << sa << std::endl;

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}

TEST(DC, merge_parallel_opt) {
    sacabench::util::string input_string_cont = sacabench::util::make_string("caabaccaabacaa$$");
    sacabench::util::string_span input_string = input_string_cont;

    //initialize suffix array and inverse suffix array
    auto sa_0_cont = sacabench::util::container<size_t> { 12,9,3,6,0 };
    sacabench::util::span<size_t> sa_0 = sa_0_cont;
    auto sa_12_cont = sacabench::util::container<size_t> { 14,13,7,1,8,2,10,4,11,5 };
    sacabench::util::span<size_t> sa_12 = sa_12_cont;
    auto isa_12 = sacabench::util::container<size_t> { 3,5,7,9,2,4,6,8,1,0 };

    //empty SA which should be filled correctly with method merge_sa_dc
    auto sa_cont = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    sacabench::util::span<size_t> sa = sa_cont;
    
    auto comp = [&](size_t a, size_t b) {
        sacabench::util::string_span t_0;
        sacabench::util::string_span t_12;
        if (b % 3 == 1) {
            t_0 = input_string.slice(a, a+1);
            t_12 = input_string.slice(b, b+1);
        } else {
            t_0 = input_string.slice(a, a+2);
            t_12 = input_string.slice(b, b+2);
        }
        
        const bool less_than = t_0 < t_12;
        const bool eq = t_0 == t_12;
        auto lesser_suf = 
                    !((2 * (b + t_12.size())) / 3 >=
                     isa_12.size()) && // if index to compare for t_12 is
                                       // out of bounds of isa then sa_0[i]
                                       // is never lexicographically smaller
                                       // than sa_12[j]
                   ((2 * (a + t_0.size())) / 3 >=
                        isa_12
                            .size() || // if index to compare for t_0 is out
                                       // of bounds of isa then sa_0[i] is
                                       // lexicographically smaller
                    isa_12[(2 * (a + t_0.size())) / 3] <
                        isa_12[2 * ((b + t_12.size())) / 3]);
                        
        return less_than || (eq && lesser_suf);
    };

    //run method to test it
    sacabench::util::merge_sa_dc_parallel_opt(sa_0, sa_12, sa, comp);

    //expected values for merged SA with DC
    auto expected = sacabench::util::container<size_t> {14,13,12,7,1,8,2,10,4,
            9,3,11,6,0,5};
    
    std::cout << "expected: " << expected << std::endl;   
    std::cout << "result: " << sa << std::endl;

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}
