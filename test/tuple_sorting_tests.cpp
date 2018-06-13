/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/tuple_sort.hpp>
#include <util/assertions.hpp>

TEST(tuple_sort, septuple) {
    using namespace sacabench::util;

    size_t number_of_tuples=5;
    auto test_tuples= sacabench::util::make_container<std::tuple<char,char,char,char,size_t,size_t, size_t>>(number_of_tuples);
    auto test_result= sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0]= std::tuple<char,char,char,char,size_t,size_t,size_t>('b','b','g','k',6,7,5);
    test_tuples[1]= std::tuple<char,char,char,char,size_t,size_t,size_t>('a','b','l','j',1,2,1);
    test_tuples[2]= std::tuple<char,char,char,char,size_t,size_t,size_t>('g','a','a','a',4,5,8);
    test_tuples[3]= std::tuple<char,char,char,char,size_t,size_t,size_t>('c','m','f','o',19,2,0);
    test_tuples[4]= std::tuple<char,char,char,char,size_t,size_t,size_t>('j','u','x','y',13,1,19);

    radixsort_septuple(test_tuples,test_result);
    for(size_t i =0;i<number_of_tuples-1;++i) {
    
        ASSERT_TRUE(test_tuples[test_result[i]]<test_tuples[test_result[i+1]]);
    }
}

TEST(tuple_sort, triple) {
    using namespace sacabench::util;

    size_t number_of_tuples=5;
    auto test_tuples= sacabench::util::make_container<std::tuple<char,char,char>>(number_of_tuples);
    auto test_result= sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0]= std::tuple<char,char,char>('a','x','g');
    test_tuples[1]= std::tuple<char,char,char>('a','u','l');
    test_tuples[2]= std::tuple<char,char,char>('a','a','a');
    test_tuples[3]= std::tuple<char,char,char>('a','a','f');
    test_tuples[4]= std::tuple<char,char,char>('a','b','x');

    radixsort_triple(test_tuples,test_result);
    for(size_t i =0;i<number_of_tuples-1;++i) {
    
        ASSERT_TRUE(test_tuples[test_result[i]]<test_tuples[test_result[i+1]]);
    }
}

TEST(tuple_sort, tuple) {
    using namespace sacabench::util;

    size_t number_of_tuples=5;
    auto test_tuples= sacabench::util::make_container<std::tuple<char,size_t>>(number_of_tuples);
    auto test_result= sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0]= std::tuple<char, size_t>('a',4);
    test_tuples[1]= std::tuple<char, size_t>('a',3);
    test_tuples[2]= std::tuple<char, size_t>('a',19);
    test_tuples[3]= std::tuple<char, size_t>('a',100);
    test_tuples[4]= std::tuple<char, size_t>('b',12);

    radixsort_tuple(test_tuples, test_result);
    for(size_t i =0;i<number_of_tuples-1;++i) {
    
        ASSERT_TRUE(test_tuples[test_result[i]]<test_tuples[test_result[i+1]]);
    }
}    
