/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/assertions.hpp>
#include <util/sort/tuple_sort.hpp>
#include <util/string.hpp>

TEST(tuple_sort, septuple) {
    using namespace sacabench::util;

    size_t number_of_tuples = 5;
    auto test_tuples = sacabench::util::make_container<
        std::tuple<char, char, char, char, size_t, size_t, size_t>>(
        number_of_tuples);
    auto test_result =
        sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0] = std::tuple<char, char, char, char, size_t, size_t, size_t>(
        'b', 'b', 'g', 'k', 6, 7, 5);
    test_tuples[1] = std::tuple<char, char, char, char, size_t, size_t, size_t>(
        'a', 'b', 'l', 'j', 1, 2, 1);
    test_tuples[2] = std::tuple<char, char, char, char, size_t, size_t, size_t>(
        'g', 'a', 'a', 'a', 4, 5, 8);
    test_tuples[3] = std::tuple<char, char, char, char, size_t, size_t, size_t>(
        'c', 'm', 'f', 'o', 19, 2, 0);
    test_tuples[4] = std::tuple<char, char, char, char, size_t, size_t, size_t>(
        'j', 'u', 'x', 'y', 13, 1, 19);

    radixsort_septuple(test_tuples, test_result);
    for (size_t i = 0; i < number_of_tuples - 1; ++i) {

        ASSERT_TRUE(test_tuples[test_result[i]] <
                    test_tuples[test_result[i + 1]]);
    }
}

TEST(tuple_sort, triple) {
    using namespace sacabench::util;
/*
    size_t number_of_tuples = 5;
    auto test_tuples =
        sacabench::util::make_container<std::tuple<char, char, char>>(
            number_of_tuples);
    auto test_result =
        sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0] = std::tuple<char, char, char>('a', 'x', 'g');
    test_tuples[1] = std::tuple<char, char, char>('a', 'u', 'l');
    test_tuples[2] = std::tuple<char, char, char>('a', 'a', 'a');
    test_tuples[3] = std::tuple<char, char, char>('a', 'a', 'f');
    test_tuples[4] = std::tuple<char, char, char>('a', 'b', 'x');

    radixsort_triple(test_tuples, test_result);
    for (size_t i = 0; i < number_of_tuples - 1; ++i) {

        ASSERT_TRUE(test_tuples[test_result[i]] <
                    test_tuples[test_result[i + 1]]);
    }*/
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1, 0,0,0};
            
    //positions i mod 3 = 0 of input_string
    sacabench::util::container<size_t> positions_mod_12 = {1,2,4,5,7,8,10,11,13,14}; 
    //positions i mod 3 = 0 of input_string
    sacabench::util::container<size_t> result_12 = sacabench::util::container<size_t>(positions_mod_12.size()); 
    
    radixsort_triple(input_string, positions_mod_12, result_12, 4, 2);
    radixsort_triple(input_string, result_12, positions_mod_12, 4, 1);
    radixsort_triple(input_string, positions_mod_12, result_12, 4, 0);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {14, 13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    for(size_t i = 0; i < result_12.size(); ++i){
        ASSERT_EQ(result_12[i], expected[i]);
    }
}

TEST(tuple_sort, tuple) {
    using namespace sacabench::util;

    size_t number_of_tuples = 5;
    auto test_tuples =
        sacabench::util::make_container<std::tuple<char, size_t>>(
            number_of_tuples);
    auto test_result =
        sacabench::util::make_container<size_t>(number_of_tuples);
    test_tuples[0] = std::tuple<char, size_t>('a', 4);
    test_tuples[1] = std::tuple<char, size_t>('a', 3);
    test_tuples[2] = std::tuple<char, size_t>('a', 19);
    test_tuples[3] = std::tuple<char, size_t>('a', 100);
    test_tuples[4] = std::tuple<char, size_t>('b', 12);

    radixsort_tuple(test_tuples, test_result);
    for (size_t i = 0; i < number_of_tuples - 1; ++i) {

        ASSERT_TRUE(test_tuples[test_result[i]] <
                    test_tuples[test_result[i + 1]]);
    }
}

TEST(tuple_sort, radixsort_with_key) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
            
    //positions i mod 3 != 0 of input_string
    sacabench::util::container<size_t> positions_mod_12 = {1,2,4,5,7,8,10,11,13}; 
    //container for sorted indices
    sacabench::util::container<size_t> result_12 = sacabench::util::container<size_t>(positions_mod_12.size()); 
    
    auto key_function = [&](size_t i, size_t p) {
        if (i+p < input_string.size()) {
            return input_string[i+p];
        }
        else { return (size_t)0; }
    };
    
    radixsort_with_key(positions_mod_12, result_12, 4, 2, key_function);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    for(size_t i = 0; i < result_12.size(); ++i){
        ASSERT_EQ(result_12[i], expected[i]);
    }
}

TEST(tuple_sort, msd_radixsort_inplace_with_key) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
            
    // positions i mod 3 != 0 of input_string
    sacabench::util::container<size_t> positions_mod_12 = {1,2,4,5,7,8,10,11,13}; 
    
    auto key_function = [&](size_t i, size_t p) {
        if (i+p < input_string.size()) {
            return input_string[i+p];
        }
        else { return (size_t)0; }
    };
    auto compare_function = [&](size_t i, size_t j, size_t index, size_t length) {
        size_t pos_1 = i+index+1;
        size_t pos_2 = j+index+1;
        sacabench::util::span<size_t> t_1;
        sacabench::util::span<size_t> t_2;
        if (pos_1 <= i+length && pos_1 < input_string.size()) {
            t_1 = input_string.slice(pos_1, pos_1+length);
        }
        else {
            t_1 = sacabench::util::span<size_t>(); 
        }
        if (pos_2 <= j+length && pos_2 < input_string.size()) {
            t_2 = input_string.slice(pos_2, pos_2+length);
        }
        else {
            t_2 = sacabench::util::span<size_t>(); 
        }
        return t_1 < t_2 ;
    };
    
    msd_radixsort_inplace_with_key(positions_mod_12, 4, 0, 2, key_function, compare_function);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    for(size_t i = 0; i < positions_mod_12.size(); ++i){
        ASSERT_EQ(positions_mod_12[i], expected[i]);
    }
}

TEST(tuple_sort, msd_radixsort_with_key) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
            
    //positions i mod 3 != 0 of input_string
    sacabench::util::container<size_t> positions_mod_12 = {1,2,4,5,7,8,10,11,13}; 
    //container for sorted indices
    sacabench::util::container<size_t> result_12 = sacabench::util::container<size_t>(positions_mod_12.size()); 
    
    auto key_function = [&](size_t i, size_t p) {
        if (i+p < input_string.size()) {
            return input_string[i+p];
        }
        else { return (size_t)0; }
    };
    
    radixsort_with_key(positions_mod_12, result_12, 4, 2, key_function);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    for(size_t i = 0; i < result_12.size(); ++i){
        ASSERT_EQ(result_12[i], expected[i]);
    }
}

TEST(tuple_sort, radixsort_for_big_alphabet) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
            
    //positions i mod 3 != 0 of input_string
    sacabench::util::container<size_t> positions_mod_12 = {1,2,4,5,7,8,10,11,13}; 
    //container for bucket_array
    sacabench::util::container<size_t> bucket_array = sacabench::util::container<size_t>(positions_mod_12.size()); 
    
    auto key_function = [&](size_t i, size_t p) {
        if (i+p < input_string.size()) {
            return input_string[i+p];
        }
        else { return (size_t)0; }
    };
    auto compare_function = [&](size_t i, size_t j, size_t index, size_t length) {
        size_t pos_1 = i+index+1;
        size_t pos_2 = j+index+1;
        sacabench::util::span<size_t> t_1;
        sacabench::util::span<size_t> t_2;
        if (pos_1 <= i+length && pos_1 < input_string.size()) {
            t_1 = input_string.slice(pos_1, pos_1+length);
        }
        else {
            t_1 = sacabench::util::span<size_t>(); 
        }
        if (pos_2 <= j+length && pos_2 < input_string.size()) {
            t_2 = input_string.slice(pos_2, pos_2+length);
        }
        else {
            t_2 = sacabench::util::span<size_t>(); 
        }
        return t_1 < t_2 ;
    };
    
    radixsort_for_big_alphabet(positions_mod_12, bucket_array, 4, 0, 2, key_function, compare_function);
    
    //expected values for induced SA with DC
    auto expected = sacabench::util::container<size_t> {13, 1, 7, 2, 8, 10, 4, 11, 5};
    
    //compare results with expected values
    for(size_t i = 0; i < positions_mod_12.size(); ++i){
        ASSERT_EQ(positions_mod_12[i], expected[i]);
    }
}

TEST(tuple_sort, countingsort) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
    sacabench::util::container<size_t> pos = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13};
            
    sacabench::util::container<size_t> buckets = {0, 0, 0, 0};
    
    auto key_function = [&](size_t i) {
        return input_string[i];
    };
    auto bucket_function = [&](size_t key) {
        return key;
    };
    
    countingsort(pos, buckets, key_function, bucket_function);
    
    //expected values for buckets
    auto expected = sacabench::util::container<size_t> {0, 0, 8, 10};
    
    //compare results with expected values
    for(size_t i = 0; i < buckets.size(); ++i){
        ASSERT_EQ(buckets[i], expected[i]);
    }
}

TEST(tuple_sort, partition_out_of_place) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
    sacabench::util::container<size_t> pos = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13};
    sacabench::util::container<size_t> result = {0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0};
            
    sacabench::util::container<size_t> buckets = {0, 0, 0, 0};
    
    auto key_function = [&](size_t i) {
        return input_string[i];
    };
    auto bucket_function = [&](size_t key) {
        return key;
    };
    
    countingsort(pos, buckets, key_function, bucket_function);
    
    partition_out_of_place(pos, result, buckets, key_function, bucket_function);
    
    //expected values for partitioning
    auto expected = sacabench::util::container<size_t> {1, 2, 4, 7, 8, 10, 12, 13, 3, 9,
            0, 5, 6, 11};
    
    //compare results with expected values
    for(size_t i = 0; i < result.size(); ++i){
        ASSERT_EQ(result[i], expected[i]);
    }
}

TEST(tuple_sort, partition_in_place) {
    using namespace sacabench::util;
    
    sacabench::util::container<size_t> input_string = {3, 1, 1, 2, 1, 3, 3,
            1, 1, 2, 1, 3, 1, 1};
    sacabench::util::container<size_t> pos = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13};
            
    sacabench::util::container<size_t> buckets_start = {0, 0, 0, 0};
    sacabench::util::container<size_t> buckets_end = {0, 0, 0, 0};
    
    auto key_function = [&](size_t i) {
        return input_string[i];
    };
    auto bucket_function = [&](size_t key) {
        return key;
    };
    
    countingsort(pos, buckets_start, key_function, bucket_function);
    std::copy(buckets_start.begin(), buckets_start.end(), buckets_end.begin());
    
    partition_in_place(pos, buckets_start, buckets_end, key_function,
        bucket_function);
    
    
    //compare results with expected values
    for(size_t i = 0; i < pos.size()-1; ++i){
        ASSERT_LE(key_function(pos[i]), key_function(pos[i+1]));
    }
}