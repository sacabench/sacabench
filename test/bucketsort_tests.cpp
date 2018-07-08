/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <util/string.hpp>
#include <util/sort/bucketsort.hpp>

TEST(Bucketsort, function_call) {
    sacabench::util::string input =
        sacabench::util::make_string("caabaccaabacaa");
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);

    auto sa = sacabench::util::make_container<uint8_t>(input.size());
    sacabench::util::span<uint8_t> sa_span = sa;
    sacabench::util::sort::bucketsort_presort(input,
            a.size_without_sentinel(), 2, sa_span);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}

TEST(Bucketsort, function_call_lightweight) {
    sacabench::util::string input =
        sacabench::util::make_string("caabaccaabacaa");
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);

    auto sa = sacabench::util::make_container<uint8_t>(input.size());
    sacabench::util::span<uint8_t> sa_span = sa;
    sacabench::util::sort::bucketsort_presort_lightweight(input,
            a.size_without_sentinel(), 2, sa_span);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}

TEST(Bucketsort, bucket_sizes){
    sacabench::util::string input = "blablablub"_s;
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);
    size_t depth = 1;
    auto buckets = sacabench::util::sort::get_buckets(input, a.max_character_value(), depth);
    ASSERT_EQ(buckets.at(0).count, (size_t) 0);
    ASSERT_EQ(buckets.at(1).count, (size_t) 2);
    ASSERT_EQ(buckets.at(2).count, (size_t) 4);
    ASSERT_EQ(buckets.at(3).count, (size_t) 3);
    ASSERT_EQ(buckets.at(4).count, (size_t) 1);
}

TEST(Bucketsort, lightweight_bucket_positions){
    sacabench::util::string input = "blablablub"_s;
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);
    size_t depth = 1;
    auto buckets = sacabench::util::sort::get_lightweight_buckets(input, a.max_character_value(), depth);
    ASSERT_EQ(buckets.at(0), (size_t) 0);
    ASSERT_EQ(buckets.at(1), (size_t) 0);
    ASSERT_EQ(buckets.at(2), (size_t) 2);
    ASSERT_EQ(buckets.at(3), (size_t) 6);
    ASSERT_EQ(buckets.at(4), (size_t) 9);
    ASSERT_EQ(buckets.at(5), (size_t) 10);
}

TEST(Bucketsort, sentinels){
    sacabench::util::string input = {'\0', '\0', '\0', '\0'};
    size_t depth = 2;
    auto buckets = sacabench::util::sort::get_buckets(input, 0, depth);
    ASSERT_EQ(buckets.size(), (size_t) 1);
    ASSERT_EQ(buckets.at(0).count, (size_t) 4);
}

TEST(Bucketsort, recursiv_bucket_sort_test) {
    using namespace sacabench::util;

    string firstString = "mmm"_s;
    string secondString = "abc"_s;
    string thirdString = "xyzz"_s;
    string fourthString = "mbm"_s;
    string fifthString = "ama"_s;
    string sixthString = "mba"_s;
    string seventhString = "abe"_s;
    string eightString = "xyza"_s;

    container<string> input {
        firstString,
        secondString,
        thirdString,
        fourthString,
        fifthString,
        sixthString,
        seventhString,
        eightString,
    };

    auto result = make_container<string>(input.size());
    sort::bucket_sort(input, 1, result);

    auto should_be = container<string> {
        secondString,
        fifthString,
        seventhString,
        firstString,
        fourthString,
        sixthString,
        thirdString,
        eightString,
    };
    ASSERT_EQ(result, should_be);

    result = make_container<string>(input.size());
    sort::bucket_sort(input, 2, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), seventhString);
    ASSERT_EQ(result.at(2), fifthString);
    ASSERT_EQ(result.at(3), fourthString);
    ASSERT_EQ(result.at(4), sixthString);
    ASSERT_EQ(result.at(5), firstString);
    ASSERT_EQ(result.at(6), thirdString);
    ASSERT_EQ(result.at(7), eightString);

    result = make_container<string>(input.size());
    sort::bucket_sort(input, 3, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), seventhString);
    ASSERT_EQ(result.at(2), fifthString);
    ASSERT_EQ(result.at(3), sixthString);
    ASSERT_EQ(result.at(4), fourthString);
    ASSERT_EQ(result.at(5), firstString);
    ASSERT_EQ(result.at(6), thirdString);
    ASSERT_EQ(result.at(7), eightString);

    result = make_container<string>(input.size());
    sort::bucket_sort(input, 4, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), seventhString);
    ASSERT_EQ(result.at(2), fifthString);
    ASSERT_EQ(result.at(3), sixthString);
    ASSERT_EQ(result.at(4), fourthString);
    ASSERT_EQ(result.at(5), firstString);
    ASSERT_EQ(result.at(6), eightString);
    ASSERT_EQ(result.at(7), thirdString);
}


