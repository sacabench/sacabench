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
            a.size, 2, sa_span);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}

TEST(Bucketsort, bucket_sizes){
    sacabench::util::string input =
        sacabench::util::make_string("blablablub");
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);
    size_t depth = 1;
    auto buckets = sacabench::util::sort::get_buckets(input, a.size, depth);
    ASSERT_EQ(buckets.at(0).count, (size_t) 0);
    ASSERT_EQ(buckets.at(1).count, (size_t) 2);
    ASSERT_EQ(buckets.at(2).count, (size_t) 4);
    ASSERT_EQ(buckets.at(3).count, (size_t) 3);
    ASSERT_EQ(buckets.at(4).count, (size_t) 1);
}


TEST(Bucketsort, recursiv_bucket_sort_test) {
    using namespace sacabench::util;

    string firstString = {'m', 'm', 'm'};
    string secondString = {'a', 'b', 'c'};
    string thirdString = {'x', 'y', 'z', 'z'};
    string fourthString = {'m', 'b', 'm'};
    string fifthString = {'a', 'm', 'a'};
    string sixthString = {'m', 'b', 'a'};
    string seventhString = {'a', 'b', 'e'};
    string eightString = {'x', 'y', 'z', 'a'};

    container<string> input = container<string> {
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


