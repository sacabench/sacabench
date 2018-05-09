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
    sacabench::util::string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);

    sacabench::util::container<uint8_t> sa =
        sacabench::util::make_container<uint8_t>(input.size());
    sacabench::util::sort::bucketsort_presort(input, a.size, 2, sa);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}

TEST(Bucketsort, recursiv_bucket_sort_test) {

    using namespace sacabench::util;
    container<string> input = make_container<string>(0);

    string firstString = {'m', 'm', 'm'};
    string secondString = {'a', 'b', 'c'};
    string thirdString = {'x', 'y', 'z', 'z'};
    string fourthString = {'m', 'b', 'm'};
    string fifthString = {'a', 'm', 'a'};
    string sixthString = {'m', 'b', 'a'};
    string seventhString = {'a', 'b', 'e'};
    string eightString = {'x', 'y', 'z', 'a'};

    input.push_back(firstString);
    input.push_back(secondString);
    input.push_back(thirdString);
    input.push_back(fourthString);
    input.push_back(fifthString);
    input.push_back(sixthString);
    input.push_back(seventhString);
    input.push_back(eightString);

    auto result = make_container<string>(0);
    sort::bucket_sort(input, 1, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), fifthString);
    ASSERT_EQ(result.at(2), seventhString);
    ASSERT_EQ(result.at(3), firstString);
    ASSERT_EQ(result.at(4), fourthString);
    ASSERT_EQ(result.at(5), sixthString);
    ASSERT_EQ(result.at(6), thirdString);
    ASSERT_EQ(result.at(7), eightString);

    result = make_container<string>(0);
    sort::bucket_sort(input, 2, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), seventhString);
    ASSERT_EQ(result.at(2), fifthString);
    ASSERT_EQ(result.at(3), fourthString);
    ASSERT_EQ(result.at(4), sixthString);
    ASSERT_EQ(result.at(5), firstString);
    ASSERT_EQ(result.at(6), thirdString);
    ASSERT_EQ(result.at(7), eightString);

    result = make_container<string>(0);
    sort::bucket_sort(input, 3, result);

    ASSERT_EQ(result.at(0), secondString);
    ASSERT_EQ(result.at(1), seventhString);
    ASSERT_EQ(result.at(2), fifthString);
    ASSERT_EQ(result.at(3), sixthString);
    ASSERT_EQ(result.at(4), fourthString);
    ASSERT_EQ(result.at(5), firstString);
    ASSERT_EQ(result.at(6), thirdString);
    ASSERT_EQ(result.at(7), eightString);

    result = make_container<string>(0);
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


