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

TEST(Bucketsort, other_function_call) {

    using namespace sacabench::util;

    try {
        container<string> input = make_container<string>(0);

        string firstString = {'a', 'b', 'c'};
        string secondString = {'a', 'b', 'e'};
        string thirdString = {'m', 'b', 'm'};
        string fourthString = {'a', 'm', 'a'};
        string fifthString = {'m', 'b', 'a'};

        input.push_back(firstString);
        input.push_back(secondString);
        input.push_back(thirdString);
        input.push_back(fourthString);
        input.push_back(fifthString);

        ASSERT_EQ(input.size(), 5);

        std::cout << "======================================" << std::endl;
        std::cout << "Starting bucket_sort with maxDepth = 1" << std::endl;

        auto result = make_container<container<string>>(0);
        sort::bucket_sort(input, 0, 1, result);

        ASSERT_EQ(result.at(0).at(0), firstString);
        ASSERT_EQ(result.at(0).at(1), secondString);
        ASSERT_EQ(result.at(0).at(2), fourthString);
        ASSERT_EQ(result.at(1).at(0), thirdString);
        ASSERT_EQ(result.at(1).at(1), fifthString);


        std::cout << "======================================" << std::endl;
        std::cout << "Starting bucket_sort with maxDepth = 2" << std::endl;
        result = make_container<container<string>>(0);
        sort::bucket_sort(input, 0, 2, result);

        std::cout << "Result of bucket sort:" << std::endl;
        for (container<string> bucket : result) {
            std::cout << "Current bucket:" << std::endl;
            for (string content : bucket) {
                for (int index = 0; index < content.size(); ++index) {
                    std::cout << content.at(index);
                }
                std::cout << std::endl;
            }
        }
    } catch (std::bad_alloc& ba) {
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }
}


