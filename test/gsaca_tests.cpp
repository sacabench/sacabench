/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>
#include "test/saca.hpp"

#include <saca/gsaca.hpp>
#include <saca/bucket_pointer_refinement.hpp>
#include <util/string.hpp>

using namespace sacabench::gsaca;

TEST(gsaca, test_corner_cases) {
    test::saca_corner_cases<sacabench::gsaca::gsaca>();
}
/*
auto alphanum = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"_s;
auto stringLength = alphanum.size();
sacabench::util::character genRandom() {
    return alphanum[rand() % stringLength];
}

bool compare_result(sacabench::util::string_span test_string) {

    std::cout << "Current test size: " << test_string.size() << std::endl;

    auto alphabet = sacabench::util::alphabet(test_string);

    auto place_new = sacabench::util::make_container<size_t>(test_string.size());
    auto out_sa_new = sacabench::util::span<size_t>(place_new);
    gsaca::construct_sa(test_string, alphabet, out_sa_new);

    auto place_original = sacabench::util::make_container<size_t>(test_string.size());
    auto out_sa_original = sacabench::util::span<size_t>(place_original);
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement::construct_sa(test_string, alphabet, out_sa_new);

    for (int index = 0; index < test_string.size(); index++) {
        if (out_sa_new[index] != out_sa_original[index]) {
            std::cout << "Calculation failed at position " << index << std::endl;
            std::cout << "Expected result from original: " << out_sa_original[index] << std::endl;
            std::cout << "Calculated result with new: " << out_sa_new[index] << std::endl;
            std::cout << std::endl;
            return false;
        }
    }
    std::cout << std::endl;
    return true;
}

TEST(gsaca, random_strings) {

    size_t min_length = 1;
    size_t max_lenght = 10000000;
    size_t repetitions = 1;

    for(size_t length = min_length; length <= max_lenght; length *= 10){
        auto test_string = sacabench::util::container<sacabench::util::character>(length);

        for(size_t current_repetition = 0; current_repetition < repetitions; current_repetition++) {

            for(size_t index = 0; index < length; index++) {
                test_string[index] =  genRandom();
            }

            auto text = sacabench::util::string(test_string);
            ASSERT_TRUE(compare_result(text));
        }
    }
}
*/