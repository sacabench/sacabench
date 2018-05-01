/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <util/sort/bucketsort.hpp>

TEST(Alphabet, function_call) {
    sacabench::util::string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);

    sacabench::util::container<uint8_t> sa =
        sacabench::util::make_container<uint8_t>(input.size());
    sacabench::util::sort::bucketsort(input, a.size, 2, sa);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}
