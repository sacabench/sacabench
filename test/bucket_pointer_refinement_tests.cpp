/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "gtest/gtest.h"
#include "util/alphabet.hpp"
#include "util/string.hpp"
#include "saca/bucket_pointer_refinement.hpp"
#include "test/saca.hpp"

using bpr = sacabench::bucket_pointer_refinement::bucket_pointer_refinement;

TEST(bucket_pointer_refinement, function_call) {
    sacabench::util::string input =
        sacabench::util::make_string("aaa");
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    sacabench::util::apply_effective_alphabet(input, a);

    auto sa = sacabench::util::make_container<size_t>(input.size());
    sacabench::util::span<size_t> sa_span = sa;
    bpr::construct_sa(sacabench::util::span(input), a.size, sa_span);

    std::cout << "Suffix Array: ";
    for (auto const& c : sa)
        std::cout << (uint32_t) c << ' ';
    std::cout << std::endl;
}

TEST(bucket_pointer_refinement, test) {
    test::saca_corner_cases<bpr>();
}
