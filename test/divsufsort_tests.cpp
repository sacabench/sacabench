/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/div_suf_sort/saca.hpp>
#include <saca/div_suf_sort/utils.hpp>
#include <saca/div_suf_sort/init.hpp>
#include <saca/div_suf_sort/induce.hpp>
#include <saca/div_suf_sort/rms_sorting.hpp>
#include "test/saca.hpp"

using namespace sacabench::div_suf_sort;
using namespace sacabench;

TEST(DivSufSort, CornerCases) {
    test::saca_corner_cases<sacabench::div_suf_sort::div_suf_sort>();
}
