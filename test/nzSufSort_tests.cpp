/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/nzSufSort.hpp>

using namespace sacabench::nzSufSort;

TEST(nzSufSort, test) {
    test::saca_corner_cases<nzSufSort>();
}
