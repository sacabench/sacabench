/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "gtest/gtest.h"
#include "util/alphabet.hpp"
#include "util/string.hpp"
#include "saca/bucket_pointer_refinement_parallel.hpp"
#include "test/saca.hpp"

using bpr = sacabench::bucket_pointer_refinement_parallel::bucket_pointer_refinement_parallel;

TEST(bucket_pointer_refinement_parallel, test) {
    test::saca_corner_cases<bpr>();
}
