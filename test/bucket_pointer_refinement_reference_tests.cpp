/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "gtest/gtest.h"
#include "util/alphabet.hpp"
#include "util/string.hpp"
#include "saca/external/bucket_pointer_refinement/bucket_pointer_refinement_wrapper.hpp"
#include "test/saca.hpp"

using bpr_ext = sacabench::bucket_pointer_refinement_ext::bucket_pointer_refinement_ext;

TEST(bucket_pointer_refinement_ext, test) {
    test::saca_corner_cases<bpr_ext>();
}
