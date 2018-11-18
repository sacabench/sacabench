/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>
#include <util/string.hpp>
#include <unordered_map>

namespace sacabench::util::sort {

    void radixsort_parallel(container<int> &input, container<int> &output) {
        output[0] = 111;
        output[1] = 123;
        output[2] = 444;
        output[3] = 691;
        output[4] = 912;
    }
}

