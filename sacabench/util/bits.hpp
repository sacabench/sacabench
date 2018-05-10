/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <climits>

namespace sacabench::util {
    /// Generic constant equal to the size of a type in bits.
    ///
    /// Example:
    /// ```cpp
    /// ASSERT_EQ(util::bits_of<uint64_t>, 64);
    /// ```
    template<typename type>
    constexpr size_t bits_of = sizeof(type) * CHAR_BIT;

}
