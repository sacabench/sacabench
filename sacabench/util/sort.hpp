/*******************************************************************************
 * util/sort.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

namespace sacabench::util {

template <typename DataType>
void sort(std::vector<DataType>& data) {
  std::sort(std::begin(data), std::end(data));
}

} // namespace sacabench::util

/******************************************************************************/
