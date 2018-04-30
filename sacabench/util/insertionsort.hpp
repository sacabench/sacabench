/*******************************************************************************
 * sacabench/util/span.hpp
 *
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#include "span.hpp"

namespace sacabench::util{

  template<typename T> void insertion_sort(span<T> A){
    for(size_t i = 1; i < A.size(); i++) {
      auto to_sort = A[i];
      auto j = i;
      while((j > 0) && (A[j-1] > to_sort)) {
        A[j] = A[j - 1];
        j = j - 1;
      }
      A[j] = to_sort;
    }
  }
}
