/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "ternary_quicksort.hpp"

namespace util::sort::ternary_quicksort {

// This should swap elements in array such that a correct ternary
// partitioning is created. The function returns the two bounds for the
// partitiongs, i and j. [0; i) is smaller than the partition [i, j), and
// the partition [j, n) is larger than the other partitions.
template<typename content_t, typename key_func_type>
std::pair<size_t, size_t> partition(span<content_t> array,
                                    const key_func_type& key_func,
                                    size_t pivot_element) {
    // Extract key function.
    // Use this to campre the content of array.
    auto cmp = &key_func::compare;
}

// This swaps elements until the array is sorted.
template<typename content_t, typename key_func_type>
void ternary_quicksort(span<content_t> array,
                       const key_func_type& key_func) {
   // Use partitioning to sort the input array.
}

/**\brief Returns pseudo-median according to three values
 * \param key_func key function for comparing elements with min/max methods
 * \param first first element of the array
 * \param middle middle element of the array
 * \param last last element of the array
 * 
 * Chooses the median of the given array by the median-of-three method
 * which chooses the median of the first, middle and last element of the array
 */
template<typename content_t, typename key_func_type>
size_t median_of_three(const key_func_type& key_func, 
		       content_t first, content_t middle, content_t last) {
  return key_func::max(key_func::min(first, middle),key_func::min(key_func::max(first,middle),last));  

}

/**\brief Returns pseudo-median according to nine values
 * \param array array of elements
 * \param key_func key function for comparing elements
 * 
 * Chooses the median of the given array by median-of-nine method 
 * according to Bentley and McIlroy "Engineering a Sort Function". 
 */
template<typename content_t, typename key_func_type>
content_t median_of_nine(span<content_t> array,
                       const key_func_type& key_func) {
  size_t n= array.size();
  //for small arrays, pseudomedian is the mid element
  content_t median= array[n/2];
  if(n>7) {
    //for larger arrays, choose median_of_three
    content_t lower = array[0];
    content_t upper = array[n-1];
    if(n>40) {
      //for large arrays, choose median_of_nine
      size_t step = (n/8);
      lower = median_of_three(key_func, array[step],array[2*step]);
      median= median_of_three(key_func,array[(n/2)-step],median, array[(n/2)+step]);
      upper = median_of_three(key_func,array[(n-1)-2*step],array[(n-1)-step],upper);
    }
    median= median_of_three(key_func,lower,median,upper);
  }
  return median;
}
}
