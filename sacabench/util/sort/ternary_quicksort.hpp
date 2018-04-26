#pragma once

namespace util {
namespace sort {
namespace ternary_quicksort {

// This should swap elements in array such that a correct ternary
// partitioning is created. The function returns the two bounds for the
// partitiongs, i and j. [0; i) is smaller than the partition [i, j), and
// the partition [j, n) is larger than the other partitions.
template<typename content_t, typename key_func_type>
std::pair<size_t, size_t> partition(span<content_t> array,
                                    const key_func_type& key_func,
                                    size_t pivot_element);

// This swaps elements until the array is sorted.
template<typename content_t, typename key_func_type>
void ternary_quicksort(span<content_t> array,
                       const key_func_type& key_func);

}
}
}
