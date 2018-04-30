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
template<typename content, typename key_func_type>
std::pair<size_t, size_t> partition(span<content> array,
                                    const key_func_type& key_func,
                                    size_t pivot_element) {
    // Extract key function.
    // Use this to campre the content of array.
    auto cmp = &key_func::compare;
    
    //Init values, which keep track of the partition position
    size_t left = 0;
    size_t mid =0;
    size_t right=0;
    for(size_t i =0;i<array.size();i++) {
        //Count Elements in less-Partition
        if(cmp(array[i],pivot_element)<0) {
            mid++;
        }
        //Count Elements in equal partition
        else if(cmp(array[i],pivot_element)==0) {
            right++;
        }
    }
    //Add #elements smaller than pivot to get correct start position
    //for greater-partition counter
    right=right+mid;
    
    //Save these values, because we need to return them afterwards
    size_t i = mid;
    size_t j = right;
    
    //Loop, which builds the less-partition
    while(left<i) {
        //If current element is the pivot_element, swap it into equal-partition
        if(cmp(array[left],pivot_element)==0){
            std::swap(array[left],array[mid]);
            mid++;
        }
        //else if the element belongs in the greater-partition, swap it there
        else if(cmp(array[left],pivot_element)>0) {
            std::swap(array[left],array[right]);
            right++;
        }
        //else, the current element is already at the right place
        else {
            left++;
        }
    }
    //Loop, which builds the equal partition
    while(mid<j) {
        //if current element is bigger than the pivot_element, swap it
        if(cmp(array[mid],pivot_element)>0) {
            std::swap(array[mid],array[right]);
            right++;
        }
        //else, the element is ar the right place
        else {
            mid++;
        }
        //we dont need to consider less elements, because they are already in
        //the right part of the array
    }
    //less- and equal-partitions are built -> greater-partition is built implicitly
    
    
    //return the bounds 
    return std::make_pair(i,j);
            
}

// This swaps elements until the array is sorted.
template<typename content, typename key_func_type>
void ternary_quicksort(span<content> array,
                       const key_func_type& key_func) {
    // Use partitioning to sort the input array.
    
    size_t n = array.size();
    
    if(n==1) {
        return;
    }
    if(n==2) {
        if(key_func::compare(array[0],array[1])>0) {
            std::swap(array[0],array[1]);
        }
        return;
    }
          
    
    //constexpr size_t MEDIAN_OF_THREE_THRESHOLD=7;
    constexpr size_t MEDIAN_OF_NINE_THRESHOLD=40;
    
    

    content pivot =(n>MEDIAN_OF_NINE_THRESHOLD)? median_of_nine(array,key_func_type):median_of_three(array,key_func_type);
    auto result=partition(array,key_func,pivot);
    ternary_quicksort(array(0,result.first));
    ternary_quicksort(array(result.second,n-1));
    return;    

}

/**\brief Returns pseudo-median according to three values
 * \param array array of elements
 * \param key_func key function for comparing elements with min/max methods
 * 
 * Chooses the median of the given array by the median-of-three method
 * which chooses the median of the first, middle and last element of the array
 */
template<typename content, typename key_func_type>
size_t median_of_three(span<content> array, key_func_type& key_func) {
    size_t first = array[0];
    size_t middle = array[(array.size()-1)/2];
    size_t last = array[array.size()-1];
  return key_func::max(key_func::min(first, middle),key_func::min(key_func::max(first,middle),last));  

}

/**\brief Returns pseudo-median according to nine values
 * \param array array of elements
 * \param key_func key function for comparing elements
 * 
 * Chooses the median of the given array by median-of-nine method 
 * according to Bentley and McIlroy "Engineering a Sort Function". 
 * TODO überarbeiten
 */
template<typename content, typename key_func_type>
content median_of_nine(span<content> array,
                       const key_func_type& key_func) {
    size_t n= array.size();
    //for small arrays, pseudomedian is the mid element
    content median= array[n/2];
    if(n>7) {
        //for larger arrays, choose median_of_three
        content lower = array[0];
        content upper = array[n-1];
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
