/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <unordered_map>
#include <util/container.hpp>
#include <vector>

#include <util/sort/insertionsort.hpp>

namespace sacabench::util {

// workaround, because you can only access tuples by constant expressions
template <typename T>
size_t get_tuple_entry_septuple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    case 2:
        return size_t(std::get<2>(tuple));
    case 3:
        return size_t(std::get<3>(tuple));
    case 4:
        return size_t(std::get<4>(tuple));
    case 5:
        return size_t(std::get<5>(tuple));
    case 6:
        return size_t(std::get<6>(tuple));
    }
    return -1;
}
template <typename T>
size_t get_tuple_entry_triple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    case 2:
        return size_t(std::get<2>(tuple));
    }
    return -1;
}
template <typename T>
size_t get_tuple_entry_tuple(T tuple, size_t index) {

    switch (index) {
    case 0:
        return size_t(std::get<0>(tuple));
    case 1:
        return size_t(std::get<1>(tuple));
    }
    return -1;
}

template <typename T>
void radixsort_septuple(container<T>& tuples, container<size_t>& result) {

    std::unordered_map<size_t, std::vector<size_t>> buckets;
    std::unordered_map<size_t, std::vector<size_t>> other_buckets;
    size_t number_of_tuples = tuples.size();

    // for iterating, iterator has not correct order
    size_t biggest = 0;
    size_t other_biggest = 0;

    // bucketing according to last number
    for (size_t index = 0; index < number_of_tuples; ++index) {
        auto entry = get_tuple_entry_septuple(tuples[index], 6);
        buckets[entry].push_back(index);
        biggest = (biggest < entry) ? entry : biggest;
    }
    size_t iterations = 1;

    while (iterations != 7) {
        if (iterations % 2 != 0) {
            other_buckets.clear();
            for (size_t i = 0; i <= biggest; i++) {
                for (auto s : buckets[i]) {
                    auto entry =
                        get_tuple_entry_septuple(tuples[s], 6 - iterations);
                    other_buckets[entry].push_back(s);
                    other_biggest =
                        (other_biggest < entry) ? entry : other_biggest;
                }
            }
        } else {
            buckets.clear();
            for (size_t i = 0; i <= other_biggest; i++) {
                for (auto s : other_buckets[i]) {
                    auto entry =
                        get_tuple_entry_septuple(tuples[s], 6 - iterations);
                    buckets[entry].push_back(s);
                    biggest = (biggest < entry) ? entry : biggest;
                }
            }
        }
        ++iterations;
    }
    // collect and save to result
    size_t counter = 0;
    for (size_t i = 0; i <= biggest; i++) {
        for (auto s : buckets[i]) {
            result[counter] = s;
            ++counter;
        }
    }
}

template <typename T, typename S, typename N>
void radixsort_triple(T& text, S& tuples, S& result, const N& alphabet_size, const N position) {
    DCHECK_MSG(tuples.size() == result.size(),
               "tuples must have the same length as result");
               
    auto buckets = util::container<size_t>(alphabet_size);

    for(size_t i = 0; i < tuples.size(); ++i){
        ++buckets[text[tuples[i]+position]];
    }
    
    size_t sum = 0;
    for(size_t i = 0; i < buckets.size(); ++i){
        sum += buckets[i];
        buckets[i] = sum-buckets[i];
    }
        
    for(size_t  i = 0; i < tuples.size(); ++i){
        result[buckets[text[tuples[i]+position]]++] = tuples[i];
    }
}

template <typename T>
void radixsort_tuple(container<T>& tuples, container<size_t>& result) {

    std::unordered_map<size_t, std::vector<size_t>> buckets;
    std::unordered_map<size_t, std::vector<size_t>> other_buckets;
    size_t number_of_tuples = tuples.size();

    // for iterating, iterator has not correct order
    size_t biggest = 0;
    size_t other_biggest = 0;

    // bucketing according to last number
    for (size_t index = 0; index < number_of_tuples; ++index) {
        auto entry = get_tuple_entry_tuple(tuples[index], 1);
        buckets[entry].push_back(index);
        biggest = (biggest < entry) ? entry : biggest;
    }
    for (size_t i = 0; i <= biggest; i++) {
        for (auto s : buckets[i]) {
            auto entry = get_tuple_entry_tuple(tuples[s], 0);
            other_buckets[entry].push_back(s);
            other_biggest = (other_biggest < entry) ? entry : other_biggest;
        }
    }
    // collect and save to result
    size_t counter = 0;
    for (size_t i = 0; i <= other_biggest; i++) {
        for (auto s : other_buckets[i]) {
            result[counter] = s;
            ++counter;
        }
    }
}

template <typename S, typename Key>
/**\brief LSD-RadixSort with an auxiliary array
* \tparam S Type of input to be sorted
* \tparam Key Type of a function for the key_function
* \param input input to be sorted
* \param result container in which the result of the RadixSort is stored
* \param radix_size size of the radix, which is the maximum value for a digit
* \param max_index maximum count of digits of all elements
* \param key_function function which takes as parameter an element of input and 
        the index of the current digit and returns the key which is used for sorting
*
* This function implements simple LSD-RadixSort in an out-of-place variant. The function 
* takes a key_function as a parameter so that the developer can specify the key which is 
* used by the sorting algorithm.
*/
void radixsort_with_key(S& input, S& result, const size_t radix_size, 
        const size_t max_index, Key key_function) {
    DCHECK_MSG(input.size() == result.size(),
               "input must have the same length as result");
               
    auto buckets = util::container<size_t>(radix_size);
    for (size_t p = max_index; p >= 0; --p) {
        // reset buckets
        for (size_t i = 0; i < radix_size; ++i) { buckets[i] = 0; }
        
        // generate Buckets
        auto key_function_count = [&](size_t i) {
            return key_function(i, p);
        };
        auto bucket_function = [&](size_t key) {
            return key;
        };
        countingsort(input, buckets, key_function_count, bucket_function);
        
        // partition input elements by key
        partition_out_of_place(input, result, buckets, key_function_count, 
            bucket_function);
        
        if (p == 0) { break; }
        
        // collect elements
        std::copy(result.begin(), result.end(), input.begin());
    }
}

/**\brief MSD-RadixSort which works in-place
* \tparam S Type of input to be sorted
* \tparam Key Type of a function for the key_function
* \tparam Comp Type of a function for the comp_function
* \param input input to be sorted
* \param radix_size size of the radix, which is the maximum value for a digit
* \param index index of the current digit
* \param max_index maximum count of digits of all elements
* \param key_function function which takes as parameter an element of input and 
        the index of the current digit and returns the key which is used for sorting
* \param comp_function TODO
*
* This function implements MSD-RadixSort in an in-place variant. The function 
* takes a key_function as a parameter so that the developer can specify the key which is 
* used by the sorting algorithm.
*/
template <typename S, typename Key, typename Comp>
void msd_radixsort_inplace_with_key(S& input, const size_t radix_size, const size_t index,
        const size_t max_index, Key key_function, Comp comp_function) {
    msd_radixsort_inplace_with_key_and_no_recursion_on_specific_buckets(input, radix_size,
        index, max_index, radix_size, key_function, comp_function);
}

template <typename S, typename Key, typename Comp>
/**\brief MSD-RadixSort which works in-place and the developer can specify a certain bucket 
* so that this bucket and following buckets are not sorted recursively.
* \tparam S Type of input to be sorted
* \tparam Key Type of a function for the key_function
* \tparam Comp Type of a function for the comp_function
* \param input input to be sorted
* \param result container in which the result of the RadixSort is stored
* \param radix_size size of the radix, which is the maximum value for a digit
* \param index index of the current digit
* \param max_index maximum count of digits of all elements
* \param first_no_rec_bucket number of the first bucket for which there should be no
        recursive call
* \param key_function function which takes as parameter an element of input and 
        the index of the current digit and returns the key which is used for sorting
* \param comp_function TODO
*
* This function implements MSD-RadixSort in an out-of-place variant. The function 
* takes a key_function as a parameter so that the developer can specify the key which is 
* used by the sorting algorithm. Additionaly the developer can specify a certain bucket 
* so that this bucket and following buckets are not sorted recursively. That is useful
* when elements in input have different lengths and the last bucket stores every element 
* whose size is smaller than the current index.
*/
void msd_radixsort_inplace_with_key_and_no_recursion_on_specific_buckets(S& input, const size_t radix_size, const size_t index,
        const size_t max_index, const size_t first_no_rec_bucket, Key key_function, Comp comp_function) {
    const size_t bucket_threshold = 150;           
            
    auto buckets_container = util::container<size_t>(2*radix_size);
    auto buckets_start = buckets_container.slice(0, radix_size);
    auto buckets_end = buckets_container.slice(radix_size, 2*radix_size);
    
    // generate buckets
    auto key_function_count = [&](size_t i) {
        return key_function(i, index);
    };
    auto bucket_function = [&](size_t key) {
        return key;
    };
    countingsort(input, buckets_start, key_function_count, bucket_function);
    
    // initialize end positions of buckets
    std::copy(buckets_start.begin(), buckets_start.end(), buckets_end.begin());
    
    // partition input elements by key
    partition_in_place(input, buckets_start, buckets_end, key_function_count, 
        bucket_function);
        
    auto comp_function_insertionsort = [&](size_t i, size_t j) {
        return comp_function(i, j, index, max_index-index);
    };
    
    // recurse when bucket size > 1 and not all indices are yet sorted
    if (index == max_index) { return; }
    for (size_t i = 0; i < first_no_rec_bucket; ++i) {
        if (buckets_end[i]-buckets_start[i] > 1) {
            auto input_rec = input.slice(buckets_start[i], buckets_end[i]);
            if (buckets_end[i]-buckets_start[i] > bucket_threshold) {
                msd_radixsort_inplace_with_key(input_rec, radix_size,
                    index+1, max_index, key_function, comp_function);
            }
            else {
                util::sort::insertion_sort(input_rec, comp_function_insertionsort);
            }
        }
    }
}

template <typename S, typename Key>
/**\brief MSD-RadixSort with an auxiliary array.
* \tparam S Type of input to be sorted
* \tparam Key Type of a function for the key_function
* \param input input to be sorted
* \param result container in which the result of the RadixSort is stored
* \param radix_size size of the radix, which is the maximum value for a digit
* \param index index of the current digit
* \param max_index maximum count of digits of all elements
* \param key_function function which takes as parameter an element of input and 
        the index of the current digit and returns the key which is used for sorting
*
* This function implements simple MSD-RadixSort in an out-of-place variant. The function 
* takes a key_function as a parameter so that the developer can specify the key which is 
* used by the sorting algorithm.
*/
void msd_radixsort_with_key(S& input, S& result, const size_t radix_size, 
        const size_t index, const size_t max_index, Key key_function) {
    DCHECK_MSG(input.size() == result.size(),
               "input must have the same length as result");
               
    auto buckets = util::container<size_t>(radix_size);
        
    // generate buckets
    auto key_function_count = [&](size_t i) {
        return key_function(i, index);
    };
    auto bucket_function = [&](size_t key) {
        return key;
    };
    countingsort(input, buckets, key_function_count, bucket_function);
    
    // partition input elements by key
    partition_out_of_place(input, result, buckets, key_function_count, 
            bucket_function);
    
    if (index == max_index) { return; }
    
    // collect elements
    std::copy(result.begin(), result.end(), input.begin());
    
    // recurse when bucket size > 1
    for (size_t i = 0; i < buckets.size()-1; ++i) {
        if (buckets[i+1]-buckets[i] > 1) {
            auto input_rec = input.slice(buckets[i], buckets[i+1]);
            auto result_rec = result.slice(buckets[i], buckets[i+1]);
            msd_radixsort_with_key(input_rec, result_rec, radix_size,
                index+1, max_index, key_function);
        }
    }
    if (input.size()-buckets[radix_size] > 1) {
        auto input_rec = input.slice(buckets[radix_size], input.size());
        auto result_rec = result.slice(buckets[radix_size], input.size());
        msd_radixsort_with_key(input_rec, result_rec, radix_size,
            index+1, max_index, key_function);
    }
}

template <typename S, typename B, typename Key_Func, typename Compare_Func>
/**\brief MSD-RadixSort with an auxiliary array.
* \tparam S Type of input to be sorted
* \tparam B Type of bucket array
* \tparam Key_Func Type of a function for the key_function
* \tparam Compare_Func Type of a function for the compare_function
* \param input input to be sorted
* \param bucket_array array which is used for the buckets
* \param alphabet_size size of the alphabet in input
* \param index index of the current digit
* \param max_index maximum count of digits of all elements
* \param key_function function which takes as parameter an element of input and 
        the index of the current digit and returns the key which is used for sorting
* \param key_function function which takes as parameter two elements i and j of input
        and TODO!!
*
* This function implements simple MSD-RadixSort in an out-of-place variant. The function 
* takes a key_function as a parameter so that the developer can specify the key which is 
* used by the sorting algorithm.
*/
void radixsort_for_big_alphabet(S& input, B& bucket_array, const size_t alphabet_size,
        const size_t index, const size_t max_index, Key_Func key_function, 
        Compare_Func compare_function) {
    const size_t bucket_threshold = 550;      
            
    const size_t bucket_size = alphabet_size/5 + (alphabet_size % 5 != 0);  
    
    // Generate buckets
    auto buckets_bucket_start = util::container<size_t>(5);
    auto buckets_bucket_end = util::container<size_t>(5);
    auto key_function_count = [&](size_t i) {
        return key_function(i, index);
    };
    auto bucket_function_bucket = [&](size_t key) {
        return key/bucket_size;
    };
    //tdc::StatPhase phase("BucketSort-CountingSort");
    countingsort(input, buckets_bucket_start, key_function_count, bucket_function_bucket);
    std::copy(buckets_bucket_start.begin(), buckets_bucket_start.end(), buckets_bucket_end.begin());
    
    // Sort Elements into buckets
    //phase.split("BucketSort-Partition");
    partition_in_place(input, buckets_bucket_start, buckets_bucket_end,
        key_function_count, bucket_function_bucket);
        
    // Do a RadixSort for each bucket
    for (uint8_t i = 0; i < 5; ++i) {
        // generate buckets
        auto input_radix = input.slice(buckets_bucket_start[i], buckets_bucket_end[i]);
        auto buckets_radix_start = bucket_array.slice(0, bucket_size);
        auto buckets_radix_end = bucket_array.slice(bucket_size, 2*bucket_size);
        auto bucket_function_radix = [&](size_t key) {
            return key-i*bucket_size;
        };
        //phase.split("RadixSort-CountingSort");
        countingsort(input_radix, buckets_radix_start, key_function_count, bucket_function_radix);
        std::copy(buckets_radix_start.begin(), buckets_radix_start.end(), buckets_radix_end.begin());
        
        // partitioning
        //phase.split("RadixSort-Partition");
        partition_in_place(input_radix, buckets_radix_start, buckets_radix_end,
            key_function_count, bucket_function_radix); 
        
        // reset buckets_radix_end
        for (size_t i = 0; i < buckets_radix_end.size(); ++i) {
            buckets_radix_end[i] = 0;
        }
        
        auto compare_function_insertionsort = [&](size_t i, size_t j) {
            return compare_function(i, j, index, max_index-index);
        };
           
        if (index < max_index) {   
            // recurse when bucket size > 1
            //phase.split("Recursion");
            for (size_t i = 0; i < buckets_radix_start.size()-1; ++i) {
                if (buckets_radix_start[i+1]-buckets_radix_start[i] > 1u) {
                    auto input_rec = input_radix.slice(buckets_radix_start[i], buckets_radix_start[i+1]);
                    if (buckets_radix_start[i+1]-buckets_radix_start[i] > bucket_threshold) {
                        auto bucket_array_rec = bucket_array.slice(bucket_size, bucket_array.size());
                        radixsort_for_big_alphabet(input_rec, bucket_array_rec, alphabet_size,
                            index+1, max_index, key_function, compare_function);
                    }
                    else {
                        util::sort::insertion_sort(input_rec, compare_function_insertionsort);
                    }
                }
            }
            if (input_radix.size()-buckets_radix_start[buckets_radix_start.size()-1] > 1) {
                auto input_rec = input_radix.slice(buckets_radix_start[buckets_radix_start.size()-1], 
                        input_radix.size());
                if (input_radix.size()-buckets_radix_start[buckets_radix_start.size()-1] > bucket_threshold) {
                    auto bucket_array_rec = bucket_array.slice(bucket_size, bucket_array.size());
                    radixsort_for_big_alphabet(input_rec, bucket_array_rec, alphabet_size,
                        index+1, max_index, key_function, compare_function);
                }
                else {
                    util::sort::insertion_sort(input_rec, compare_function_insertionsort);
                }
            }
        }
        
        // reset buckets_radix_start
        for (size_t i = 0; i < buckets_radix_start.size(); ++i) {
            buckets_radix_start[i] = 0;
        }   
    }
}

template <typename S, typename B, typename Key_Func, typename Bucket_Func>
/**\brief Counts the occurences of characters in input and stores the front of the buckets.
* \tparam S Type of input to be sorted
* \tparam B Type of bucket array
* \tparam Key_Func Type of a function for the key_function
* \tparam Bucket_Func Type of a function for the bucket_function
* \param input input for which the buckets are determined
* \param buckets front of the buckets are stored in buckets
* \param key_function function which takes as parameter an element of input and 
*        returns the key which is used for sorting
* \param bucket_function function which takes as parameter a key and 
*        returns the bucket in which the key is placed
*/
void countingsort(S& input, B& buckets, Key_Func key_function, 
        Bucket_Func bucket_function) {
    // count key elements
    for(size_t i = 0; i < input.size(); ++i){
        auto key = key_function(input[i]);
        ++buckets[bucket_function(key)];
    }
    // generate first position of buckets
    size_t sum = 0;
    for(size_t i = 0; i < buckets.size(); ++i){
        sum += buckets[i];
        buckets[i] = sum-buckets[i];
    }
}

template <typename S, typename B, typename Key, typename Bucket_Func>
/**\brief Partitions the elements in input into buckets with an auxiliary array.
* \tparam S Type of input to be sorted
* \tparam B Type of bucket array
* \tparam Key_Func Type of a function for the key_function
* \tparam Bucket_Func Type of a function for the bucket_function
* \param input input to be partitioned
* \param result result of the partitioning
* \param buckets front of the buckets in which the elements are partitioned
* \param key_function function which takes as parameter an element of input and 
*        returns the key which is used for sorting
* \param bucket_function function which takes as parameter a key and 
*        returns the bucket in which the key is placed
*/
void partition_out_of_place(S& input, S& result, B& buckets, Key key_function,
        Bucket_Func bucket_function) {
    for(size_t i = 0; i < input.size(); ++i){
        auto key = key_function(input[i]);
        result[buckets[bucket_function(key)]++] = input[i];
    }
}

template <typename S, typename B, typename Key, typename Bucket_Func>
/**\brief Partitions the elements in input into buckets in-place.
* \tparam S Type of input to be sorted
* \tparam B Type of bucket array
* \tparam Key_Func Type of a function for the key_function
* \tparam Bucket_Func Type of a function for the bucket_function
* \param input input to be partitioned
* \param buckets front of the buckets in which the elements are partitioned
* \param key_function function which takes as parameter an element of input and 
*        returns the key which is used for sorting
* \param bucket_function function which takes as parameter a key and 
*        returns the bucket in which the key is placed
*/
void partition_in_place(S& input, B& buckets_start, B& buckets_end, Key key_function,
        Bucket_Func bucket_function) { 
    size_t i = 0;
    size_t current_bucket = 0;
    while(i < input.size()) {
        // swap current element with first free element of bucket
        auto key = key_function(input[i]);
        auto bucket = bucket_function(key);
           
        bool swapped = false;
        if (i != buckets_end[bucket]) {
            auto tmp = input[buckets_end[bucket]];
            input[buckets_end[bucket]] = input[i];
            input[i] = tmp;
            swapped = true;
        }
        ++buckets_end[bucket];
        
        if (!swapped) {
            if (i == input.size()-1 || current_bucket == buckets_start.size()-1 || i+1 < buckets_start[current_bucket+1]) { 
                ++i; 
            }
            else {
                while (current_bucket < buckets_start.size()-1 && i+1 >= buckets_start[current_bucket+1]) {
                    //std::cout << "ee" << std::endl;
                    i = buckets_end[current_bucket+1];
                    ++current_bucket;
                    if (current_bucket < buckets_start.size()-1 && i < buckets_start[current_bucket+1]) { break; }
                }
            }
        }
    }
}
} // namespace sacabench::util
