/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>

namespace sacabench::util::sort {

    template <typename index_type>
        struct bucket {
            std::size_t count = 0;
            index_type position = 0;
        };

    template <typename index_type>
        void bucketsort_presort(const string& input, const std::size_t alphabet_size,
                const std::size_t depth, container<index_type>& sa) {
            const std::size_t length = input.size();
            // the real alphabet includes $, so it has one more character
            const std::size_t real_alphabet_size = alphabet_size + 1;
            const std::size_t bucket_count = pow(real_alphabet_size, depth);
            container<bucket<index_type>> buckets =
                make_container<bucket<index_type>>(bucket_count);

            // calculate code pre first suffix
            std::size_t initial_code = 0;
            for (index_type index = 0; index < depth - 1; ++index) {
                initial_code *= (real_alphabet_size);
                initial_code += input[index];
            }

            // calculate modulo for code induction
            const std::size_t code_modulo = pow(real_alphabet_size, depth - 1);

            // count occurrences
            std::size_t code = initial_code;
            std::cout << "Initial Code: " << code << std::endl;
            std::cout << "Code Modulo:  " << code_modulo << std::endl;
            for (index_type index = 0; index < length - depth + 1; ++index) {
                // induce code for nth suffix from (n-1)th suffix
                std::cout << "Code before modulo: " << code << "\t";
                code %= code_modulo;
                std::cout << "Code before multiplication: " << code << "\t";
                code *= real_alphabet_size;
                std::cout << "Code before adding: " << code << "\t";
                std::cout << "Index: " << index + depth - 1 << "\t";
                code += input[index + depth - 1];
                std::cout << "Code after adding: " << code << std::endl;
                ++buckets[code].count;
            }

            // same as above, but for buckets containing at least one $
            for (index_type index = length - depth + 1; index < length;
                    ++index) {
                // induce code for nth suffix from (n-1)th suffix
                std::cout << "Code before modulo: " << code << "\t";
                code %= code_modulo;
                std::cout << "Code before multiplication: " << code << "\t";
                code *= real_alphabet_size;
                std::cout << "Code before adding: " << code << "\t";
                std::cout << "Index: " << index + depth - 1 << "\t";
                std::cout << "Code after adding: " << code << std::endl;
                ++buckets[code].count;
            }
            std::cout << std::endl;

            // calculate positions
            std::cout << "Bucket 0 has position " << (uint32_t) buckets[0].position << " and size " << buckets[0].count << std::endl;
            for (size_t index = 1; index < bucket_count; ++index) {
                buckets[index].position =
                    buckets[index - 1].position + buckets[index-1].count;
                std::cout << "Bucket " << (uint32_t) index << " has position " << (uint32_t) buckets[index].position << " and size " << buckets[index].count << std::endl;
            }
            std::cout << std::endl;

            // insert entries in suffix array
            code = initial_code;
            for (index_type index = 0; index < length - depth + 1; ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                code += input[index + depth - 1];
                std::cout << "Code " << code << " at index " << (uint32_t) index << "\t";
                bucket<index_type>& current_bucket = buckets[code];
                std::cout << "Bucket position " << (uint32_t) current_bucket.position << "\t";
                sa[current_bucket.position] = index;
                std::cout << "Set sa[" << (uint32_t) current_bucket.position << "] to " << (uint32_t) index << std::endl;
                ++current_bucket.position;
            }

            // same as above, but for entries containing at least one $
            for (index_type index = length - depth + 1; index < length;
                    ++index) {
                // induce code for nth suffix from (n-1)th suffix
                code %= code_modulo;
                code *= real_alphabet_size;
                std::cout << "Code " << code << " at index " << (uint32_t) index << "\t";
                bucket<index_type>& current_bucket = buckets[code];
                std::cout << "Bucket position " << (uint32_t) current_bucket.position << "\t";
                sa[current_bucket.position] = index;
                std::cout << "Set sa[" << (uint32_t) current_bucket.position << "] to " << (uint32_t) index << std::endl;
                ++current_bucket.position;
            }
            std::cout << std::endl;

            // free memory
            //delete[] buckets;
        }


    void print_result(container<string> strings) {
        std::cout << "Result: ";
        for (string currentString : strings) {
            for (int index = 0; index < currentString.size(); ++index) {
                std::cout << currentString.at(index);
            }
            std::cout << ", ";
        }
        std::cout << std::endl;
    }

    void bucket_sort(const container<string>& strings,
                     const int currentDepth,
                     const int maxDepth,
                     container<container<string>>& buckets) {

        std::cout << "Started iteration of bucket sort." << std::endl;
        std::cout << "Number of input strings: " << strings.size() << std::endl;
        std::cout << "Current depth: " << currentDepth << std::endl;
        std::cout << "Max depth: " << maxDepth << std::endl;

        // check end of recursion
        if (currentDepth == maxDepth) {
            print_result(strings);
            return;
        }
        for (string currentString : strings) {
            if (currentString.size() < currentDepth) {
                print_result(strings);
                return;
            }
        }

        // build new buckets
        for (string currentString : strings) {

            std::cout << "For string in strings with current string: ";
            for (int index = 0; index < currentString.size(); ++index) {
                std::cout << currentString.at(index);
            }
            std::cout << std::endl;

            bool bucketFound = false;
            char currentChar = currentString.at(currentDepth);

            std::cout << "Current char: " << currentChar << std::endl;

            // Add new bucket, if it is the first one.
            if (buckets.empty()) {

                std::cout << "There are no buckets yet." << std::endl;

                container<string> newBucket;
                newBucket.push_back(currentString);
                buckets.push_back(newBucket);
                bucketFound = true;
                continue;
            }

            // Check each bucket to get one with the current key.
            for (int index = 0; index < buckets.size(); ++index) {
                container<string> bucket = buckets.at(index);
                string firstStringInBucket = bucket.front();
                char sortingKeyOfBucket = firstStringInBucket.at(currentDepth);
                if (sortingKeyOfBucket == currentChar) {
                    bucket.push_back(currentString);
                    bucketFound = true;
                    buckets.erase(buckets.begin() + index);
                    buckets.insert(buckets.begin() + index, bucket);
                }
            }
            // There is no bucket with this key yet, add a new one.
            if (!bucketFound) {
                container<string> newBucket;
                newBucket.push_back(currentString);
                buckets.push_back(newBucket);
            }
        }

        // new recursion
        for (container<string> bucket : buckets) {
            bucket_sort(bucket, currentDepth + 1, maxDepth, buckets);
        }
    }

}
