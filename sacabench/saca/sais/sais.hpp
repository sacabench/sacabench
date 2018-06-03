/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/signed_size_type.hpp>

namespace sacabench::saca::sais {
using namespace sacabench::util;

class Sais {
    public:        
        static const size_t L_Type = 0;
        static const size_t S_Type = 1;

        template<typename T>
        static T get_char_value(int position, unsigned char* s) {
            return ((T*)s)[position];
        }

        template<typename T>
        static void compute_types(size_t* t, size_t* p_1, unsigned char* s, size_t len) {
            t[len - 1] = S_Type;

            for (ssize i = len - 2; i >= 0; i--) {
                if (get_char_value<T>(i + 1, s) < get_char_value<T>(i, s)) {
                    t[i] = L_Type;
                    if (t[i + 1] == S_Type) { p_1[i + 1] = 1; }
                }
                else if (get_char_value<T>(i + 1, s) > get_char_value<T>(i, s)) { t[i] = S_Type; }
                else { t[i] = t[i + 1]; }
            }
        }

        template<typename T>
        static void generate_buckets(unsigned char* s, size_t len, size_t* buckets, size_t K, bool end) {
            size_t sum = 0;

            for (size_t i = 0; i <= K; i++) { buckets[i] = 0; }

            // bucket size for each char
            for (size_t i = 0; i < len; i++) { buckets[get_char_value<T>(i, s)]++; }

            // sum up to bucket ends
            for (size_t i = 0; i <= K; i++) {
                sum += buckets[i];
                buckets[i] = (end ? sum : sum - buckets[i]);
            }
        }

        template<typename T>
        static void induce_L_Types(unsigned char* s, size_t len, size_t *buckets, size_t K, bool end, ssize* SA, size_t* t) {
            generate_buckets<T>(s, len, buckets, K, end);
            for (size_t i = 0; i < len; i++) {
                ssize pre_index = SA[i] - 1; // pre index of the ith suffix array position

                if (pre_index >= 0 && t[pre_index] == L_Type) { // pre index is type L
                    SA[buckets[get_char_value<T>(pre_index, s)]++] = pre_index; // "sort" index in the bucket
                }
            }
        }

        template<typename T>
        static void induce_S_Types(unsigned char* s, size_t len, size_t *buckets, size_t K, bool end, ssize* SA, size_t *t) {
            generate_buckets<T>(s, len, buckets, K, end);
            for (ssize i = len - 1; i >= 0; i--) {
                ssize pre_index = SA[i] - 1;

                if (pre_index >= 0 && t[pre_index] == S_Type) {
                    SA[--buckets[get_char_value<T>(pre_index, s)]] = pre_index;
                }
            }
        }

        template<typename T>
        static void SAIS(unsigned char* s, ssize* SA, size_t len, ssize K) {
            size_t* t = new size_t[len]();
            size_t* p_1 = new size_t[len]();
            size_t *buckets = new size_t[K + 1]();

            compute_types<T>(t, p_1, s, len);

            generate_buckets<T>(s, len, buckets, K, true);
            // Initialize each entry in SA with -1
            for (size_t i = 0; i < len; i++) { SA[i] = -1; }
            // iterate from left to right (starting by 1 cause 0 can never be LMS) and put 
            // LMS to end of the bucket and move bucket's tail backwards
            for (size_t i = 1; i < len; i++) {
                if (p_1[i] == 1) {
                    SA[--buckets[get_char_value<T>(i, s)]] = i;
                }
            }

            // sort LMS substrings
            induce_L_Types<T>(s, len, buckets, K, false, SA, t);
            induce_S_Types<T>(s, len, buckets, K, true, SA, t);

            // because we have at most n/2 LMS, we can store the sorted indices in the first half of the SA
            ssize n1 = 0;
            for (size_t i = 0; i < len; i++) {
                if (p_1[SA[i]] == 1) {
                    SA[n1++] = SA[i];
                }
            }

            // All LMS are now stored in the (at most) first half of the SA, so the rest half of the suffix array can be used 
            for (size_t i = n1; i < len; i++) { SA[i] = -1; }

            // The given names correspond to the buckets the LMS are sorted ssizeo. 
            // To find the names, the strings have to be compared by its type and lexicographical value per char
            ssize name = 0;
            ssize previous_LMS = -1;
            for (ssize i = 0; i < n1; i++) { // max n/2 iterations
                bool diff = false;
                // compare types and chars
                ssize current_LMS = SA[i];
                for (size_t j = 0; j < len; j++) {
                    if (previous_LMS == -1 || get_char_value<T>(current_LMS + j, s) != get_char_value<T>(previous_LMS + j, s) || 
                        t[current_LMS + j] != t[previous_LMS + j]) {
                        diff = true;
                        break;
                    }
                    else if (j > 0 && (p_1[current_LMS + j] == 1 || p_1[previous_LMS + j] == 1)) { // check if next LMS is reached
                        break; // no diff was found
                    }
                }

                if (diff) { // if diff was found, adjust the name and continue with current LMS as previous
                    name++;
                    previous_LMS = current_LMS;
                }

                current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2 : (current_LMS - 1) / 2;
                SA[n1 + current_LMS] = name - 1;
            }

            // not needed
            for (ssize i = len - 1, j = len - 1; i >= n1; i--) {
                if (SA[i] >= 0) {
                    SA[j--] = SA[i];
                }
            }

            // if there are more as less names than LMS, we have duplicates and have to recurse
            ssize* SA1 = SA;
            ssize* s1 = (SA + len - n1);
            if (name < n1) {
                // new string that consists of ssizeegers now, as they represent the names
                SAIS<size_t>((unsigned char*)s1, SA1, n1, name - 1);
            }
            else {
                for (ssize i = 0; i < n1; i++) {
                    SA1[s1[i]] = i;
                }
            }

            // induce the final SA
            generate_buckets<T>(s, len, buckets, K, true);
            size_t j;
            for (size_t i = 1, j = 0; i < len; i++) {
                if (p_1[i] == 1) {
                    s1[j++] = i;
                }
            }
            for (ssize i = 0; i < n1; i++) {
                SA1[i] = s1[SA1[i]];
            }
            for (size_t i = n1; i < len; i++) {
                SA[i] = -1;
            }
            for (ssize i = n1 - 1; i >= 0; i--) {
                j = SA[i];
                SA[i] = -1;
                SA[--buckets[get_char_value<T>(j, s)]] = j;
            }

            induce_L_Types<T>(s, len, buckets, K, false, SA, t);
            induce_S_Types<T>(s, len, buckets, K, true, SA, t);

            delete[] buckets;
            delete[] t;
            delete[] p_1;
        }
 
	    template<typename sa_index>
	    static void construct_sa(util::string_span text, size_t alphabet_size, util::span<sa_index> out_sa) {
	        
	        // obsolete once the null-byte is appended by the framework
            string copy = string(text.size() + 1);
            for (size_t i = 0; i < text.size(); i++) { copy[i] = text[i]; }
            copy[text.size()] = 0;
            
            size_t len = copy.size();
            ssize* SA = new ssize[len]();	        
	        
            SAIS<unsigned char>((unsigned char*)copy.data(), SA, len, alphabet_size);
            
            for (size_t i = 1; i < len; i++) { out_sa[i - 1] = SA[i]; } 
        }   
    };
} 

